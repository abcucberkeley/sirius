# SIRIUS compute worker (`sirius_worker`)

A small TCP service that runs the parts of a pipeline that live in Python:
Torch segmentation models, and -- when it runs on a cluster node -- every
step the Python step library implements, which is how the application's
**HPC** backend works. The application launches one locally for Torch
models (`python -m sirius_worker --port 0`), reads the port it announces
and talks to it over the protocol below; on a cluster the same worker is a
Slurm job reached through an SSH tunnel (see `slurm/README.md`).

No third-party dependency is needed for the service itself: the standard
library plus `numpy`. `torch` enables `model_info` / `torch_segment` /
`seg`, `scipy` the label post-processing and resampling, the `sirius`
package SIM reconstruction, `huggingface_hub` the model hub methods,
`onnxruntime` ONNX models, and `cellpose` / `micro_sam` the model
families (see *Segmentation models* below).

```
python -m sirius_worker [--host 127.0.0.1] [--port 0] [--token T] [--device auto|cuda|cpu]
                        [--allow-install] [--log-level INFO]
```

Once listening it prints exactly one JSON line to stdout,
`{"port": 41237, "pid": 12345, "host": "127.0.0.1", "device": "cuda"}`, and
logs to stderr. `--token` (or `$SIRIUS_TOKEN`) is a shared secret the client
must present in `hello`; always set one on a shared machine. **Binding
anything but a loopback address without a token is refused at startup** (the
worker says so and exits 2): reaching the port is the whole of the
authorisation model, and whoever completes the handshake can run code as the
user who started the worker. Loopback without a token still works, with a
warning. `--allow-install` opts into the `install` method. See `SECURITY.md`
next to this file.

## Where the step code lives

There is one implementation of the steps: `bindings/python/sirius/workbench.py`
(`sirius.workbench`). The worker uses the installed `sirius` package when
there is one; otherwise `sirius_worker/steps.py` loads that file directly
-- from `$SIRIUS_WORKBENCH_PY`, from a checkout or build tree found by
walking up from the worker directory (`build/<preset>/app/python` reaches
the repository root), or from a `workbench.py` copied next to the package.
Loaded that way there is no `sirius` extension, so SIM reconstruction
reports itself unavailable while the numpy and Torch steps work.

`sirius.workbench.run_pipeline(dataset_path, pipeline_json)` is also what
the application's "Export pipeline as Python script" calls.

## Protocol

Mirrors `app/core/rpc.hpp`. One frame:

```
u32 header_len (LE) | header: UTF-8 JSON | u64 payload_len (LE) | payload
```

Header fields: `id` (request id, echoed by every reply), `type`
(`request` | `progress` | `result` | `error`), `method`, `params`,
`tensors` (`[{name, dtype, shape, offset, nbytes}]` describing raw
little-endian C-order arrays concatenated in the payload), `message`,
`fraction`. dtypes: `float32 float64 uint8 int8 uint16 int16 uint32 int32
uint64 int64`.

Every length in a frame comes from the peer, so both ends bound them before
using them: a header is at most 64 MiB, a payload at most 32 GiB, and a peer
that has not completed `hello` is held to 16 KiB per frame — checked against
the length prefix, before the bytes it announces are read. A tensor's
`offset`/`nbytes` must fit the payload and match the product of its shape,
which is itself bounded as it is computed.

`hello` also agrees the protocol version: `PROTOCOL_VERSION` in
`sirius_worker/protocol.py` and `kProtocolVersion` in `app/core/rpc.hpp`,
currently `1`. Both ends must send the same number — a peer that sends none
counts as version 0 — and a mismatch is refused with a message naming both
versions and which end to update.

| method | params | reply |
| --- | --- | --- |
| `hello` | `{token, protocol_version}` | `result`: `{version, protocol_version, methods, cuda, device, hostname, python, torch, sirius, workbench}`; `error` on a bad token or another protocol version, and the connection is closed |
| `ping` | | `result`: `{time}` |
| `model_info` | `{spec}` (or `path`) | `result`: `{format, input_shape, output_shape, dtype, size_bytes, channels_out}` for a file; `{format: "cellpose" \| "micro-sam", available, install_hint, returns: "labels"}` for a model family; `{format: "hf", cached: false, repo, file}` for an `hf:` file not downloaded yet |
| `hub_search` | `{query, limit?, filter?, token?}` | `result`: `{models: [{id, downloads, likes, tags, last_modified, pipeline_tag, library, gated, private}]}` (Hugging Face, sorted by downloads; `gated` is `"manual"` / `"auto"` for repositories whose terms must be accepted, else `false`) |
| `hub_files` | `{repo}` | `result`: `{repo, files: [{name, size, model}]}` (`model`: a `.pt` / `.pts` / `.pth` / `.onnx`) |
| `hub_download` | `{repo, file?, token?}` | `progress`\* then `result`: `{path, bytes, spec}`; cancellable like a run; without `file` the repository's single model file |
| `install` | `{family, dry_run?}` | `progress`\* (one frame per output line) then `result`: `{ok, returncode, available, command, installer, tail}`; runs the family's install command (`pip install cellpose`; `conda install -c conda-forge micro_sam` when the interpreter lives in a conda environment, pip otherwise) in the worker's own Python; cancellable |
| `model_prepare` | `{spec, token?}` | `progress`\* then `result`: `{spec, path, cached}`; fetches a family model's weights (or an `hf:` file) now instead of on the first run |
| `models_list` | | `result`: `{cache, models: [{spec, path, bytes, repo, file}]}` -- the local model cache |
| `models_delete` | `{path}` | `result`: `{path, bytes, removed_directories}`; removes one file or one repository directory from the model cache, and only from there -- a path anywhere else is refused, since a model can be named from outside the cache and that file is the user's own |
| `run` | `{kind, params, meta?}` + tensors | `progress`\* (`{fraction, message}`) then `result` + tensors, or `error` |
| `cancel` | `{id}` | `result`: `{cancelled: id}`; the cancelled run replies `error` `"cancelled"` |
| `shutdown` | | `result` `{}` and the worker exits |

`methods` lists `run:<kind>` for every kind the worker can run, so the
application knows what to route. One run executes at a time (a second
`run` gets `error "busy"`); the reader keeps running so `cancel` is
honoured between tiles / volumes. Every `result` carries `seconds`. No method
but `hello` is served before a successful `hello`, and `install` and
`shutdown` are logged with the peer's address as privileged requests.

### Run kinds and tensors

| kind | input tensors | output tensors | result |
| --- | --- | --- | --- |
| `torch_segment` | `input` (z, y, x) float32 | `prob` (C, z, y, x) float32 -- or, for a model family, `labels` (z, y, x) uint32 and optionally `prob` (1, z, y, x) | `{channels, device}` / `{labels, format, model, device}` |
| `sim` | `input` (sections, y, x) or (c, t, sections, y, x) float32 | `output` (same rank, zoomed) | `{meta, info: {fits, wiener, ...}}` |
| `skimage_seg` | `input` (z, y, x) float32 | `labels` (z, y, x) uint32 | `{method, seeds?, labels, note?, device}`; one frame at a time, so the ids start again at 1 in each |
| `btrack` | `labels` (t, z, y, x) or (t, y, x) uint32 | `labels` (same shape) renumbered by track | `{tracks, objects, divisions, mean_length, longest}` |
| any other kind | `input` (c, t, z, y, x) float32, optional `labels` (t, z, y, x) uint32 | `output` (c', t', z', y', x') float32, optional `labels`, `prob` | `{meta, info}` |

`meta` in `params`/results is the dataset metadata dict of `sirius.workbench`
(`dims`, `voxel_um` [x, y, z], `channels`, `rgb`, `sim`).

### Step parameters

`params` are the step's parameters exactly as the application saves them:
every key below is the `key` of a `ParamSpec` in the matching
`app/core/ops/*.cpp`, and the defaults are the C++ defaults. A few older
Python-only spellings are still accepted as aliases (see the docstrings in
`workbench.py`), but the canonical key always wins; **any key that is
neither is reported through an `UnknownParameterWarning` naming the step**
instead of being ignored. `bindings/python/sirius/op_schema.json` is a
snapshot of the C++ parameter tables and
`bindings/tests/test_workbench_schema.py` fails if the two drift apart.

* `einsum`: `keep` (the axes that survive, e.g. `czyx`), `reduction`
  (`sum` | `mean` | `max` | `min`). `maxproj`: `axis` (`z` | `t` | `c`).
  `meant`: no parameters.
* `contrast`: `min`, `max` (the manual window; `max <= min` means automatic,
  from `lo_percentile` / `hi_percentile`), `gamma`, `lo_percentile` (0.2),
  `hi_percentile` (99.8), `bake`. The window is taken once over the whole
  input, not per channel.
* `flatfield`: `flat`, `dark` (TIFF paths; one page, or one page per channel).
* `bleach`: `mode` (`Match first frame` | `Match mean`), `over` (`t` | `z`).
* `croppad`: `z0`, `y0`, `x0` (origin, may be negative = pad), `z`, `y`, `x`
  (size, 0 = to the edge), `fill`. Labels are cropped with the intensities.
* `resample`: `voxel_x`, `voxel_y`, `voxel_z` (µm, 0 = keep that axis),
  `interpolation` (`linear` | `cubic` | `nearest`).
* `merge`: `blend` (`Additive` | `Screen` | `Max`), `colors` (`#rrggbb` per
  channel, empty = the channels' own colours), `weights` (per-channel gain),
  `normalize_percentile` (99.9).
* `threshold`: `channel`, `method` (`Manual` | `Otsu` | `Percentile`),
  `value` (the manual cut), `percentile` (90), `post`
  (`Connected components` | `Watershed (distance)`), `min_voxels` (20),
  `seed_distance` (5), `class_name`.
* `classic` (classical segmentation): `channel`, `tophat` (white top-hat
  radius, 0 = off), `sigma`, `method` (`Otsu` | `Manual` | `Percentile` |
  `Local mean`), `value`, `percentile`, `window`, `local_ratio`,
  `local_offset`, `opening`, `fill_holes`, `post`, `seed_distance` (8),
  `min_voxels`, `class_name`.
* `cleanup` (label cleanup, needs the labels of a segmentation step
  upstream): `min_voxels` (50), `remove_border`, `relabel`, `low_conf`,
  `size_outlier_factor` -- the last two only set review flags, reported in
  `info["flags"]`.
* `seg` (the application's Torch segmentation, labels out): `model` (a model
  spec, below), `input_channel`, `tile` [z, y, x], `overlap`, `threshold`,
  `post` (`Watershed on boundary channel` | `Connected components` |
  `None (raw probabilities)`), `min_voxels`, `label_opacity`, `class_name`,
  `seed_distance`, plus the inference-only keys `normalize` (percentile
  1..99.9 → 0..1, default true), `activation` (`auto` | `sigmoid` |
  `softmax` | `none`), `pad_to`, `fg_channel`, `boundary_channel`, and the
  model-family keys `diameter`, `do_3d`, `anisotropy`, `flow_threshold`,
  `cellprob_threshold`, `stitch_threshold` (Cellpose), `mode`, `amg`,
  `checkpoint` (micro-SAM).
* `torch_segment` (probabilities out, no labels): `model`, `tile`,
  `overlap`, `normalize`, `activation`, `pad_to` and the model-family keys
  above. TorchScript models take (1, 1, z, y, x) float32 and return
  (1, C, z, y, x); ONNX runs through `onnxruntime`.
* `sim`: `mode` (`Estimate` | `Manual` | `From file`), `params_file` (TOML
  or a legacy cudasirecon config, loaded first), `angles`, `phases`,
  `wiener`, `apodization` (`Cosine` | `Triangle` | `None`), `otf`, `na`,
  `nimm`, `wavelength_nm`, `linespacing_um`, `k0_angles`, `k0_start_angle`,
  `suppress_zero_order`, `bleach_correction`,
  `zoomfact`, `z_zoom`, `orders`, `dz_psf`, `otfcutoff`, `background`,
  `apodize_input`, `napodize`, `suppression_radius`,
  `suppress_singularities`, `no_kz0`, `filter_overlaps`, `explodefact`,
  `equalizez`; `dx`, `dy`, `dz` override the voxel size of `meta`. `otf` is
  required here -- the theoretical OTF exists only in the application.
* `load`: `path`, `read_as`, `tile`, `page_order`, `c`, `t`, `z`,
  `voxel_x`, `voxel_y`, `voxel_z`, `sim_ndirs`, `sim_nphases`, `sim_fast`,
  `sheet_angle`. `run_pipeline` reads the dataset itself, so a `load` step
  in a pipeline only overrides the metadata.

Kinds the Python side does not implement (`decon`, `deskew`, `volrec`,
`stitch`, `register`) are reported as unsupported; the application runs
those natively. A step whose parameters ask for something numpy/scipy
cannot do (label post-processing without `scipy`, SIM without the `sirius`
extension) raises `NotAvailable` naming the missing package rather than
silently computing something else.

## Tracking

`run {kind: "btrack"}` takes a `labels` tensor of shape (t, z, y, x) uint32 and
returns it renumbered by track, with `{tracks, objects, divisions, mean_length,
longest}`. It needs [btrack](https://github.com/quantumjot/btrack) (MIT,
`pip install btrack`), whose tracking core is a compiled C++ library shipped in
the wheel and whose lineage step is an integer program solved with cvxopt /
GLPK. `sirius_worker.tracking.available()` reports whether it is installed *and*
whether its library loads: the wheel is built against a newer libstdc++ than
some conda environments carry, and that only shows up on load.

## Segmentation models

The `model` of `torch_segment` / `seg` (the application's Torch
segmentation step, whose **Hub…** button opens a browser for all of these)
is a *spec* resolved by `sirius_worker/models.py`:

| spec | what runs | needs |
| --- | --- | --- |
| `/path/model.pt` (`.pts`, `.pth`, `.onnx`) | the file, tile-wise, probabilities out | `torch` (`onnxruntime` for ONNX) |
| `hf:<owner>/<repo>[:<file>]` | the file downloaded once from Hugging Face into the cache; without `<file>` the repository must hold exactly one model file | `huggingface_hub` |
| `cellpose:<model>` -- `default` (the installed version's built-in model: `cpsam` on Cellpose 4, `cyto3` on Cellpose 3), one of `cellpose.models.MODEL_NAMES`, or a path / `hf:` spec of a custom Cellpose model | `cellpose.models.CellposeModel`, 3D through `do_3D` or per-plane stitching; instance labels plus the cell probability | `pip install cellpose` |
| `microsam:<model_type>` -- `vit_b_lm`, `vit_l_lm`, `vit_t_lm`, `vit_b_em_organelles`, ... | micro-SAM's automatic instance segmentation, per plane or with its 3D linking; instance labels out | `conda install -c conda-forge micro_sam` |

The cache is `$SIRIUS_MODEL_CACHE` or `~/.sirius/models`
(`hf/<owner>--<repo>/<file>` for downloads); `models_list` reports it. A
missing package is reported by `model_info` (`available: false` with an
`install_hint`, `install` = the exact command) and by `run` as a `NotAvailable` error naming the `pip
install`; the worker itself starts without any of them. The `install` method runs that command on request (the
application asks first), and `model_prepare` fetches a model's weights ahead of the first run. Cellpose 4 keeps only
its own built-in models: a Cellpose 3 name such as `cyto3` is refused with the installed version's list rather than
silently mapped to another model. Gated Hugging Face repositories need an access token: `HF_TOKEN` in the worker's
environment, a `token` in the hub calls, or `huggingface-cli login`. The model
families skip the application's threshold / watershed stage: their labels
go straight into the label volume (`min_voxels` still applies, and a
probability map, when the family provides one, gives the per-label
confidence). `sirius.workbench.load_model` / `model_info` accept the same
specs.

## Tests

```
python -m unittest discover -s app/python/tests -v      # protocol, socket, torch (skipped without torch)
python -m unittest app/python/tests/test_models.py -v   # model specs, cache, hub methods (Hub calls skipped offline)
python -m unittest discover -s bindings/tests -p "test_workbench*.py" -v   # run_pipeline / steps / key drift
```

`bindings/tests/test_workbench_schema.py` compares the steps' declared keys,
defaults and choices with `bindings/python/sirius/op_schema.json`, the
snapshot of the C++ parameter tables. Regenerate the snapshot whenever a
parameter is added or renamed in `app/core/ops`:

```
cmake --build build/<preset> --config Debug --target sirius_tests
SIRIUS_OP_SCHEMA_OUT=bindings/python/sirius/op_schema.json     build/<preset>/tests/Debug/sirius_tests.exe "[schema]"
```

Without the environment variable the same case (`tests/test_app_schema.cpp`)
only checks that the export is well formed; the test is skipped on the Python
side when the snapshot is missing.
