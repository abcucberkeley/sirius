# SIRIUS compute worker (`sirius_worker`)

A small TCP service that runs the parts of a pipeline that live in Python:
Torch segmentation models, and -- when it runs on a cluster node -- every
step the Python step library implements, which is how the application's
**HPC** backend works. `sirius-app` and `sirius-cli` launch one locally
(`python -m sirius_worker --port 0 --exit-with-parent`), in SIRIUS's own
Python environment unless an interpreter is named (below), read the port it
announces and talk to it over the protocol below. `sirius-app` starts it
shortly after its window opens, to load the user operations (at once when it
is given a pipeline); `sirius-cli` the first time a step or the user
operations need it. On a cluster the same worker is a Slurm job reached
through an SSH tunnel (see `slurm/README.md`).

No third-party dependency is needed for the service itself: the standard
library plus `numpy`, which `requirements.txt` lists
(`requirements-extra.txt` adds the optional scipy and scikit-image). `torch`
enables `model_info` / `torch_segment` /
`seg`, `scipy` the label post-processing and resampling, the `sirius`
package SIM reconstruction, `huggingface_hub` the model hub methods,
`onnxruntime` ONNX models, and `cellpose` / `micro_sam` the model
families (see *Segmentation models* below).

```
python -m sirius_worker [--host 127.0.0.1] [--port 0] [--token-file F] [--device auto|cpu|cuda|cuda:N]
                        [--allow-install] [--exit-with-parent] [--max-clients N] [--idle-timeout S]
                        [--log-level INFO]
python -m sirius_worker --check
```

Once listening it prints exactly one JSON line to stdout,
`{"port": 41237, "pid": 12345, "host": "127.0.0.1", "hostname": "n042", "device": "cuda"}`,
and logs to stderr. The token is a shared secret that the client and the
worker each prove they know in the handshake, without sending it: give it in
a file only you can read (`--token-file` or `$SIRIUS_TOKEN_FILE`; the worker
deletes it once read) or in `$SIRIUS_TOKEN`. `--token T` still works but
warns, since a command line is visible to every user of the machine. Always
set one on a shared machine. **Binding
anything but a loopback address without a token is refused at startup** (the
worker says so and exits 2): reaching the port is the whole of the
authorisation model, and whoever completes the handshake can run code as the
user who started the worker. Loopback without a token still works, with a
warning. `--allow-install` opts into the `install` method: `sirius-app`
passes it for the worker it starts on your machine (the model hub installs
model packages through it), while `sirius-cli` never does, and neither does
the Slurm script. `--exit-with-parent` makes the worker stop when its stdin
reaches end-of-file; both executables pass it, so a parent that crashed or was
killed leaves no worker behind. See `SECURITY.md` next to this file.

## What it needs to start

`requirements.txt` is what the worker cannot start without (numpy);
`requirements-extra.txt` is optional (scipy for label clean-up and
resampling, scikit-image for the foundation step's watershed in `latents`
and some plugins). Nothing is pinned. `sirius_worker/__init__.py` holds the same
knowledge as import name → distribution tables: `REQUIRED` (`numpy`) and
`OPTIONAL` (`scipy`, `skimage` → `scikit-image`, `torch`, `huggingface_hub`,
`onnxruntime`, `cellpose`, `micro_sam`, `btrack`). The C++ side mirrors
`OPTIONAL` in `pyenv::optionalDistributions()`, a test keeps the two in step,
and it is the only list an agent may ask `sirius-cli`'s `setup_worker_env`
to install from.

**The start check.** Before it imports anything that needs numpy,
`__main__` looks for every `REQUIRED` module. When one is missing it prints
one line on stdout, where the port line would have been:

```
{"error": "missing_packages", "missing": ["numpy"], "python": "/usr/bin/python3", "version": "3.12.3"}
```

logs `missing packages: numpy (not installed in <python>)` on stderr, and
**exits with code 3**. The launcher (`app/core/local_worker.cpp`) reads that
line and reports "the Python worker cannot start: numpy is not installed in
<python> (found on PATH)", followed by the host's next step: `sirius-app`
offers to set up its own environment, and `sirius-cli` answers
`worker_unavailable` (exit 4) with the hint to run
`sirius-cli worker setup --yes`.

**`--check`** prints one JSON line and exits 0, without importing numpy:

```
{"python": "...", "executable": "...", "base_executable": "...", "version": "3.14.5", "venv": true,
 "externally_managed": false, "pip": true, "ensurepip": true, "free_threaded": false, "bits": 64,
 "missing": [], "optional": {...}, "packages": {...}, "worker": "0.1.0"}
```

`missing` lists the absent `REQUIRED` modules, `optional` the installed
version of each optional package (or `null`), `packages` the version of each
required, optional and `pip` distribution that is installed, and `worker`
the worker's own version. Setting up SIRIUS's
environment ends with this check, and so does *Check* in Preferences.

## Which Python runs it

The first of these that applies:

| # | source | where it comes from |
| --- | --- | --- |
| 1 | explicit | `sirius-cli --python <exe>` |
| 2 | environment | `$SIRIUS_PYTHON` |
| 3 | configured | the Python field in `sirius-app`'s Preferences ▸ Compute (`sirius-cli` does not read the GUI's settings) |
| 4 | managed | SIRIUS's own environment, when its marker and its interpreter exist |
| 5 | discovered | the first Python on PATH (`python3`, `python`, on Windows the newest `python3.N.exe`; never the Microsoft Store alias, and on macOS `/usr/bin/python3` only when the Command Line Tools are installed) |
| 6 | fallback | `python` on Windows, `python3` elsewhere |

The managed environment is chosen even when it needs an update or a repair,
so that you are offered that instead of a silent fall-back to a Python
without numpy. When the worker runs in it, `PYTHONHOME` is emptied in the
worker's environment, so a stray value cannot break the venv. `sirius-cli
worker status` shows the choice and its source.

## SIRIUS's own Python environment

A virtual environment the application makes for the worker, from a Python
already on the machine:

| system | directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%/sirius/python-env` (its interpreter: `Scripts/python.exe`) |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/sirius/python-env` (`bin/python`) |
| macOS | `~/Library/Application Support/sirius/python-env` (`bin/python`) |

`$SIRIUS_PYTHON_ENV` names another directory, for a home quota on a cluster
or a scratch directory in tests. A venv holds absolute paths, so the
directory is not meant to roam or be copied.

**Made only with consent**: `sirius-cli worker setup` (it asks, or takes
`--yes`; see [app/cli/README.md](../cli/README.md#the-python-worker)), or
`sirius-app`'s offer when the worker cannot start and Preferences ▸ Compute ▸
Python environment, which also update, repair and remove it. A worker start
never installs anything.

**How.** The base is a Python 3.9 or newer that is not a free-threaded
build: the one named, else the one the environment was made from, else the
one found on PATH, else the first usable of the usual install locations
(python.org, uv's managed Pythons, `~/.local/bin`). With uv (`$SIRIUS_UV`, or
on PATH) the environment is `uv venv --seed` and `uv pip install`; without,
`python -m venv` and pip. Installs take binary wheels only (`--only-binary
:all:`), so a missing wheel fails in seconds instead of starting a compile,
and pip stays in the environment so the model hub can install into it later.
Proxy, index and certificate variables (`HTTPS_PROXY`, `PIP_INDEX_URL`,
`UV_INDEX_URL`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`) are passed through;
credentials in index URLs are shown as `***` everywhere they appear.

**What is in the directory.**

- `sirius-env.json`, written last: who made it and when, the base Python and
  its version, the installer, the index, what was installed with its version,
  and a fingerprint of the requirements (over the normalised list — comments,
  blank lines, case, order and line endings do not matter — plus the extras).
  A folder without it is a setup that did not finish.
- `README.txt`: that SIRIUS made it for its worker and that it is safe to
  delete.
- Beside it, `python-env.lock` while a setup or a removal runs, so that
  `sirius-app` and `sirius-cli` never change it at the same time (a lock whose
  process is gone, or older than an hour, is taken over).

**States**: *absent*; *incomplete* (no marker, or no interpreter); *outdated*
(the requirement files changed since, so the fingerprint differs); *broken*
(found by a check: it no longer runs, e.g. the Python it was made from was
removed, or `--check` reports a missing package); *ready*. An update installs
into the existing environment; a recreate moves the old one aside and puts
it back if anything fails, so a working environment is never deleted before
its replacement works.

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
currently `2`. Both ends must send the same number — a peer that sends none
counts as version 0 — and a mismatch is refused with a message naming both
versions and which end to update. `hello` and `auth` together are the
handshake: a challenge-response in which the worker proves first that it
holds the token, then the client (`SECURITY.md`); the token itself is never
sent, and requests do not carry it.

| method | params | reply |
| --- | --- | --- |
| `hello` | `{protocol_version, client_nonce}` | `result`: `{protocol_version, server_nonce, server_proof}`; `error` on another protocol version, and the connection is closed |
| `auth` | `{client_proof}` | `result`: `{version, protocol_version, methods, cuda, device, hostname, python, torch, sirius, workbench, encodings, max_clients, tiff_reader}` (`tiff_reader`: `{sirius: version or null, nvtiff}`, who reads cluster TIFF datasets: the sirius package, decoding on the GPU when `nvtiff`; null means TIFF cannot be opened); `error` on a wrong proof (and the connection is closed), or `busy` when every client slot is taken |
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
  `equalizez`; `dx`, `dy`, `dz` override the voxel size of `meta`. An empty
  `otf` is the theoretical OTF, as it is in the application: built in 3D when
  the stack holds several planes and in 2D when it holds one, which is what
  `sirius::selectOTF` decides for all three fronts.
* `load`: `path`, `read_as`, `tile`, `page_order`, `c`, `t`, `z`,
  `voxel_x`, `voxel_y`, `voxel_z`, `sim_ndirs`, `sim_nphases`, `sim_fast`,
  `sim_layout`, `sheet_angle`. `run_pipeline` reads the dataset itself, the way the
  application's Load step does: `page_order` and the counts shape the TIFF
  pages (a count left at 0 keeps the OME / ImageJ metadata's), length units
  and resolution tags give the voxel size, and the voxel and SIM parameters
  then override the metadata. `sim_layout` is read by the same parser as the
  application's (`parseSimStorage`), so it accepts and canonicalises exactly
  the same texts and takes the angle and phase counts from the layout itself,
  whatever axes it puts them on; a text that does not read is an error naming
  what is wrong, and a `sim_ndirs` / `sim_nphases` beside it that disagrees is
  refused as the Load step refuses it. A layout that packs everything on z
  (`z=[angle 3, z, phase 5]`, or the fast-SI `z=[z, angle 3, phase 5]`) is
  reconstructed here; one that puts the angles on the channels or tiles a
  montage is gathered only by the application's SIM step, and `sim` says so.

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
python -m unittest app/python/tests/test_environment.py -v   # --check, the exit-3 start check, the requirement files
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
