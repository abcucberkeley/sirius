# sirius-cli — the workbench without a window

`sirius-cli` is the SIRIUS workbench core (`app/core`) as a command-line
program, for scripts and for agents. It opens the same datasets, runs the same
operations and pipelines and renders what the viewer draws, but it has no
window: no Dear ImGui, GLFW, OpenGL, native file dialogs or libcurl. It works
in three ways:

- **one-shot commands** (`info`, `run`, `render`, `stats`, `export` …), each
  printing one JSON document;
- **a session** (`sirius-cli session`): one live workbench, driven by JSON
  lines on stdin and answering on stdout;
- **an MCP server** (`sirius-cli mcp`): the same tools for an agent such as
  Claude Code, over the Model Context Protocol on stdio.

The `worker` commands look after SIRIUS's own Python environment for the
worker.

All three modes call one tool table, `HeadlessWorkbench`
(`app/core/headless.hpp`): a one-shot command is a short sequence of the tool
calls a session or an MCP client would make, so the modes cannot drift apart.
[docs/agent-guide.md](../../docs/agent-guide.md) is the guide for using it
from an agent; this page is the reference.

## Building

`sirius-cli` is built whenever the workbench is (`SIRIUS_ENABLE_APP=ON`, the
`*-app-*` presets), and on its own with `-DSIRIUS_ENABLE_CLI=ON`. The second
needs none of the GUI's system packages, so a cluster node without X11 or
Wayland will do:

```sh
cmake --preset linux-gcc-dev -DSIRIUS_ENABLE_CLI=ON -DSIRIUS_ENABLE_PYTHON_BINDINGS=OFF
cmake --build --preset linux-gcc-dev --target sirius-cli
build/linux-gcc-dev/app/sirius-cli version
```

```powershell
cmake --preset win-msvc-app-dev
cmake --build --preset win-msvc-app-dev --target sirius-cli
build/win-msvc-app-dev/app/Debug/sirius-cli.exe version
```

The executable lands in `build/<tree>/app/sirius-cli` with a single-config
generator (Ninja, the Linux presets) and in
`build/<tree>/app/<Config>/sirius-cli.exe` with Visual Studio, beside
`sirius-app` when that is built too. The build copies the help pages, the
Python worker (`python/`) and the example plugins beside both executables
(the `sirius_app_runtime_files` target), so either runs from the build tree
as it is.

On Linux, `cmake --install <build> --component app --prefix <prefix>` installs
`bin/sirius-cli` with the data it shares with `sirius-app` under
`share/sirius/` (help, python, plugins). `ctest -R app.install` installs into
a scratch prefix and checks that the installed `sirius-cli` reads its own help
pages and, with `SIRIUS_PYTHON` set, starts its own copy of the worker.

## A first look

```sh
sirius-cli info tests/data/raw.tif
sirius-cli ops contrast --detail
sirius-cli run --pipeline examples/sim_bundled.sirius.toml --render mip.png --plane mip --stats
sirius-cli render --dataset tests/data/raw.tif --plane xz --out xz.png
sirius-cli --dataset tests/data/raw.tif call statistics --t all
sirius-cli worker status
```

`sirius-cli --help` lists the commands:

```
Data and operations
  info <dataset>          dimensions, pixel type, voxel size, channels (reads no pixels)
  ops [kind...]           operations that can be steps, with their parameters
  help [page]             help pages (Markdown); --markdown prints the page itself
Pipelines
  validate                check the pipeline against the dataset without running it
  run                     run the pipeline; then --render / --export / --stats
  render --out <png>      a slice (xy, xz, yz) or a maximum projection of a step, as PNG
  stats                   per-channel intensity statistics and label statistics
  diagnostics             a step's diagnostics (tables, curves, images)
  export --out <path>     write a step's output (OME-TIFF, TIFF, zarr, N5, raw)
  export-training --dir   write a step's labels as a training sample
  export-python --out     write the pipeline as a Python script
Tools and servers
  call <tool>             call one tool of the tool API (the tools MCP serves)
  tools | schema          the tool list (MCP format) | commands, options and exit codes
  session                 a live workbench: JSON lines on stdin / stdout
  mcp                     a Model Context Protocol server on stdio
Python worker
  worker status | check | setup | remove    SIRIUS's own Python environment for the worker
Other
  devices | version
```

followed by the global, state and open options (below), the exit codes and
a few examples. `sirius-cli <command> --help` describes one command: its
usage, its options with their defaults, a sample of its output, its exit
codes and examples. `sirius-cli schema` gives all of it as JSON, which is
what a program should read rather than parse the help text.

## Options

### Where options go

- Global and state options may come before or after the command word, for
  every command except `call`.
- After `call <tool>`, every `--x` is a parameter of that tool. Global and
  state options must therefore come **before** `call`:
  `sirius-cli --dataset raw.tif call render --plane mip`.
- Inside `run`, `--to` and `--force` come before the first action flag. The
  options after `--render P`, `--export P` or `--stats` belong to that action,
  up to the next action flag (`--render`, `--export`, `--stats`,
  `--save-pipeline`, `--export-python`):

  ```sh
  sirius-cli run --pipeline p.sirius.toml --to 3 \
      --render xy.png --plane xy --z 40 \
      --render mip.png --plane mip --channels 0 \
      --export out.ome.tif --dtype uint16 --scaling minmax \
      --stats --t all
  ```

The option names never collide: the dataset's open options are spelled
`--page-order`, `--page-c/t/z`, `--dataset-tile` … so that `render --z/--t`,
`export --tile` and a tool's own parameters after `call` keep their meaning.
After the command word, the command's own options are looked up first. A
value may also be written `--name=value`, and `--` ends the options. A state
or open option given to a command that has no use for it is ignored, with a
warning, so a wrapper script may pass the same options to every command.

### Global options

| option | meaning |
| --- | --- |
| `--python <exe>` | the interpreter for the worker (section "The Python worker") |
| `--worker-dir <dir>` | the directory holding `sirius_worker/` (default: the first that holds it of an installed tree's `share/sirius/python`, the `python/` beside the executable, `$SIRIUS_WORKER_DIR`, `python/` in the working directory, the checkout's `app/python`) |
| `--backend auto\|cpu\|cuda\|hpc` | default `auto`: CUDA when the build has it and a device is present, else CPU; `hpc` needs `--hpc` |
| `--cuda-device <n\|all>` | default 0; `all` spreads the volumes over every GPU |
| `--hpc-device gpu\|cpu` | default `gpu`: where the HPC worker computes, its job's GPU or its CPU; sent with each step, so `set_backend` switches it without a new job. A GPU asked of a job without one fails with "this worker job has no GPU; choose CPU or reconnect with GPUs >= 1" |
| `--hpc <host:port>` | the HPC worker to use, and the only one the process ever uses; its token comes from `$SIRIUS_HPC_TOKEN` |
| `--plugins auto\|on\|off` | user operations: `auto` (default) loads them only when something needs them (below), `on` at start, `off` never |
| `--scratch <dir>` / `--keep-scratch` | where the executor's disk cache and `renders/` go; by default a new temporary directory, removed at exit unless `--keep-scratch` |
| `--record <file.jsonl>` | record the session as `sirius-app --record` does |
| `--timeout <s>` | one-shot: cancel and exit 124 once reached; session: how long end of input waits for a run still going |
| `--progress none\|text\|json` | progress on stderr: none, text, or one JSON object per line (with `json` the log lines are JSON objects too); default `text` when stderr is a terminal, else `none` |
| `--pretty` / `--compact` | stdout JSON indented or on one line; default pretty on a terminal, else compact |
| `--quiet` | no log lines on stderr (a failure still prints its one summary line) |
| `-h`, `--help`, `--version` | |

With `--plugins auto`, user operations load when a pipeline names a kind that
is not built in, on `add_step` of an unknown kind, on `list_operations` with
`include_plugins` and on `list_plugins`. Dataset information, the built-in
operations, the help pages and pipelines of built-in steps therefore work
without any Python at all.

### State options

These set up the workspace before the command runs, for the commands that act
on one: `validate`, `run`, `render`, `stats`, `diagnostics`, `export`,
`export-training`, `export-python`, `call`, `session` and `mcp`. They are
applied in this order, whatever order they are written in: `--pipeline`, then
`--dataset`, then `--steps`, then `--set`.

| option | meaning |
| --- | --- |
| `--dataset <path>` | the dataset (TIFF, OME-TIFF, DeltaVision / MRC `.dv` / `.mrc`, zarr, N5, a folder with a `sirius-dataset.toml`), with the open options below |
| `--pipeline <file.sirius.toml>` | the pipeline, and its dataset unless `--dataset` names one |
| `--steps <json\|@file\|->` | a JSON array of `{kind, preset?, params?, enabled?, name?}`, appended to the pipeline (a preset is applied before the params) |
| `--set <step>.<key>=<value>` | a parameter, repeatable. `<step>` is a number (1 = Load) or a name; the value is read as JSON when it parses, else as text, then converted to the parameter's type |

The open options, also taken by `info <dataset>`:

| option | meaning |
| --- | --- |
| `--page-order czt` | the order of the TIFF pages, fastest first, for a file without dimension metadata (default `czt`) |
| `--page-c N`, `--page-t N`, `--page-z N` | channels and time points (at least 1, default 1) and z planes (default 0: the page count divided by c × t) |
| `--voxel x,y,z` | voxel size in µm |
| `--sim d,p[,fast]` / `--sim <layout>` / `--no-sim` | a SIM raw stack of d directions and p phases, or one with the general storage layout (`--sim 'c=angle 3; z=phase 3'`, the Load step's `sim_layout`), or not one |
| `--dataset-tile N` | the tile of a tiled dataset |
| `--full-load` | read the whole dataset now instead of plane by plane on demand |

Giving any of `--page-order` or `--page-c/t/z` lays the pages out as they
say: a count left out is then 1 for c and t, not the metadata's, and derived
from the page count for z. `0` is not a valid `--page-c` or `--page-t`.

Every option that takes JSON also accepts `| `--steps <json|@file|->` | a JSON array of `{kind, preset?, params?, enabled?, name?}`, appended to the pipeline (a preset is applied before the params) |` (the file's contents) or
`-` (stdin). That spares the quoting trouble JSON meets in PowerShell and Git
Bash; in PowerShell quote the `@` (`'@steps.json'`), which otherwise starts a
splat. `session` and `mcp` are the exception to `-`: stdin is their protocol,
so `--steps -` there is a usage error and `--steps | `--steps <json|@file|->` | a JSON array of `{kind, preset?, params?, enabled?, name?}`, appended to the pipeline (a preset is applied before the params) |` is the way.

## Commands

Each command prints exactly one JSON envelope on stdout (below); its logs and
progress go to stderr. The only exceptions are `--help` and `help --markdown`,
which print text. One-shot commands recompute from scratch each time, because
the executor's cache lives in the process: iterative work belongs in
`session` or `mcp`.

| command | does | `result` |
| --- | --- | --- |
| `version` | nothing | `{name, version, build:{build, version, commit, dirty, ops_schema, api}, schema, protocols:{session, mcp:[…]}, features:{cuda, cuda_devices, zarr, export_formats, readable_extensions}, paths:{executable_dir, help, worker, python_env}}` |
| `devices` | the `list_devices` tool | `{backend, cuda_available, cuda_device, devices:[{index, name, memory_gb, compute}]}` |
| `info <dataset> [open options] [--open]` | `dataset_info`: probes the file; `--open` opens it (lazily), which also fills `metadata_summary` and `dims_from_metadata` (null without it) | DatasetInfo |
| `ops [kind…] [--group G] [--detail] [--plugins]` | `list_operations`; with one kind and `--detail`, `describe_operation` | `{operations:[…]}`, or the description |
| `help [page] [--markdown]` | `get_help`; `--markdown` prints the page itself (not JSON; exit 3 when there is none) | `{pages:[{page, title}]}`, or the page |
| `validate` | `validate` | the check per step; exit 3 (`validation`) when a step has errors |
| `run [--to S] [--force] [actions…]` | `run` (to the end, or until `--timeout`), then each action in the order written | `{run, renders, exports, statistics?, pipeline?, python?}` |
| `render --out P [render options]` | runs what is needed (unless `--no-run`), `render`, then writes the image to P | the caption, with `path` (and `base64` with `--base64`) |
| `stats [stats options] [--no-run]` | `statistics` | per channel, and the labels |
| `diagnostics [--step S] [--detail] [--images DIR] [--no-run]` | `get_diagnostics`; with `--images`, also every image through `render_diagnostics`, written into DIR | the diagnostics |
| `export --out P [export options] [--no-run]` | runs what is needed, then `export_result` | `{path, format, dtype, shape, files, bytes, seconds}` |
| `export-training --dir D [options] [--no-run]` | `export_training_data` | what it wrote |
| `export-python --out PY` | `export_python` | `{path}` |
| `call <tool> [--args JSON\|| `--steps <json|@file|->` | a JSON array of `{kind, preset?, params?, enabled?, name?}`, appended to the pipeline (a preset is applied before the params) |\|-] [--<param> v]…` | any tool | the tool's value |
| `tools [--names]` | nothing | the array MCP's `tools/list` returns (names only with `--names`) |
| `schema` | nothing | `{commands:[{name, synopsis, options:[{name, type, default, help}]}], exit_codes, error_codes, envelope}` |
| `session [--allow-worker-setup] [--allow-network-paths] [state options]` | the session protocol | (protocol) |
| `mcp [--allow-worker-setup] [--read-only] [--allow-network-paths] [state options]` | the MCP server | (protocol) |
| `serve [--host H] [--port P] [--token-file F] [--max-clients N] [--device D] [--scratch DIR] [--no-python-worker]` | SIRIUS's engine for the HPC backend | (the worker protocol on TCP; one announce line on stdout) |
| `worker status` | where the worker's Python comes from, and the state of SIRIUS's environment | see "The Python worker" |
| `worker check` | starts the worker (never installs) and says hello | `{interpreter:{path, source}, capabilities, seconds}` |
| `worker setup [options]` | plans, asks, and sets up SIRIUS's environment | `{env_dir, python, base_python, python_version, installer, mode, packages, extras, seconds}` |
| `worker remove [--yes]` | removes it | `{removed}` |

### `version`, `devices`, `tools`, `schema`

```sh
sirius-cli version
sirius-cli tools --names
sirius-cli schema --pretty
```

`version` reports the protocols this build speaks (`sirius-session/1`; MCP
2026-07-28, 2025-11-25, 2025-06-18, 2025-03-26 and 2024-11-05), whether CUDA
and zarr are available, the export formats and readable extensions, and the
directories it found (help pages, worker, Python environment). `schema` is
what a program should read instead of parsing `--help`.

### `info`

```sh
sirius-cli info tests/data/raw.tif
sirius-cli info stack.tif --page-order zct --page-c 2 --voxel 0.1,0.1,0.3 --open
```

The result is a DatasetInfo, here of a SIM raw stack:

```json
{"path": "C:/data/stack.tif", "name": "stack", "format": "tiff", "shape": "c1 t1 z120 y256 x256",
 "dims": {"c": 1, "t": 1, "z": 120, "y": 256, "x": 256}, "dtype": "uint16",
 "bytes_on_disk": 15728640, "float32_bytes": 31457280, "voxel_um": [0.08, 0.08, 0.125],
 "frame_interval_s": 0, "channels": [{"index": 0, "label": "…", "wavelength_nm": 0, "color": "#…"}],
 "acquisition": "3D-SIM raw · 15 phases", "sim": {"present": true, "ndirs": 3, "nphases": 5, "fast": false},
 "rgb": false, "light_sheet": false, "sheet_angle_deg": 0, "tiles": [{"index": 0, "name": "…", "position_um": […]}],
 "tile": 0, "metadata_summary": "…", "dims_from_metadata": false, "full_load_skipped": ""}
```

`sim.present` is true only when the file's metadata or the open options say
it is a SIM acquisition; `ndirs`, `nphases` and `fast` are then its layout,
and a stack opened with the general storage layout carries it as `layout`
(`"sim": {"present": true, "ndirs": 3, "nphases": 3, "fast": false, "layout":
"c=angle 3; z=phase 3"}`). A stack that is not SIM says `{"present": false}`
and nothing else. A raw stack without that metadata (`tests/data/raw.tif`,
3 directions × 5 phases × 9 planes) reads as a plain z stack. `voxel_um` is
what the file says, or 0.1, 0.1, 0.2 when it says nothing.

### `ops` and `help`

```sh
sirius-cli ops                          # every operation: kind, name, group, presets, needs_worker …
sirius-cli ops --group Segment
sirius-cli ops classic --detail         # one operation: parameters with types, ranges, defaults; presets
sirius-cli help                         # the help pages
sirius-cli help classic --markdown      # one page, as Markdown
```

### `validate`

```sh
sirius-cli validate --pipeline examples/sim_bundled.sirius.toml
```

The result has `ok`, `has_dataset`, `needs_worker`, the worker's interpreter,
and per step its errors, warnings, input and output shapes, estimated bytes
and whether it needs the worker. A step with errors makes the command fail
with `validation` (exit 3).

### `run`

```
run [--to S] [--force]
    [--render P [render options]]… [--export P [export options]]… [--stats [stats options]]
    [--save-pipeline T] [--export-python PY]
```

```sh
sirius-cli --backend cpu run --dataset tests/data/raw.tif --steps '[{"kind":"contrast"}]' \
    --render c.png --plane mip --export out.ome.tif --stats
```

`--to` names the last step to run (a number or a name; default the last
enabled one);
`--force` clears every cache first. The run waits until it finishes, or
until `--timeout`, when it is cancelled and the command exits 124. Then the
actions run in the order written. The result:

```json
{"run": {RunOutcome}, "renders": [{caption, "path": "…/c.png"}], "exports": [{…}],
 "statistics": {…}, "pipeline": {"path": "…"}, "python": {"path": "…"}}
```

A run that fails is the error `run_failed` (or `worker_unavailable`), with the
RunOutcome in `error.data`.

### `render`

```
render --out P [--step S --plane xy|xz|yz|mip --z N[,N…] --t N --y N --x N --channels 0,1
       --layout blend|channels --window auto|full|c=lo:hi[:gamma],… --labels|--no-labels
       --label ID --solo --label-opacity F --region x,y,w,h --max-size N --format png|jpeg
       --no-physical-z --base64 --no-run]
```

```sh
sirius-cli render --dataset stack.tif --plane xz --y 256 --out xz.png
sirius-cli render --pipeline seg.sirius.toml --step 4 --z 10,20,30,40 --labels --out grid.png
sirius-cli render --dataset two-colour.tif --layout channels --window 0=100:2000,1=50:900:0.8 --out ch.png
```

Without `--step` the step is the pipeline's last enabled one, and so for
`stats` and `export`. It runs what the step needs first (`--no-run` uses only
what is computed, which for a one-shot command means the Load step, and then
the default step is the last one with an output), renders as the viewer does
and writes the bytes to P. The arguments are those of the `render` tool (the
section "Rendering" below); the format follows P's extension unless
`--format` names one. `--out` never overwrites an existing file whose
extension is not `.png`, `.jpg` or `.jpeg` (`invalid_argument`). `--base64`
also puts the image into the result.

### `stats`

```
stats [--step S --t N|all --channels 0,1 --percentiles 0.1,1,50,99,99.9 --histogram N --no-labels --no-run]
```

```sh
sirius-cli stats --dataset tests/data/raw.tif --no-run --t all --histogram 64
```

Per channel: min, max, mean, standard deviation, count, NaN count and
percentiles (from at most 4 Mi strided samples; `sampled` says when), the
fraction of saturated values for an integer-typed Load step, and a histogram
when asked for. With labels: their count, sizes in voxels and µm³, flags,
classes and tracks.

### `diagnostics`

```
diagnostics [--step S] [--detail] [--images DIR] [--no-run]
```

```sh
sirius-cli diagnostics --pipeline examples/sim_bundled.sirius.toml --step 2 --detail --images diag/
```

What the step reports in the GUI's diagnostics dock: its summary, facts,
tables, curves and histograms (with `--detail`, the curves' points, at most
200 each, and the histograms' bins) and images (with `--images`, rendered
into DIR as PNG). It runs the pipeline up to the step first unless
`--no-run`.

### `export`, `export-training`, `export-python`

```
export --out P [--step S --format tiff|ome-tiff|zarr|n5|raw --dtype T --scaling cast|minmax|fixed|percentile
       --range lo,hi --percentiles lo,hi --t a:b --z a:b --channels … --compression none|lzw|deflate
       --level N --tiled --tile W,H --bigtiff|--no-bigtiff --pyramid N --chunk c,t,z,y,x --codec C
       --zarr-version 2|3 --labels --labels-only --pipeline-sidecar --no-run]
export-training --dir D [--step S --sample N --slices --min-voxels N --image-dtype T --image-scaling S] [--no-run]
export-python --out PY
```

```sh
sirius-cli export --pipeline p.sirius.toml --out result.ome.tif --dtype uint16 --scaling percentile --percentiles 0.1,99.9
sirius-cli export --pipeline p.sirius.toml --out result.zarr --zarr-version 3 --chunk 1,1,16,256,256 --codec blosc-zstd
sirius-cli export-training --pipeline seg.sirius.toml --dir training/ --slices
sirius-cli export-python --pipeline p.sirius.toml --out p.py
```

The format follows the extension (`.ome.tif`, `.tif`, `.zarr`, `.n5`, `.raw`)
unless `--format` names one; values are written as `float32`, cast, unless
`--dtype` and `--scaling` say otherwise. `--t` and `--z` take `a:b`, from a
up to but not including b (`a:` to the end); `--pyramid 1` means no
pyramid. The export is the GUI's *Export result*: the same containers and
options, `--pipeline-sidecar` writes `<out>.pipeline.toml`, `--labels` adds
the labels and `--labels-only` writes only them (a Deflate uint32 TIFF). zarr
and N5 need a build with TensorStore (`version` says so); otherwise they are
`unsupported`.

`export-training` adds one sample to a training-data folder, as the GUI's
*Export training data* does: `--sample` names the sample folder (default the
dataset's name), `--slices` adds one 8-bit plane and one YOLO file per z,
`--min-voxels` (default 1) leaves out smaller objects, and `--image-dtype`
(`uint8`, `uint16` or `float32`; default `uint16`) and `--image-scaling`
(`cast`, `minmax` or `percentile`; default `percentile`) say how `image.tif`
is written.

### `call`

```
call <tool> [--args JSON|| `--steps <json|@file|->` | a JSON array of `{kind, preset?, params?, enabled?, name?}`, appended to the pipeline (a preset is applied before the params) ||-] [--<param> value]…
```

```sh
sirius-cli --dataset raw.tif call statistics --t all
sirius-cli --dataset raw.tif call render --plane mip --max-size 512
sirius-cli --pipeline p.sirius.toml call get_step --step 3
sirius-cli --dataset raw.tif call probe --args '{"x": 10, "y": 12, "z": 4}'
```

Calls one tool of the table below. The flags come from the tool's input
schema: `max_size` is `--max-size`, arrays are comma lists or JSON, objects
are JSON, and booleans are `--flag` / `--no-flag`. `--args` gives them all as
one object, whose keys the flags override. An unknown tool is `unknown_tool`
(exit 2). The images `render`
makes stay in the scratch directory, which a one-shot command removes (unless
`--keep-scratch`); the `render` command writes one where you say.

## Output, errors and exit codes

**Success**:

```json
{"command": "run", "ok": true, "result": {…}, "schema": "sirius-cli/1", "warnings": […]}
```

**Failure**, followed by one summary line on stderr, `sirius-cli: <code>: <message>`:

```json
{"command": "run", "error": {"code": "worker_unavailable", "data": {…}, "hint": "…", "message": "…"},
 "exit_code": 4, "ok": false, "schema": "sirius-cli/1", "warnings": []}
```

- Keys are sorted. NaN and infinity become `null`, with a warning. Paths are
  absolute, with forward slashes.
- `warnings` collects what did not stop the command: an unknown argument
  name that was ignored, a stale output that was used, a value that could not
  be written as a number.

| exit | meaning | `error.code` |
| --- | --- | --- |
| 0 | ok | |
| 1 | the operation failed | `failed`, `run_failed`, `export_failed`, `io_error`, `internal` |
| 2 | usage | `usage`, `unknown_tool` (in `call`) |
| 3 | invalid input, not found, or cannot run as asked | `not_found`, `open_failed`, `invalid_argument`, `unknown_step`, `unknown_operation`, `no_dataset`, `validation`, `not_computed`, `unsupported`, `too_large`, `busy`, `stale_workspace` |
| 4 | the Python worker is unavailable | `worker_unavailable`, `python_not_found` |
| 5 | consent required | `consent_required` |
| 124 | timed out | `timeout` |
| 130 | interrupted (Ctrl+C) or cancelled | `cancelled` |

`worker_unavailable` carries
`data:{kind, python, source, missing, python_version, fix:"sirius-cli worker setup --yes"}`,
where `kind` is `missing_packages`, `no_interpreter`, `broken_environment`,
`no_worker_scripts` or `failed`. `consent_required` carries what would be
downloaded. The session and the MCP server report the same codes as
`error.code`. The servers themselves exit 0 at end of input, on `shutdown` or
on a termination signal, and 1 when something fatal happens while serving.
Before they serve, they apply the state options; a failure there ends the
process with that error's exit code from the table (2 for a malformed `--set`
or `--steps -`, 3 for a `--dataset` that is not found or a `--steps` kind
that does not exist …), with its summary line on stderr and nothing on
stdout.

**stderr.** The log lines of the workbench and the worker go to stderr as
text, or as `{"event":"log","source","line"}` objects with `--progress json`;
`--quiet` drops them. Progress goes there as `--progress` says: not at all, as
a status line (redrawn in place on a terminal), or as
`{"event":"progress","fraction","message"}` objects. A failure always leaves
its one summary line. stdout carries nothing but the envelope (or, in
`session` and `mcp`, the protocol).

**Interrupts.** The first Ctrl+C cancels what runs (a run, an export, a
worker setup, which then rolls back), and a one-shot command then fails with
`cancelled` (exit 130); a second Ctrl+C ends the process at once, with 130.
SIGTERM, SIGHUP and closing the console are treated as end of input followed
by a cancel: the worker is stopped, the scratch directory removed, and the
process exits. A one-shot command ended this way exits 130 with a `cancelled`
envelope; `session` and `mcp` end with 0.

## The tools

`HeadlessWorkbench::tools()` is the single list `call`, `session` and `mcp`
serve (`sirius-cli tools` prints it). Rules they share:

- Every input schema is `{"type": "object", "properties": {…},
  "additionalProperties": false}`. Unknown argument names are ignored with a
  warning. Parameter keys inside `params` (`add_step`, `set_params`) are
  checked instead: an unknown one fails the call with `invalid_argument`, whose
  hint lists the operation's keys, and a number outside a parameter's range is
  clamped into it, with a warning and the step's `clamped` list saying so.
- **Steps** are addressed by number (1 = Load) or by name or kind, wherever an
  argument is called `step`. The inspecting tools default to the target of
  the last run when it has an output, else the last enabled step with one,
  else Load.
- **Workspace.** Each process has one workbench, its *workspace*, with an id
  `ws_<12 hex>` that `open_dataset`, `load_pipeline`, `get_state` and the
  session's `ready` event report. Every tool accepts an optional `workspace`
  argument; a mismatch is `stale_workspace`, which tells a client that the
  server it talks to has restarted.
- **A result is a failure exactly when it carries an error code**
  (`ToolApi`'s `error_kind`); `error.code`, `message`, `hint` and `data` come
  from it. A successful result may still contain a key named `error`.
- **During a run** only the tools marked *yes* below work; the others answer
  `busy`.
- **Hints** (MCP annotations): RO = read-only, DE = destructive (writes files
  outside the scratch directory, or downloads), ID = idempotent. Open-world
  (reaches beyond this machine): `run`, whose steps may download model
  weights from Hugging Face and whose `hpc` backend sends the data to a
  remote worker, and `setup_worker_env`, which downloads packages.

| tool | arguments (* = required) | returns | hints | during a run |
| --- | --- | --- | --- | --- |
| `open_dataset` | `path`*, `page_order`, `c`, `t`, `z`, `voxel_um` [x,y,z], `sim` ({ndirs, nphases, fast}, {layout: "c=angle 3; z=phase 3"}, a layout text, or false; a layout and the counts together are refused, since the layout already names them), `tile`, `full_load` | DatasetInfo + `workspace` | | |
| `dataset_info` | `path` (probe that file; else the open dataset) + the open options | DatasetInfo | RO | yes (with `path`) |
| `load_pipeline` | `path`*, `dataset` | `{workspace, pipeline_path, steps, dataset, missing_kinds, plugins_loaded}` | | |
| `save_pipeline` | `path`* | `{path}` | DE | yes |
| `clear_pipeline` | | `{steps}` (Load only; one undo entry) | | |
| `get_state` | | `{workspace, dataset, pipeline_path, steps, backend, cuda_device, hpc_device, hpc_configured, running, run, can_undo, undo_label, can_redo, redo_label, plugins, worker, cwd, scratch}` | RO | yes |
| `add_step` | `kind`*, `preset`, `params`, `name`, `at` (the preset is applied first, then the params) | the step (+ `clamped`) | | |
| `remove_step` | `step`* | `{ok}` | | |
| `move_step` | `step`*, `delta`* | the step | | |
| `set_step_enabled` | `step`*, `enabled`* | the step | ID | |
| `set_params` | `step`*, `params`* | the step (+ `ignored`, `clamped`) | ID | |
| `apply_preset` | `step`*, `preset`* | the step | ID | |
| `set_cache` | `step`*, `policy`* (memory, disk, recompute) | the step | ID | |
| `undo`, `redo` | | `{ok, undone \| redone}` | | |
| `load_example_pipeline` | | `{steps}` | | |
| `get_step` | `step`* | the step + `{errors, warnings, output_shape, has_output, output_fresh, diagnostics_summary, note}`; a step cached as `recompute` has no output once the steps after it have read it | RO | yes |
| `list_operations` | `kind`, `group`, `detail`, `include_plugins` | `{operations:[{kind, name, group, params (the keys; with detail, {key, label, default, schema}), presets, produces_labels, needs_labels, needs_worker, plugin, gpu}]}` | RO | yes |
| `describe_operation` | `kind`* | `{kind, name, group, params:[{key, label, type, default, choices, min, max, unit, advanced, schema}], presets, needs_worker, produces_labels, needs_labels, gpu, help}` | RO | yes |
| `get_help` | `kind` \| `page` \| `step` | none given: `{pages}`; else `{page, title, path, exists, markdown, truncated}` | RO | yes |
| `validate` | | `{ok, has_dataset, needs_worker, worker, steps:[…]}` | RO | yes |
| `run` | `step` (the target; default the last enabled step), `wait_s` (50; 0 = return at once; -1 = to the end), `force` | RunOutcome | open-world | |
| `run_status` | `wait_s` (0) | RunOutcome of the active or last run, or `{status:"idle"}` | RO | yes |
| `cancel_run` | | `{cancelled, status}` | | yes |
| `set_backend` | `backend`* (cpu, cuda, hpc), `cuda_device` (a number or "all"), `hpc_device` (gpu, cpu: where the HPC worker computes; kept until changed) | `{backend, cuda_device, hpc_device}` | ID | yes |
| `list_devices` | | `{backend, cuda_available, cuda_device, devices}` | RO | yes |
| `render` | see "Rendering" | the caption + the image | RO | |
| `render_diagnostics` | `step`, `tab`, `index` (0), `max_size` (768) | `{step, tab, index, title, width, height}` + the image | RO | |
| `probe` | `step`, `x`*, `y`*, `z`, `t` (0) | `{step, fresh, values:[{channel, label, value}], label}` | RO | |
| `statistics` | `step`, `t` (0 or "all"), `channels`, `percentiles` ([0.1, 1, 50, 99, 99.9]), `histogram_bins` (0), `labels` (true), `max_samples`, `run` (false) | `{step, fresh, shape, sampled, channels:[…], labels?}` | RO | |
| `get_diagnostics` | `step`, `detail` (false) | tables, curves, images, notes | RO | |
| `list_tracks` | `step`, `limit` (50) | the tracks | RO | |
| `list_labels` | `step`, `t` (0), `limit` (200), `flag`, `unreviewed` | `{step, t, count, listed, max_label, tracked, shape, labels:[{id, voxels, class, confidence, flags, reviewed, bbox, centre}]}` | RO | |
| `get_log` | `lines` (30, at most 500) | `{lines}` | RO | yes |
| `export_result` | `path`*, `step`, `format`, `dtype` ("float32"), `scaling`, `range`, `percentiles`, `t`, `z`, `channels`, `tiff`{…}, `zarr`{…}, `include_labels`, `include_pipeline`, `labels_only`, `run` | `{path, format, dtype, shape, files, bytes, seconds}` | DE | |
| `export_training_data` | as the GUI's training export | what it wrote | DE | |
| `export_labels` | `path`*, `step` | `{step, path, pages, shape, bytes, labels, max_label}`: one uint32 TIFF, t*z pages, what an `import_labels` step reads back | DE | |
| `paint_label` | `x`*, `y`*, `z`*, `step`, `t` (0), `label` (0 = new), `radius` (3), `z_radius` (0), `erase` | `{step, t, label, voxels, labels}` | | |
| `fill_label` | `x`*, `y`*, `z`*, `step`, `t`, `label` (0 = new) | `{step, t, label, voxels, labels}` | | |
| `merge_labels` | `ids`* (two or more), `step`, `t` | `{step, t, into, voxels, labels}` | | |
| `split_label` | `label`*, `a`* [x, y, z], `b`* [x, y, z], `step`, `t` | `{step, t, label, created, voxels, labels}` | | |
| `delete_label` | `label`*, `step`, `t` | `{step, t, label, voxels, labels}` | | |
| `clear_labels` | `step`, `t` (default every time point) | `{step, t, voxels, labels: 0}` | | |
| `set_label_reviewed` | `label`*, `step`, `t`, `reviewed` (true) | `{step, t, label, reviewed}` | ID | |
| `export_python` | `path` | `{path}`, or `{script}` without one | DE | yes |
| `list_plugins` | `reload` (false) | `{plugins:[{kind, name, file, error}], dirs, registered}` | RO | |
| `worker_status` | `check` (false; true starts the worker and adds `capabilities`) | as `worker status`, + `running` | RO | yes (without `check`) |
| `setup_worker_env` | `confirm`* (must be true), `extras`, `packages`, `base_python` | as `worker setup` | DE, open-world | |

A step, as the editing tools return it:
`{step, number, kind, name, enabled, pinned, cache, params, summary}`.
`view_step`, `select_step`, `set_view` and `focus_track`, which only move the
GUI's view, are not served: every tool here takes explicit steps and
coordinates instead.

**Label editing.** `paint_label`, `fill_label`, `merge_labels`, `split_label`, `delete_label`, `clear_labels`
and `set_label_reviewed` are the viewer's paint tools by step and time point (default: the last computed step,
t 0): each is one undo entry, as a stroke in the viewer is, and `list_labels` is the review table. `export_labels`
writes a step's labels as one uint32 TIFF, and an `import_labels` step (`add_step` with `params.path`) loads such a
file -- its own, a sidecar of `export_result`, or labels made elsewhere on the same grid -- as the labels of its
input, which the same tools then edit. The GUI reaches the same tools through `--tool` and the assistant.

**Prompt steps.** A `foundation` step (a model folder whose `model.json` tasks include `prompt`)
or a `seg` step with a `microsam:` model, whose `task` is `"Prompt objects"`,
segments only what its `prompts` parameter points at. `add_step` and
`set_params` take the list, `get_step` returns it, `describe_operation` gives
its schema (type `prompts`): records in voxels of the step's input, x y z
order, each on one time point `t` (default 0) and belonging to one object
`object` (an id >= 1) --
`{"kind": "point", "x", "y", "z", "t", "label", "object"}` (label 1 object, 0
background), `{"kind": "box", "x0", "y0", "z0", "x1", "y1", "z1", "t",
"object"}` (upper corner exclusive, one per object) and `{"kind": "scribble",
"points": [[x, y, z], ...], "t", "label", "object"}`; an entry without a kind
is a point. **One mask per object**: all of an object's prompts on a time
point are sent together (the worker's joint `objects` form), so a background
point with an object's id is a correction that refines that object's mask, and
the mask comes back with the object's id as its label, the same id on every
re-run. Without `object`, a box, object point or scribble starts an object of
its own and a background point joins the nearest object on its time point. An
object with only background prompts is not sent; a time point without objects
is left empty and costs no worker call. micro-SAM's masks are per plane, so
all of one object's prompts must share a plane. A Prompt step whose `apply` is `The input's
labels` writes its objects over the labels that reach it instead of answering with its own: each
prompted object becomes a cell of its own, which takes its voxels out of whatever cell held them, so
one click inside a merged cell splits off the one that was swallowed and every other cell keeps its
id. That is how a label map loaded with `import_labels` is corrected by the model rather than
replaced by it.

```
sirius-cli --dataset raw.tif call add_step --args '{"kind": "foundation", "params": {"model": "models/coat-sam-s2/v1",
  "task": "Prompt objects", "prompts": [{"kind": "box", "x0": 10, "y0": 12, "z0": 60, "x1": 30, "y1": 34, "z1": 75, "object": 1},
  {"x": 22, "y": 30, "z": 67, "label": 0, "object": 1}, {"x": 40, "y": 40, "z": 67, "object": 2}]}}'
```

**Inspecting a step without an output.** `render`, `statistics`, `probe` and
the others need the step computed. `render`, `statistics` and `export_result`
take `run: true` to run it first; otherwise, and for the others, the error is
`not_computed`. An output computed before the last edit is used with
`fresh: false` and a warning.

**`get_help`** takes a `page` or `kind` of letters, digits, `_` and `-`, or
a registered operation kind, which may contain a `.` (a plugin's
`user.denoise`). `/`, `\`, `:` and a leading `.` are always refused; anything
refused is `invalid_argument`. Pages longer than 60 000 characters are cut,
with `truncated: true`.

**`load_pipeline`** loads the user operations when the pipeline names a kind
that is not built in (unless `--plugins off`). With `dataset`, that dataset is
opened with the pipeline's Load parameters (the light-sheet angle is not an
open option and stays as the step has it).

**`set_backend`** takes no host: `hpc` works only when the process was
started with `--hpc`, and is `invalid_argument` otherwise, with the hint to
restart with `--hpc host:port`.

**`setup_worker_env`** refuses with `consent_required` unless the server was
started with `--allow-worker-setup` *and* the call says `confirm: true`. Its
`packages` must come from the worker's list of optional packages (scipy,
scikit-image, torch, huggingface_hub, onnxruntime, cellpose, micro_sam,
btrack) and its `base_python` from the interpreters `worker status` lists as
candidates; there is no index or URL argument. Anything else is
`invalid_argument`, with the allowed values in `data`. Over MCP it carries
`_meta["anthropic/requiresUserInteraction"]: true`.

### Runs

`run` starts the run on a thread of its own and waits `wait_s` seconds (50 by
default, to stay inside a client's tool timeout). A run still going then
returns `status: "running"`, and `run_status` waits again or reports; the
session also sends a `run_finished` event when it ends. `cancel_run` stops it,
including while it is still starting the worker. A run that cannot start is
an error: `busy`, `no_dataset`, `validation` (with the step and its errors in
`data`) or `worker_unavailable`.

RunOutcome:

```json
{"status": "succeeded|failed|cancelled|running", "run_id": "r3", "target_step": 3, "seconds": 12.4,
 "backend": "CPU", "worker": "Local worker: cpu",
 "steps": [{"step": 2, "kind": "sim", "name": "SIM reconstruction", "state": "ran|cached|skipped|failed",
            "seconds": 9.1, "note": "…", "error": ""}],
 "output": {"step": 3, "shape": "c1 t1 z16 y1024 x1024", "dims": {…}, "fresh": true,
            "labels": {"count": 412}, "diagnostics_summary": "…"},
 "progress": {"fraction": 0.42, "step": 2, "message": "…"},
 "log": ["… the last 50 lines logged during the run …"]}
```

`failed` is reported as the error `run_failed`, or `worker_unavailable` when
the run stopped because the worker could not start, with the RunOutcome in
`data`; `cancelled` as `cancelled`.

### Rendering

| argument | meaning |
| --- | --- |
| `step` | the step; default as above |
| `plane` | `"xy"` (default), `"xz"`, `"yz"`, `"mip"` (the maximum projection along z) |
| `z` | a plane number, or a list of them for a grid (xy only; at most 16 tiles); default the middle plane |
| `t` | default 0 |
| `y` (xz), `x` (yz) | default the middle |
| `channels` | default all |
| `layout` | `"blend"` (default: the viewer's additive tint) or `"channels"` (one grey panel per channel, side by side) |
| `window` | `"auto"` (default, robust percentiles) or `"full"` (the whole range) |
| `windows` | `[{channel, lo, hi, gamma}]`, per channel |
| `labels` | draw the label overlay; default on when the output has labels. Never drawn over a `mip`, as in the viewer: a warning says so |
| `label_opacity` | default 0.45 |
| `label`, `solo` | one label id to select, and whether to draw it alone |
| `region` | `[x, y, w, h]` in voxels of the plane |
| `max_size` | the longer side in pixels at most: default 1024, at most 1568; 0 = native, still at most 1568. A smaller image is not enlarged |
| `physical_z` | default true: xz and yz are stretched by the voxel aspect |
| `format` | `"png"` or `"jpeg"`; unset, PNG, or JPEG (quality 90) when the PNG is over 1 MiB |
| `inline` | session only: also return the image as base64 |
| `run` | compute the step first (default false) |

An image is never larger than 4 MiB: over that, the render is coarsened
(at most three times) and then fails with `too_large` ("lower max_size or pick
a region"). xz and yz sections need the whole (c, t) volume in memory, up to
3 GiB; the maximum projection streams plane by plane and works at any size.
The image is written to `<scratch>/renders/render-NNNN.png` (or `.jpg`) and
the caption says so:

```json
{"step": 3, "rendered_step": 3, "fresh": true, "plane": "xy", "t": 0, "z": 60, "y": null, "x": null,
 "width": 1024, "height": 1024, "factor": 2, "pixel_um": [0.16, 0.16],
 "channels": [{"index": 0, "label": "…", "color": "#00ff00", "window": [112, 3890, 1.0]}],
 "labels_drawn": true, "path": "C:/…/renders/render-0003.png", "format": "png", "bytes": 482113}
```

## The session (`sirius-cli session`, "sirius-session/1")

One workbench that lives as long as the process, driven by JSON lines. Each
line on stdin is a request, each line on stdout is one compact JSON object.

- **Start**: `{"event":"ready","protocol":"sirius-session/1","tools":39,"version":"0.1.0","workspace":"ws_…"}`.
- **Request**: `{"id": <number or string>, "method": "<tool or control method>", "params": {…}}`.
  A `jsonrpc` key is tolerated and ignored.
- **Response**: `{"id", "ok": true, "result", "changes", "undoable", "warnings"}`,
  or `{"id", "ok": false, "error": {"code", "message", "hint", "data"}}`.
  `changes` lists what the call did to the workspace; `undoable` says whether
  it made an undo entry.
- **Protocol errors**: a line that does not parse is answered with
  `{"id": null, "ok": false, "error": {"code": "parse_error", …}}`, an unknown
  method with `unknown_method`, a request without an id with
  `invalid_request`.
- **Images** (`render`, `render_diagnostics`) come back as
  `result.image = {path, mime_type, width, height, bytes}`; the file is in
  `<scratch>/renders/` and stays until the session ends. `"inline": true` adds
  `result.image.base64`; it may go inside `params` or at the top level of the
  request. All the images of one answer together may carry at most 4 MiB
  inline.

| control method | does |
| --- | --- |
| `tools` | the tool list, as MCP's `tools/list` has it |
| `status` | `{running, fraction, step, message, run_id, workspace, active_tool}`, answered at once |
| `cancel {id?}` | cancels the active tool or run, answered at once |
| `subscribe {progress: true, log: false}` | turns the event streams on or off |
| `ping` | answered at once |
| `shutdown` | ends the session |

Events:

- `{"event":"progress","id":…,"fraction":0.42,"step":2,"step_name":"SIM reconstruction","message":"…"}`:
  at most four a second, while the request `id` runs (on by default).
- `{"event":"log","source":"workbench|worker","line":"…"}`: after
  `subscribe {"log": true}`.
- `{"event":"run_finished","run_id":"r3","result":{RunOutcome}}`: for a run
  whose `run` call returned `running`.

Requests are handled one at a time, in order. `status`, `cancel` and `ping`
are answered while another request is still working. At the end of input
the queued requests are handled first; a run still going is then waited for,
with its events, up to `--timeout` if one was given, after which it is
cancelled; then the worker is stopped, the scratch directory removed and the
process exits 0. A file of requests is therefore a batch script:

```sh
sirius-cli session --dataset tests/data/raw.tif < script.jsonl
```

A transcript (`>` stdin, `<` stdout, shortened with `…`):

```
< {"event":"ready","protocol":"sirius-session/1","tools":39,"version":"0.1.0","workspace":"ws_5b0e19c4a2f7"}
> {"id":1,"method":"add_step","params":{"kind":"contrast","params":{"hi_percentile":99.5}}}
< {"changes":["…"],"id":1,"ok":true,"result":{"kind":"contrast","name":"Contrast","step":2,…},"undoable":true,"warnings":[]}
> {"id":2,"method":"run"}
< {"event":"progress","fraction":0.5,"id":2,"message":"…","step":2,"step_name":"Contrast"}
< {"changes":[],"id":2,"ok":true,"result":{"run_id":"r1","status":"succeeded","steps":[…],…},"undoable":false,"warnings":[]}
> {"id":3,"method":"render","params":{"step":2,"plane":"mip","max_size":512}}
< {"changes":[],"id":3,"ok":true,"result":{"image":{"bytes":…,"height":…,"mime_type":"image/png","path":"…/renders/render-0001.png","width":…},"plane":"mip",…},"undoable":false,"warnings":[]}
> {"id":4,"method":"get_step","params":{"step":99}}
< {"error":{"code":"unknown_step","data":null,"hint":"…","message":"…"},"id":4,"ok":false}
> {"id":5,"method":"undo"}
< {"changes":["…"],"id":5,"ok":true,"result":{"ok":true,"undone":"…"},"undoable":true,"warnings":[]}
```

## The MCP server (`sirius-cli mcp`)

```
sirius-cli mcp [--allow-worker-setup] [--read-only] [--allow-network-paths] [global and state options]
```

[docs/agent-guide.md](../../docs/agent-guide.md) shows how to register it
with Claude Code and how an agent should use it.

**Transport.** stdio, JSON-RPC 2.0, one compact message per line. stdout
carries only protocol messages; the log goes to stderr as text and is never
sent as a notification. Images are kept within 4 MiB, so every line stays well
under the 10 MB some clients allow.

**Protocol versions.** Both handshakes are served, request by request:

1. A request whose `params._meta` has `io.modelcontextprotocol/protocolVersion`
   is **2026-07-28** (the only version accepted that way; another one gets
   -32022 with `data: {supported, requested}`). It must also carry
   `io.modelcontextprotocol/clientCapabilities` (-32602 otherwise). Its
   results carry `"resultType": "complete"` and
   `_meta: {"io.modelcontextprotocol/serverInfo": {name, version}}`;
   `tools/list` and `server/discover` add `"ttlMs": 3600000` and
   `"cacheScope": "public"`. `server/discover` answers
   `{supportedVersions, capabilities: {tools: {}}, instructions, …}`.
2. Otherwise `initialize` negotiates **2025-11-25, 2025-06-18, 2025-03-26 or
   2024-11-05**: the requested version is echoed, anything else is answered
   with 2025-11-25. The result is
   `{"protocolVersion", "capabilities": {"tools": {}}, "serverInfo": {"name": "sirius", "title": "SIRIUS microscopy workbench", "version"}, "instructions"}`
   (`title` from 2025-06-18 on). `notifications/initialized` gets no reply.
3. A request that is neither, before any `initialize`, gets -32602 ("send
   initialize first or use 2026-07-28 request metadata").

**Methods.** `ping` (answers `{}`), `tools/list` (every tool, one page) and
`tools/call`. Notifications are never answered. `subscriptions/listen`,
`prompts/*`, `resources/*`, `logging/setLevel`, `tasks/*` and anything else
get -32601; a JSON array (a batch) gets one -32600 "batches are not
supported". The server offers tools only: no resources, prompts or logging.

**Tools** are `{name, title, description, inputSchema, annotations:
{readOnlyHint, destructiveHint, idempotentHint, openWorldHint}, _meta}`, with
`annotations` from 2025-03-26 on and `title` and `_meta` from 2025-06-18 on.
`--read-only` leaves out the tools marked DE above (`save_pipeline`,
`export_result`, `export_training_data`, `export_python`,
`setup_worker_env`) and refuses them if called anyway.

**Network paths.** In `session` and `mcp`, every path a tool takes -- a
dataset, a pipeline file and the paths its steps name, an export target or
folder, a step's Path parameter (a model, a PSF, a flat field) -- must be
local: `\\server\share`, `//server/share` and `\\?\UNC\…` are
refused with `invalid_argument`, since Windows opens them by connecting to
that server with the user's credentials. `--allow-network-paths` lifts this
for a server whose user wants it. The one-shot commands take paths from the
command line, as typed, and are not restricted.

**Results.**

- Success: `{"content": [{"type": "text", "text": "<the value as compact JSON>"}], "structuredContent": {value}, "isError": false}`
  (`structuredContent` from 2025-06-18 on).
- An image adds `{"type": "image", "data": "<base64>", "mimeType": "image/png"}`
  after the text item, whose caption names the copy in the scratch directory.
- A tool that fails (a bad step, a missing file, a failed run, `busy`,
  `worker_unavailable`, `consent_required` …) is a result with
  `"isError": true`, the text `"<code>: <message>\nhint: <hint>"` and
  `structuredContent: {"error": {code, message, hint, data}}`: something the
  model can read and act on.
- Protocol errors: -32700 (parse error, no `id`), -32600 (not a valid
  request: `jsonrpc` not "2.0", no `method`, a null id, a batch), -32601
  (unknown or unsupported method), -32602 (unknown tool, `arguments` not an
  object, missing modern `_meta`), -32603 (an internal error outside a
  tool), -32022 (an unsupported modern version). Request ids are echoed with
  their JSON type.

**Progress** is sent only for a request with `params._meta.progressToken`:
`notifications/progress {progressToken, progress, total: 100, message: "Step 03 · SIM reconstruction · …"}`,
strictly increasing, at most four a second, and never after the response.

**Cancellation.** `notifications/cancelled {requestId}` for the request that
runs sets its cancel flag (a run is cancelled, a worker setup rolls back, an
export stops), and that request gets **no response**; a queued request with
that id is dropped. For `run_status` it only ends the wait. Unknown ids and
malformed cancels are ignored.

**End of input** does not stop the server at once: the requests queued
before it are still answered, in order, for a grace of 10 seconds. Then the
active tool and run are cancelled, what is still queued is dropped, the worker
is stopped, the scratch directory removed, and the process exits 0. A client
that wants an immediate stop sends SIGTERM (on Windows, terminates the
process), which cancels at once.

The `instructions` the server sends, which tell a model how to use it:

```
SIRIUS: a microscopy workbench (3D/4D TIFF, OME-TIFF, zarr; SIM reconstruction, deconvolution, deskew,
contrast, stitching, registration, segmentation, tracking). One live workspace per server.
Workflow: open_dataset(path) or load_pipeline(path) -> list_operations / describe_operation / get_help
-> add_step, set_params, apply_preset (every edit is undoable: undo / redo) -> validate -> run (returns
status "running" after wait_s; then poll run_status) -> look with render (an image you can see; at most
1024 px unless max_size says otherwise), statistics, probe, get_diagnostics -> export_result /
save_pipeline. Step 1 is always Load; address steps by number (1 = Load) or name. Relative paths resolve
against the server's working directory (get_state.cwd); prefer absolute paths. Steps that need Python
(segmentation models, btrack tracking, plugins) use the Python worker: if a tool reports
worker_unavailable, ask the user to run `sirius-cli worker setup --yes` (it downloads numpy from
pypi.org) instead of working around it.
```

A handshake by hand (`>` stdin, `<` stdout, shortened):

```
> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"by-hand","version":"1"}}}
< {"id":1,"jsonrpc":"2.0","result":{"capabilities":{"tools":{}},"instructions":"SIRIUS: …","protocolVersion":"2025-11-25","serverInfo":{"name":"sirius","title":"SIRIUS microscopy workbench","version":"0.1.0"}}}
> {"jsonrpc":"2.0","method":"notifications/initialized"}
> {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"open_dataset","arguments":{"path":"/data/raw.tif"}}}
< {"id":2,"jsonrpc":"2.0","result":{"content":[{"text":"{\"dims\":{…},…}","type":"text"}],"isError":false,"structuredContent":{"dims":{…},…}}}
```

## The engine (`sirius-cli serve`)

`sirius-cli serve` is SIRIUS's C++ engine as the worker of the HPC backend:
the cluster job runs it on the node. It speaks the Python worker's protocol
(`app/core/rpc.hpp`: the frames, the HMAC handshake, progress and cancel), so
the application connects to it as to that worker, and its hello carries an
`engine` block more: the build (`version`'s `build`: the git commit, the
operation schema's hash and the engine API), the GPUs, and the state of its
Python worker.

- It serves the cluster datasets itself -- `dataset_info`, `dataset_read`,
  `dataset_view`, `dataset_stats`, with the replies of the Python worker's
  `datasets.py` -- reading TIFF with SIRIUS's own reader (nvTIFF on a CUDA
  device when the build has it and a request's `device` asks for one) and
  `.npy`.
- It runs the application's pipelines (`pipeline_run`: the pipeline, the
  step to run to and the device, `cuda` or `cpu`) with the same executor and
  operations as the application, CUDA where SIRIUS has it, and keeps every
  output on the node under a handle, `sirius-out:<session>/<step id>/<node
  fingerprint>`, which `dataset_view` / `_read` / `_stats` accept in place of a
  path: the application draws a node's result at screen size and downloads
  nothing else. Progress frames carry the step and its state; the result
  carries each step's report, meta, diagnostics (images as float32 tensors)
  and where it ran. `step_preview` and `step_validate` answer for a step on
  its input and files as the node has them, `output_stats` measures an output
  or a dataset there, `put_file` / `stat_file` receive a file the user agreed
  to upload (into the scratch, `--scratch`; a folder of its own, removed when
  the engine ends), `outputs_release` and `cache_status` manage the outputs.
  The session is the process's: a new engine holds none of an older one's
  handles, and says so.
- Every other request is relayed to a Python worker it starts beside it, on
  127.0.0.1 with a token of its own (`--python`, `--worker-dir` choose it);
  `--no-python-worker` refuses those requests instead, saying so.
- The token comes from `--token-file` or `$SIRIUS_TOKEN_FILE` (read, then
  deleted; on POSIX it must be a file of this user's that nobody else can
  read), else `$SIRIUS_TOKEN`. Listening on an address other than loopback
  without one is refused (exit 2).
- Once it listens it prints one line, `{"port": N, "pid", "host", "hostname",
  "device", "engine": {...}}`, the line the cluster session waits for in the
  job's log, and nothing else on stdout. It ends on a client's `shutdown`,
  SIGTERM, Ctrl+C, or (with `--exit-with-parent`) the end of its stdin.

## The Python worker

Steps that live in Python (segmentation models, btrack tracking, user
operations) run in the Python worker
([app/python/README.md](../python/README.md)), which `sirius-cli` starts when
something first needs it: a step, or the user operations (see `--plugins`).
It runs with `--exit-with-parent`, never with
`--allow-install`: a worker `sirius-cli` started installs nothing.

**Which Python.** The first of:

1. `--python <exe>`;
2. `$SIRIUS_PYTHON`;
3. SIRIUS's own Python environment, when it exists (also when it needs an
   update or a repair: `worker status` then says so);
4. the first Python on PATH (`python3`, then `python`; on Windows also the
   newest `python3.N.exe`, never the Microsoft Store's alias);
5. `python` (Windows) or `python3`.

`sirius-cli` does not read `sirius-app`'s settings, so the interpreter set in
its Preferences is not used here; pass it with `--python` when you want it.

**SIRIUS's own environment** is a virtual environment in the user's data
directory, holding what the worker needs (numpy, from
`app/python/requirements.txt`) and, optionally, scipy and scikit-image
(`requirements-extra.txt`):

| system | directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%/sirius/python-env` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/sirius/python-env` |
| macOS | `~/Library/Application Support/sirius/python-env` |

`$SIRIUS_PYTHON_ENV` names another directory (a larger disk on a cluster; a
scratch directory in tests). `sirius-app` and `sirius-cli` share it, and a
lock keeps them from setting it up at the same time.

```sh
sirius-cli worker status          # interpreter and source, the environment's state, uv, candidates
sirius-cli worker check           # start the worker and say hello; exit 4 when it cannot start
sirius-cli worker setup           # asks, then creates or updates the environment
sirius-cli worker setup --yes --extras
sirius-cli worker setup --dry-run # only the plan: base Python, installer, packages, commands
sirius-cli worker remove
```

```
worker setup [--yes] [--extras] [--package P]… [--base-python P] [--update|--recreate] [--no-uv]
             [--index-url U] [--find-links D]… [--no-index] [--dry-run]
```

- **Consent.** On a terminal, `worker setup` asks on stderr: "Download
  <packages> (about N MB) from <index> into <dir>? [y/N]". Without a terminal
  it needs `--yes`, and otherwise exits 5 with `consent_required` and the plan
  in `error.data`. `worker remove` asks the same way. Nothing else ever
  installs: a worker that cannot start for want of numpy is reported as
  `worker_unavailable`, with the hint to run `sirius-cli worker setup --yes`.
- **How.** With uv when it is found (`$SIRIUS_UV`, else on PATH; `--no-uv`
  turns it off), otherwise with the base Python's `venv` and pip. Only binary
  wheels are installed, nothing is pinned, and `sirius-env.json` in the
  environment records what was installed, from where and from which Python.
  `--package` adds packages (the model packages, say); `--index-url`,
  `--find-links` and `--no-index` choose the source, and so do the usual
  `PIP_INDEX_URL`, `UV_INDEX_URL`, `HTTPS_PROXY`, `SSL_CERT_FILE` and
  `REQUESTS_CA_BUNDLE`. Credentials in index URLs are shown as `***`.
- **Safety.** An environment that works is never deleted before its
  replacement does: `--recreate` moves it aside and restores it if anything
  fails. Ctrl+C rolls back, ends every installer process, and exits 130.
- **States.** `absent`, `incomplete` (a setup that did not finish), `ready`,
  `outdated` (the requirement files changed since), `broken` (it no longer
  runs, e.g. the Python it was made from is gone). `worker setup` does what
  the state calls for: create, update, recreate, or nothing.
- **Failures** have their own messages and hints: no Python found, a Python
  older than 3.9 or a free-threaded build, no `venv` support, offline, TLS
  interception, no wheel for this Python, disk full, the environment in use.

`worker status` returns
`{interpreter: {path, source}, environment: {state, dir, python, problem, marker, …}, uv: {path, version} | null, candidates: [path…], worker_dir, requirements: {required, extras}}`.
`worker check` returns `{interpreter, capabilities: {version, protocol, methods, cuda, device, hostname, python}, seconds}`.

## Environment variables

| variable | read by | meaning |
| --- | --- | --- |
| `SIRIUS_PYTHON` | both executables | the worker's interpreter, unless `--python` names one |
| `SIRIUS_PYTHON_ENV` | both | where SIRIUS's own Python environment lives |
| `SIRIUS_UV` | both | the uv executable to set it up with |
| `SIRIUS_HPC_TOKEN` | `sirius-cli` | the token for `--hpc` (never on the command line, which other users can see) |
| `SIRIUS_TOKEN_FILE`, `SIRIUS_TOKEN` | `sirius-cli serve`, the worker | the server's token: a file read and then deleted, else the value itself |
| `SIRIUS_WORKER_VIEW_CACHE_MB` | `sirius-cli serve`, the worker | how much of the cluster datasets' volumes is kept for re-slicing (default 4096) |
| `HF_TOKEN` | `sirius-cli`, the worker | a Hugging Face token for gated models |
| `SIRIUS_HELP_DIR` | both | a directory of help pages to use first |
| `SIRIUS_WORKER_DIR` | both | a directory holding `sirius_worker/`, used when there is no installed or built copy (see `--worker-dir`) |
| `SIRIUS_PLUGIN_DIRS` | the worker | more directories of user operations (`os.pathsep`-separated) |
| `SIRIUS_PYTHON_OFFER` | `sirius-app` only | `always` or `never`: whether the GUI offers to set up the environment when the worker cannot start (for screenshots and tests) |

## For contributors

**Where the code is.** `app/cli` is the command line only:

| file | what it does |
| --- | --- |
| `main.cpp` | the global options, the dispatch, the exit code |
| `args.*` | the parser, with the placement rules above |
| `commands.*` | the one-shot commands (as tool calls) and the envelope |
| `stdio.*` | the protocol stream, the stdin reader, signals, the console |
| `usage.cpp` | the help texts and `schema` |

Everything else is in the core, so the tests reach it without a process:
`headless*` (the tool table, runs, rendering, export), `agent_protocol`,
`agent_session` and `agent_mcp` (the session and MCP servers over any
`ToolDispatcher`, with no I/O of their own), `python_env`, `local_worker`,
`worker_error`, `host`, `process`, `display_model`, `image_encode` and
`statistics`. None of it includes anything from `app/imgui`.

**Threads.**

- The main thread is the only one that touches the `Workbench`, the
  `ToolApi` and the `DisplayModel`. It runs the one-shot command or
  `Server::step()`, and `HeadlessWorkbench::pump()`, which folds finished
  runs back, forwards log lines and queues events. `call()` pumps first.
- A detached stdin reader feeds `Server::receive()`. It answers `ping`,
  `status`, `cancel` and `notifications/cancelled` itself, through the
  thread-safe `status()`, `cancelActive()` and per-request cancel flags, and
  queues everything else.
- A run executes `RunJob::execute()` on a thread `HeadlessWorkbench` owns;
  that is also where the worker starts.
- The worker's stderr reader only pushes lines into a queue.
- A signal watcher turns POSIX signal flags into cancels; on Windows the
  console control handler does the same.

**stdout is the protocol.** The first thing `main` does is keep a duplicate of
the real stdout for itself and point fd 1 (and, on Windows, the standard
output handle) at stderr, so a stray `printf` or `std::cout` in a library can
never corrupt a JSON line. Results and protocol messages go through
`cli::writeLine()` only, one `\n`-terminated line at a time under a mutex,
flushed each time; JSON is dumped with invalid UTF-8 replaced. Log lines reach
stderr through `HeadlessOptions::logSink`, never directly. On Windows stdin and
the kept stdout are binary, the console code page is UTF-8 for the process's
life, and argv is UTF-8 because the executable embeds
`cmake/utf8-codepage.manifest`.

**Shutdown** at end of input, on `shutdown` or on a termination signal: the
server finishes (the session waits for a run, MCP cancels), the server is
closed, `HeadlessWorkbench` is destroyed (it cancels and joins the run and
stops the worker), the scratch directory is removed, the output is flushed
and the console code page restored, and the process ends with `std::_Exit`,
so no static destructor races the reader thread still blocked on stdin.

**Adding a tool** takes two edits in `app/core/headless.cpp`
(`headless_tools.cpp` holds only the pure halves of some tools: DatasetInfo,
the open options, the help pages, the operation descriptions and the
diagnostics as JSON):

- an `addTool(...)` call in `Impl::installTools()`, with a description of one
  or two sentences that starts with what the tool is for, an input schema
  (`schema(...)`, which adds `additionalProperties: false`) and the function;
- a row in `toolTraits()`, where it goes in the order `tools()` lists them:
  its title, whether it is answered during a run (every other tool says
  `busy`), its honest hints (`readOnly`, `destructive`, `idempotent`,
  `openWorld`), and whether its result may be long or needs a person's
  agreement (the `_meta` keys for MCP clients).

It then appears in `call`, `session` and `mcp` at once;
`tests/test_app_headless.cpp` checks the table's rules, a title for every
tool among them. A tool that writes files does so only when that is its
purpose, and is marked destructive.

**Adding a command or an option** touches `args.cpp`, `commands.cpp` and
`usage.cpp` (its `--help` text and its `schema` entry), and this page.

**Tests.**

```sh
ctest --preset linux-gcc-app-dev -L "app\.(headless|agent_protocol|python_env|local_worker)"
ctest --preset linux-gcc-app-dev -L cli              # the executable, end to end (tests/cli)
ctest --preset linux-gcc-app-dev -L cli -LE cli.slow # without the bundled SIM run
```

The end-to-end tests run `sirius-cli` with `SIRIUS_PYTHON_ENV` pointing into
the build tree, never at your own environment; `sirius::cli.worker_check`
needs `SIRIUS_PYTHON`, and `sirius::cli.agent_client` a Python 3 (standard
library only) to act as an MCP client.
