# Contributing to SIRIUS

How the repository is organised for people writing code in it: which branch a
change goes to, how to build and test it, and what CI will insist on. The
architecture is in [README.md](README.md) and
[docs/design/README.md](docs/design/README.md).

## Branches

- **`main`** — releases. Nothing is pushed to it directly; it moves when `dev`
  is tagged and merged.
- **`dev`** — the integration branch. This is what CI gates and what feature
  branches are cut from and merged back into.
- **`feature/…`**, **`fix/…`** — one branch per piece of work, branched from
  `dev` and opened as a pull request **into `dev`**.

Rebase or merge `dev` into your branch before opening the PR, so the diff CI
runs is the diff a reviewer reads. A PR into `main` is only ever a release.

## Building

The build is driven entirely by `CMakePresets.json`; there is no other set of
flags to remember. Configure, build and test all name the same preset:

```sh
cmake --preset linux-gcc-dev        # library + tests + python bindings, Debug
cmake --build --preset linux-gcc-dev
ctest --preset linux-gcc-dev
```

| Preset | What it adds |
| --- | --- |
| `linux-gcc-dev`, `linux-clang-dev`, `win-msvc-dev` | the library, tests, warnings, Python bindings (Debug) |
| `linux-gcc-app-dev`, `win-msvc-app-dev` | the workbench (`app/`, Dear ImGui; its dependencies are fetched), `sirius-cli`, and TensorStore (zarr / N5) |
| `linux-cuda-dev`, `win-msvc-cuda-dev` | CUDA, cuFFT, nvTIFF, `CMAKE_CUDA_ARCHITECTURES=native` |
| `fiona-avx2-*` | the cluster builds: AVX2, optionally CUDA |
| `*-release` | Release, no tests |

On Linux the `*-app-*` presets need GLFW's X11 / Wayland development packages
and D-Bus (`xorg-dev libwayland-dev libxkbcommon-dev libdbus-1-dev` on Debian
and Ubuntu); everything else the workbench uses is fetched. TensorStore's
first configure fetches and builds ~40 dependencies (several minutes, ~1.5 GB,
needs `nasm` and a `python3`); pass `-DSIRIUS_ENABLE_TENSORSTORE=OFF` while you
are not touching the zarr paths. Options live in `cmake/ProjectOptions.cmake`.

`sirius-cli` ([app/cli/README.md](app/cli/README.md)) is built with the
workbench. `-DSIRIUS_ENABLE_CLI=ON` on a preset without the app builds the
workbench core and `sirius-cli` alone, without the GUI and the packages it
needs, which is the quicker tree when you work on the core, the worker or the
CLI:

```sh
cmake --preset linux-gcc-dev -DSIRIUS_ENABLE_CLI=ON -DSIRIUS_ENABLE_PYTHON_BINDINGS=OFF
```

## Testing

`ctest --preset <name>` runs the Catch2 suites in `tests/` (the library, and
the app's GUI-free core when the app is enabled). There is one test binary per
unit -- `test_tiff_io`, `test_registration`, `test_app_labels` -- linking that
unit and its dependencies and nothing else (docs/architecture.md), plus
`sirius_tests`, which holds all of them over the archives for running across
units by tag:

```sh
ctest --preset linux-gcc-dev -L lib.tiff_io            # one unit's cases
cmake --build build/linux-gcc-dev --target test_tiff_io && \
    ./build/linux-gcc-dev/tests/test_tiff_io --list-tests
./build/linux-gcc-dev/tests/sirius_tests "[tiff]"      # by tag, across units
```

`python3 tools/check_units.py` checks that the `#include` lines still agree
with the unit graph the build declares; the lint job runs it.

`sirius-cli` itself is tested end to end under the label `cli`
(`tests/cli`): one-shot commands, a session transcript, both MCP handshakes
and a Python MCP client. The run of the bundled SIM pipeline is also labelled
`cli.slow`, since it takes long in a Debug MSVC build; the Windows CI job
leaves it out:

```sh
ctest --preset linux-gcc-app-dev -L cli
ctest --preset win-msvc-app-dev -L cli -LE "cli\.slow"
```

Cases skip rather than fail when what they need is absent — no GPU, no
TensorStore, no `SIRIUS_PYTHON`. Set `SIRIUS_PYTHON` to an interpreter with
`numpy` to run the end-to-end cases that start the Python worker
(`tests/test_app_rpc.cpp`, `test_app_local_worker.cpp`,
`sirius::cli.worker_check`); add `torch` for the segmentation case.

SIRIUS's own Python environment for the worker lives in your data directory
(`%LOCALAPPDATA%/sirius/python-env`, `~/.local/share/sirius/python-env`), and
the tests never touch yours: every case that could reach it points
`SIRIUS_PYTHON_ENV` at a temporary directory, and the `cli` tests at one in the
build tree. Do the same when you try `sirius-cli worker setup` or the GUI's
offer by hand. The variables for tests and screenshots:

| variable | effect |
| --- | --- |
| `SIRIUS_PYTHON_ENV=<dir>` | where the environment is made and looked for |
| `SIRIUS_TEST_PYTHON_SETUP=1` | `test_app_python_env` also runs a real setup into a temporary directory and removes it; needs the network, and uv or a Python with `venv` |
| `SIRIUS_PYTHON_OFFER=always\|never` | `sirius-app` shows its "Set up Python for SIRIUS" offer whenever the worker cannot start, or never; without it the offer appears at most once a session and never in a scripted run |

For example, to capture the offer without touching your own setup (on a
machine whose Python on PATH has no numpy):

```sh
env -u SIRIUS_PYTHON SIRIUS_PYTHON_ENV=/tmp/scratch-env SIRIUS_PYTHON_OFFER=always \
    build/linux-gcc-app-dev/app/sirius-app --settings scratch --screenshot offer.png
```

Python:

```sh
python -m unittest discover -s bindings/tests -v      # the bindings (needs pip install -e .)
python -m unittest discover -s app/python/tests -v    # the worker: protocol, steps, plugins, models, start check
```

## The core stays GUI-free

`app/core` is what `sirius-app` and `sirius-cli` share, and it has to build
and test without a display. Core code therefore includes nothing from
`app/imgui` and nothing of Dear ImGui, GLFW, OpenGL, the native file dialogs
or libcurl. The `cli-headless` CI job is what catches a GUI dependency: it
builds the core and `sirius-cli` on a machine with none of the GUI's
packages and without `app/imgui`, so such an include or call fails to
compile or link there. `tools/check_units.py` keeps the core's own graph
honest — every file under `app/core` must belong to a unit, and every
include of a library or core header must be one of that unit's declared
dependencies — but it skips headers no unit owns, so an `imgui/…`,
`imgui.h` or `<GLFW/…>` include passes it.

What both front ends need lives in the core — the OS helpers
(`core/host`), child processes (`core/process`), the worker launcher
(`core/local_worker`), the viewer's display model (`core/display_model`) —
and `app/imgui` re-exports it under the names it had. Messages written in the
core do not send the user to a place in one front end ("Preferences ▸ …"):
each host adds its own next step, `sirius-app` through
`Workbench::setWorkerHint` and `LocalWorker::setSetupHint`, `sirius-cli`
through its own setup hint.

## Formatting and lint

C++ and CUDA use `.clang-format`; Python uses `[tool.ruff]` in
`pyproject.toml`. Install the hooks once and both run on what you commit:

```sh
pip install pre-commit && pre-commit install
```

`.clang-format` was written to reproduce the style the tree already had, so
formatting a file you are editing does not rewrite it. It is still not a
no-op on every file: **CI only checks the files a change touches** (see the
`lint` job in `.github/workflows/dev-tests.yml`), so reformat what you edit
and leave the rest alone.

`ruff check` runs over the whole tree. `ruff format` does not: it has no
options that reproduce the continuation style the Python here uses, and would
rewrite about a third of `models.py`, `server.py` and `workbench.py`. Keep
lines under 130 columns (E501) and imports sorted (I001) and ruff is
satisfied; match the surrounding file for everything else.

Include order is not enforced by re-sorting (`SortIncludes: Never`). The
convention is: the file's own header, then the standard library, then
third-party (Dear ImGui, nlohmann/json …), then `sirius/…`, then `core/…` and
`imgui/…`, blank line
between the groups.

`python tools/check_versions.py` asserts the version in `CMakeLists.txt`
(canonical), `pyproject.toml` and `sirius_worker.__version__` still agree;
change all three together.

## Commits

One imperative subject line prefixed with the component it touches, no
trailing period, body only when the "why" is not obvious:

```
Viewer: solo mode shows one label alone, in the slices and in 3D
App: folder datasets, model hub and plugin manager integration
Tests: plugins land in the User group with the declared group as label
Model hub: install packages on request, fetch weights, gated repos
```

Prefixes in use: `App`, `Core`, `CLI`, `Worker`, `Shell`, `Viewer`, `Tests`,
`Model hub`, `Bindings`, `Build`, `CI`, `Docs` — or the subsystem's own name.
Keep a commit to one change; a branch may have several.

## What CI gates

`.github/workflows/dev-tests.yml` runs on every push to `dev` and every PR:
`lint`, `cpp-tests` (GCC, and the only job where warnings are errors),
`app-tests` (the workbench, its install check, a headless run of the bundled
SIM pipeline under Xvfb, the same pipeline through `sirius-cli`, and SIRIUS's
Python environment set up from scratch with uv), `cli-headless` (the core and
`sirius-cli` built and tested without the GUI), `python-tests`, `sanitizers`
(ASan + UBSan), `windows` (MSVC: the library, the workbench and `sirius-cli`,
built and unit-tested without `cli.slow`, and the Python environment set up
from scratch with pip) and `cuda-build` (compiles the CUDA paths; the GPU cases
skip). A run is cancelled when you push again to the same branch.

## Security-sensitive areas

The Python worker executes what a client asks it to; the trust model and what
`--allow-install` and `--token` mean are in
[app/python/SECURITY.md](app/python/SECURITY.md). Secrets the application
stores go through `app/imgui/secret_store.hpp`, never straight into the settings
file.

`sirius-cli mcp` hands the workbench to an agent. Its tools' hints are a
promise clients act on (some approve read-only tools without asking), so a
tool that writes outside the scratch directory or installs packages is marked
destructive, and a read-only one writes nothing but scratch. (Computing a
model step may still fetch the weights its parameters name; the guide says
so.) Downloading into SIRIUS's Python environment takes the user's consent
every time: `worker setup` asks or needs `--yes`, and `setup_worker_env`
needs the server flag `--allow-worker-setup`, `confirm: true`, and packages
from the worker's own list. The HPC endpoint comes only from the command
line. Keep it so, and keep [docs/agent-guide.md](docs/agent-guide.md) from
suggesting that anyone pre-approve the tools that write or download.
