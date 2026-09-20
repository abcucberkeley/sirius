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
| `linux-gcc-app-dev`, `win-msvc-app-dev` | the Qt workbench (`app/`) and TensorStore (zarr / N5) |
| `linux-cuda-dev`, `win-msvc-cuda-dev` | CUDA, cuFFT, nvTIFF, `CMAKE_CUDA_ARCHITECTURES=native` |
| `fiona-avx2-*` | the cluster builds: AVX2, optionally CUDA |
| `*-release` | Release, no tests |

Point `SIRIUS_QT_DIR` at a Qt prefix for the `*-app-*` presets. TensorStore's
first configure fetches and builds ~40 dependencies (several minutes, ~1.5 GB,
needs `nasm` and a `python3`); pass `-DSIRIUS_ENABLE_TENSORSTORE=OFF` while you
are not touching the zarr paths. Options live in `cmake/ProjectOptions.cmake`.

## Testing

`ctest --preset <name>` runs the Catch2 suites in `tests/` (the library, and
the app's Qt-free core when the app is enabled). There is one test binary per
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

Cases skip rather than fail when what they need is absent — no GPU, no
TensorStore, no `SIRIUS_PYTHON`. Set `SIRIUS_PYTHON` to an interpreter with
`numpy` to run the end-to-end cases that start the Python worker
(`tests/test_app_rpc.cpp`); add `torch` for the segmentation case.

Python:

```sh
python -m unittest discover -s bindings/tests -v      # the bindings (needs pip install -e .)
python -m unittest discover -s app/python/tests -v    # the worker: protocol, steps, plugins, models
```

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
third-party and Qt, then `sirius/…`, then `core/…` and `qt/…`, blank line
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

Prefixes in use: `App`, `Shell`, `Viewer`, `Tests`, `Model hub`, `Bindings`,
`CI` — or the subsystem's own name. Keep a commit to one change; a branch may
have several.

## What CI gates

`.github/workflows/dev-tests.yml` runs on every push to `dev` and every PR:
`lint`, `cpp-tests` (GCC, and the only job where warnings are errors),
`app-tests` (Qt 6 plus a headless run of the bundled SIM pipeline),
`python-tests`, `sanitizers` (ASan + UBSan), `windows` (MSVC + Qt 6) and
`cuda-build` (compiles the CUDA paths; the GPU cases skip). A run is
cancelled when you push again to the same branch.

## Security-sensitive areas

The Python worker executes what a client asks it to; the trust model and what
`--allow-install` and `--token` mean are in
[app/python/SECURITY.md](app/python/SECURITY.md). Secrets the application
stores go through `app/qt/secret_store.hpp`, never straight into `QSettings`.
