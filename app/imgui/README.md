# sirius-imgui — the workbench over Dear ImGui

The same application as `app/qt` (docs/design/README.md), on the same GUI-free
core (`app/core`, `sirius::app_core`), drawn with Dear ImGui (docking branch),
GLFW and OpenGL 3.3, with ImPlot for the diagnostics charts. Nothing here needs
Qt; every dependency is fetched and built in-tree (`cmake/Dependencies.cmake`).

```
cmake --preset win-msvc-imgui-dev          # or linux-gcc-imgui-dev / -release
cmake --build --preset win-msvc-imgui-dev --target sirius-imgui
build/win-msvc-imgui-dev/app/imgui/Debug/sirius-imgui tests/data/raw.tif
```

`-DSIRIUS_ENABLE_TENSORSTORE=ON` adds zarr / N5 (needs `nasm` and `python3`, a
long first build). The command line is the Qt application's: `--dataset`,
`--pipeline`, `--run`, `--tool`, `--action`, `--ask`, `--drop`, `--stroke`,
`--wheel`, `--record`, `--screenshot`, `--settings`, `--quit-after`.

## Layout of the code

| Files | What they are | Qt counterpart |
| --- | --- | --- |
| `main.cpp` | command line, scripting, the objects' lifetimes | `qt/main.cpp` |
| `app.hpp/.cpp` | window, title/menu bar, docks, status bar, actions and shortcuts, dialogs and boxes, every command | `qt/main_window.cpp` |
| `bridge.hpp/.cpp` | workbench observer → revisions and signals; runs and tasks on a worker thread | `qt/workbench_bridge.cpp` |
| `theme.hpp/.cpp` | design tokens, fonts, the ImGui / ImPlot style | `qt/theme.cpp` |
| `widgets/controls`, `widgets/icons` | the design's controls and icon set | `qt/widgets/*` |
| `settings`, `secret_store`, `platform`, `process`, `worker_launcher`, `http`, `gl` | services: persistent settings, secrets, OS dialogs and paths, child processes, the Python worker, HTTP(S), textures | `QSettings`, `qt/secret_store`, `QFileDialog`, `QProcess`, `qt/worker_launcher`, `QNetworkAccessManager`, `QImage` |
| `viewer/*` | toolbar, tool strip, ortho / 3D / compare views, dims strip | `qt/viewer/*` |
| `panels/*` | operations, parameters, diagnostics, log, help, assistant | `qt/panels/*` |
| `dialogs/*` | open, folder dataset, export, training export, preferences, model hub, plugin manager | `qt/dialogs/*` |

Everything is in namespace `sirius::app::gui`; includes are written from `app/`
(`"core/workbench.hpp"`, `"imgui/theme.hpp"`).

## How it differs from the Qt layer

**Immediate mode.** A panel is a class with a `draw()` that the application
calls every frame inside the window it began. There are no widgets to keep in
step with the workbench: `draw()` reads `app.wb()` and draws what it finds.
State a panel owns (a scroll position, the text being typed, which tab) lives
in its `Impl`.

**Revisions instead of signals.** What is expensive to derive — a rendered
slice, a texture, a parsed help page, a sorted table — is cached and rebuilt
when the revision of what it came from moves: `app.bridge().rev().outputs`,
`.pipeline`, `.labels`, `.viewState` … (`bridge.hpp`). The `Signal`s on the
bridge are for what must happen once (raise a box when a run failed).

**Nothing blocks.** `QDialog::exec()` becomes `App::showDialog()` plus a
callback; `QMessageBox::question` becomes `App::ask(..., answer)`. Native file
dialogs (`platform::openFileDialog` …) do block, and are fine to call from a
button handler. Work that takes time goes to `Bridge::startTask` (one at a
time, with progress in the status bar) or a thread of the panel's own that
reports back through `Bridge::post`; `http::Fetch` does that for requests.
`App::defer(fn)` runs `fn` between two frames — the place for anything that
opens dialogs from inside a popup, and the only place `App::waitUntil` may be
used.

**Design pixels.** Every size in the panels is in the design's pixels and goes
through `theme::px()`; font sizes are given to `theme::FontScope` /
`widgets::text` in design pixels too. Positions read from or handed to Dear
ImGui are display pixels. The monitor's content scale (and `ui/scale`) is the
factor between the two.

**Drawing.** Custom controls draw into `ImGui::GetWindowDrawList()` with the
tokens of `theme.hpp`; borders go through `widgets::crispRect` so a 1.5 px line
is whole pixels at any scale. Images are `gui::Texture`s (`gl.hpp`) drawn with
`ImDrawList::AddImage(texture.ref(), …)`.

## Settings

`Settings` (`settings.hpp`) is one JSON object in
`<config>/sirius/sirius-imgui.json`, keyed like the Qt application's
`QSettings` (`worker/python`, `recent/datasets`, `assistant/model`, …). Dear
ImGui's window layout is `imgui.ini` beside it. Secrets never go there:
`secrets::read / write` (DPAPI on Windows, `~/.sirius/secrets.json` elsewhere,
shared with the Qt application).
