# sirius-app — the workbench's GUI

The application of docs/design/README.md over the GUI-free core (`app/core`,
`sirius::app_core`), drawn with Dear ImGui (docking branch), GLFW and OpenGL
3.3, with ImPlot for the diagnostics charts. Every dependency is fetched and
built in-tree (`cmake/Dependencies.cmake`); nothing needs installing but, on
Linux, the X11 / Wayland and D-Bus development headers.

```
cmake --preset win-msvc-app-dev            # or linux-gcc-app-dev / -release
cmake --build --preset win-msvc-app-dev --target sirius-app
build/win-msvc-app-dev/app/Debug/sirius-app tests/data/raw.tif
```

The `*-app-*` presets turn on zarr / N5 (TensorStore: needs `nasm` and
`python3`, a long first build); `-DSIRIUS_ENABLE_TENSORSTORE=OFF` skips it. The
command line: `--dataset`, `--pipeline`, `--run`, `--tool`, `--action`, `--ask`,
`--drop`, `--stroke`, `--wheel`, `--record`, `--screenshot`, `--settings`,
`--size`, `--quit-after` (`sirius-app --help`).

## Layout of the code

| Files | What they are |
| --- | --- |
| `main.cpp` | command line, scripting, the objects' lifetimes |
| `app.hpp/.cpp` | window, title/menu bar, docks, status bar, actions and shortcuts, dialogs and boxes, every command |
| `bridge.hpp/.cpp` | workbench observer → revisions and signals; runs and tasks on a worker thread |
| `theme.hpp/.cpp` | design tokens, fonts, the ImGui / ImPlot style |
| `widgets/controls`, `widgets/icons`, `widgets/code_editor` | the design's controls, icon set and the plugin editor |
| `settings`, `secret_store`, `platform`, `process`, `worker_launcher`, `http`, `gl` | services: persistent settings, secrets, OS dialogs and paths, child processes, the Python worker, HTTP(S), textures and PNG |
| `viewer/*` | toolbar, tool strip, ortho / 3D / compare views, dims strip, the volume loader and ray caster |
| `panels/*` | operations, parameters, diagnostics, log, help, assistant |
| `dialogs/*` | open, folder dataset, export, training export, preferences, model hub, plugin manager |

Everything is in namespace `sirius::app::gui`; includes are written from `app/`
(`"core/workbench.hpp"`, `"imgui/theme.hpp"`). The fonts and the icons are in
`app/resources`, copied beside the executable by the build and installed with it.

## How the layer works

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

**Nothing blocks.** A dialog is shown with `App::showDialog()` and reports
through a callback; a question is `App::ask(..., answer)`. Native file dialogs
(`platform::openFileDialog` …) do block, and are fine to call from a button
handler. Work that takes time goes to `Bridge::startTask` (one at a time, with
progress in the status bar) or a thread of the panel's own that reports back
through `Bridge::post`; `http::Fetch` does that for requests. `App::defer(fn)`
runs `fn` between two frames — the place for anything that opens dialogs from
inside a popup, and the only place `App::waitUntil` may be used.

**Keys.** The menu actions own their shortcuts (`App`'s action table). They
stand back while a text field or a popup has the keyboard, while a non-modal
dialog has focus, for a chord a panel claims with `App::claimKey` on the
frames it uses it itself (Ctrl+C in the log, the arrows in a focused pane),
and for a plain key (no Ctrl, Alt or Super) that a Dear ImGui item owns: plain
chords fire only when
`ImGui::IsKeyChordPressed(chord, ImGuiInputFlags_None, ImGuiKeyOwner_NoOwner)`
holds, so a focused widget that uses plain keys calls
`ImGui::SetKeyOwner(key, ImGui::GetItemID())` on each frame it has focus (the
arrows in `widgets::slider` / `sliderInt`, Page Up / Page Down / Home / End in
the dims strip).

**Design pixels.** Every size in the panels is in the design's pixels and goes
through `theme::px()`; font sizes are given to `theme::FontScope` /
`widgets::text` in design pixels too. Positions read from or handed to Dear
ImGui are display pixels. The monitor's content scale (times `ui/scale`, or
`$SIRIUS_UI_SCALE`) is the factor between the two.

**Drawing.** Custom controls draw into `ImGui::GetWindowDrawList()` with the
tokens of `theme.hpp`; borders go through `widgets::crispRect` so a 1.5 px line
is whole pixels at any scale. Images are `gui::Texture`s (`gl.hpp`) drawn with
`texture.draw(dl, a, b)`, which asks the OpenGL backend for its nearest sampler
when the texture is not smooth; a plain `AddImage(texture.ref(), …)` always
gets the backend's linear sampler (`setSmooth(false)` is then ignored), so use
it only for textures that are always smooth. Dear ImGui reports API misuse — an
unbalanced push / pop, a cursor moved past a window's end — as `[imgui-error]`
lines instead of crashing; CI's headless run fails on any.

## Settings

`Settings` (`settings.hpp`) is one JSON object in
`<config>/sirius/sirius-app.json` (`%APPDATA%` on Windows, `$XDG_CONFIG_HOME` or
`~/.config` elsewhere), keyed `group/name` (`worker/python`, `recent/datasets`,
`assistant/model`, …). Dear ImGui's window layout is `imgui.ini` beside it.
Secrets never go there as plain text: `secrets::read / write` keeps them as
DPAPI-encrypted blobs in that file on Windows and in `~/.sirius/secrets.json`
(mode 0600) elsewhere.
