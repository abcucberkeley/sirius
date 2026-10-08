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

`--tool` serves the whole `ToolApi` table (`core/tool_api.hpp`), `export_result`,
`statistics` and `probe` included, so a run driven through the window can be
compared with `sirius-cli`'s by file and by numbers rather than off a
screenshot. `export_result` refuses `needs_download` for an output that stays
on the cluster, as File ▸ Export result asks the user first; `download: true`
is that answer written down. An argument the named tool does not take is
dropped and reported in the result's `warnings`, as it has always been in a
session: a typo used to be taken here in silence.

## Layout of the code

| Files | What they are |
| --- | --- |
| `main.cpp` | command line, scripting, the objects' lifetimes |
| `app.hpp/.cpp` | window, title/menu bar, docks, status bar, actions and shortcuts, dialogs and boxes, every command |
| `bridge.hpp/.cpp` | workbench observer → revisions and signals; runs and tasks on a worker thread |
| `theme.hpp/.cpp` | design tokens, fonts, the ImGui / ImPlot style |
| `widgets/controls`, `widgets/icons`, `widgets/code_editor` | the design's controls, icon set and the plugin editor |
| `settings`, `secret_store`, `platform`, `worker_launcher`, `http`, `gl` | services: persistent settings, secrets, OS dialogs, the Python worker, HTTP(S), textures and PNG |
| `cluster_link` | the cluster session (`core/cluster.hpp`): ssh's prompts in a box, the cluster profiles in the settings file (`core/cluster_profiles.hpp`), the HPC backend and cluster datasets once connected, the status-bar indicator and the title bar's Cluster button |
| `viewer/*` | toolbar, tool strip, ortho / 3D / compare views, dims strip, the volume loader and ray caster |
| `panels/*` | operations, parameters, diagnostics, log, help, assistant |
| `dialogs/*` | open, folder dataset, export, training export, preferences, model hub, plugin manager, the Python environment offer (`python_env_dialog.cpp`), connect to cluster and the cluster's file browser (`cluster_dialog.cpp`), the settings file in the code editor (`settings_editor_dialog.cpp`) |

Everything is in namespace `sirius::app::gui`; includes are written from `app/`
(`"core/workbench.hpp"`, `"imgui/theme.hpp"`). The fonts and the icons are in
`app/resources`, copied beside the executable by the build and installed with it.

What `sirius-cli` needs as well lives in the core, and the GUI uses it from
there under the names it had:

- `platform.hpp` keeps the file dialogs and re-exports the OS helpers of
  `core/host.hpp` it uses (the home, config and temporary directories, the
  executable's directory, the process id, the environment, `findPython`,
  `makePath`, whole-file reads and atomic writes) with using-declarations;
- child processes are `core/process.hpp` (`ChildProcess`);
- `worker_launcher.hpp` is `class WorkerLauncher : public LocalWorker`
  (`core/local_worker.hpp`), whose constructor only wires in the settings:
  `worker/python`, `worker/dir`, `--allow-install` for the model hub, and the
  hint that points at Preferences ▸ Compute;
- `viewer/display_model.hpp` re-exports `core/display_model.hpp`, so that
  `sirius-cli` renders exactly what the viewer draws.

## The Python environment offer

When the local worker cannot start — its Python lacks numpy, there is no
Python at all, or SIRIUS's own environment no longer runs — the start-failure
handler `main.cpp` installs (on the connecting thread, so it only posts)
calls `offerPythonEnvironment` on the GUI thread. It shows "Set up Python for
SIRIUS" at most once a session, never in an unattended (scripted) run, and
only while `worker/offerEnvironment` is on; `SIRIUS_PYTHON_OFFER=always` or
`never` overrides all of that, for screenshots and tests. The dialog works
out a plan (`pyenv::planSetup`) on a thread of its own, runs the setup as a
`Bridge::startTask` (so not during a run), shows the installer's lines as
they come, and on success stops the worker and reloads the plugins. Its
variants cover a missing package in a found Python, no Python at all, an
interpreter the user or `$SIRIUS_PYTHON` named, and an environment that needs
a repair or an update. Preferences ▸ Compute has the same environment's
status line with Set up / Update / Repair / Recreate, Check, Remove and Open
folder.

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

**Docks.** Operations, Parameters, the viewer, Diagnostics, the log and the
assistant are plain Dear ImGui dock windows in one dockspace over the main
window (`App::Impl::drawDockWindows`), and every dock shows its tab bar, even
for one panel: the tab is how a panel is moved. Dropped on another dock's
centre target it becomes a tab there, on an edge target it splits that dock,
and dragged away it floats; out of the main window it gets a window of its own
where multi-viewport works (not on Wayland, and not while the monitors have
different scales: `App::Impl::updateViewports`), and those windows take
dropped files as the main window does. A floating panel docks again by its
title bar. `buildDefaultLayout` makes the design's arrangement, the viewer in
the central node, when there is no saved one and for *Window ▸ Reset layout*;
after that the arrangement is the user's. The Window menu shows and hides the
panels (the viewer has no close box). Nothing may assume where a panel is: the
maximised diagnostics cover the viewer, below its tab bar, only while it is
docked in the main window with its tab in front, and a viewer that is not
drawn does not keep the frame loop awake for its playback.

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

`Settings` (`core/settings_store.hpp`, spelled `settings()` here through
`settings.hpp`) is one TOML file, `<config>/sirius/sirius-app.toml` (`%APPDATA%`
on Windows, `$XDG_CONFIG_HOME` or `~/.config` elsewhere), held in memory as one
JSON object keyed `group/name` (`worker/python`, `recent/datasets`,
`cluster/<profile>`, …), which the file has as `[group]` tables
(`core/settings_toml.hpp`). A file that does not read is never written over: the
application starts with its defaults and says where the file is wrong;
*Preferences ▸ Edit settings file…* (`dialogs/settings_editor_dialog.cpp`) edits it
with the line and column of each problem. A `sirius-app.json` of before is
converted once and renamed `sirius-app.json.migrated`. Dear ImGui's window layout is `imgui.ini` beside it
(`Settings::layoutPath`); a layout saved when the docks hid their tabs is
loaded with them shown again, and written back so.
Secrets never go there: `secrets::read / write` keeps them as DPAPI-encrypted
blobs in `secrets.json` beside it on Windows and in `~/.sirius/secrets.json`
(mode 0600) elsewhere.

Nothing about SIRIUS's Python environment is kept in the settings: its state
is read from the environment itself (`core/python_env.hpp`), which
`sirius-cli` shares. The settings only hold the user's choices about it:
`worker/offerEnvironment` (offer a setup when the worker cannot start,
default on), `worker/environmentExtras` (the scipy / scikit-image checkbox)
and `worker/useUv` (use uv when it is installed, default on).
