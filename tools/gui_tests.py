#!/usr/bin/env python3
"""Scripted tests of the Qt layer.

The core is covered by tests/test_app_*.cpp, which run without a display. The
widgets were covered by one screenshot that only proved the window came up.
This drives the real application through the hooks it already has for
scripting -- ``--tool`` for the assistant API, ``--action`` for a menu item,
``--key`` for a key press on a named widget, ``--stroke`` and ``--wheel`` for
mouse input on the XY pane, ``--drop`` for a drag and drop, ``--record`` for a
machine-readable log of what happened -- and asserts on what comes back rather
than on the process surviving.

    python3 tools/gui_tests.py --app build/linux-gcc-app-dev/app/sirius-app

Runs offscreen; no display needed.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "tests" / "data" / "raw.tif"
PIPELINE = ROOT / "examples" / "sim_bundled.sirius.toml"

_TOOL = re.compile(r"^tool (\w+) -> ", re.M)


class Failure(Exception):
    pass


class Skip(Exception):
    """The scenario cannot run here (a platform, a permission); said, not failed."""


def check_offscreen_plugin(app: Path) -> Optional[str]:
    """Why ``-platform offscreen`` will not start, if it will not.

    windeployqt does not copy the offscreen platform plugin, so a Windows build
    that runs fine on screen fails every scenario here with a Qt abort that says
    nothing about the cause. Say it up front instead.
    """
    if os.name != "nt":
        return None
    platforms = app.parent / "platforms"
    if any((platforms / name).is_file() for name in ("qoffscreen.dll", "qoffscreend.dll")):
        return None
    return (
        f"no offscreen platform plugin in {platforms}\n"
        "      windeployqt does not deploy it; copy it from the Qt kit, e.g.\n"
        f'      cp "$QTDIR/plugins/platforms/qoffscreend.dll" "{platforms}"'
    )


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill the application and anything it started.

    sirius-app spawns ``python -m sirius_worker`` for the Python steps. Killing
    only the application leaves that worker running, and the next scenario then
    competes with it for the port and the GPU.
    """
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run(app: Path, args: List[str], timeout: int = 300, env: Optional[Dict[str, str]] = None) -> str:
    """Run the application once, offscreen, and return everything it printed."""
    # Settings of this run's own. Otherwise every scenario reads the settings of
    # whoever is logged in -- dock widths, backend, cache policy -- and saves its
    # own layout back over them when it exits. A scenario that passes here and
    # fails on another machine, or the other way round, is the usual sign. A
    # scenario that seeds or inspects the settings names its directory
    # (isolated_settings); the rest get a scratch one the run removes.
    env = dict(env or {})
    settings = env.pop(SETTINGS_DIR, "scratch")
    full = [str(app), "-platform", "offscreen", "--settings", settings, *args]
    # The scenarios read the application's own qInfo lines. sirius-app is a
    # WIN32 (no console) binary, and Qt's default handler then sends those to
    # OutputDebugString rather than the pipe, so every run comes back empty
    # unless stderr logging is asked for by name.
    environment = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "QT_FORCE_STDERR_LOGGING": "1",
        **env,
    }
    # Files, not pipes. A worker that outlives the application inherits the
    # write end of a pipe, and reading one to EOF then blocks for as long as
    # that worker lives -- past this function's own timeout, for ever. A file
    # has no such end to wait for: the run is over when the application is,
    # whatever it left behind. (That leftover does keep the file open, and on
    # Windows an open file cannot be deleted, so the cleanup forgives it.)
    box = Path(tempfile.mkdtemp(prefix="sirius-run-"))
    try:
        out_path, err_path = box / "stdout.txt", box / "stderr.txt"
        with out_path.open("wb") as out, err_path.open("wb") as err:
            process = subprocess.Popen(full, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=environment)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_tree(process)
                raise Failure(f"timed out after {timeout}s: {' '.join(full)}") from None
        # Qt writes its messages in the local 8-bit encoding: a decode error in
        # one log line must not look like a failed scenario.
        text = out_path.read_text(errors="replace") + err_path.read_text(errors="replace")
    finally:
        shutil.rmtree(box, ignore_errors=True)
    if code != 0:
        raise Failure(f"exit {code}: {' '.join(full)}\n{text[-4000:]}")
    return text


# The key of an isolated_settings() environment that run() turns into
# --settings <dir> rather than passing on to the application.
SETTINGS_DIR = "--settings"


def isolated_settings(tmp: Path, name: str) -> Dict[str, str]:
    """A settings directory, a HOME and an XDG_CONFIG_HOME of the scenario's own.

    run() gives the application --settings <directory>, which keeps its
    settings there as an INI file on every platform, and its secret store
    beside them; HOME and XDG_CONFIG_HOME are the scenario's too, so that
    anything still reaching for the user's own (~/.sirius, ~/.config) finds a
    directory the scenario can look into instead.
    """
    home, config, settings = tmp / f"{name}-home", tmp / f"{name}-config", tmp / f"{name}-settings"
    for d in (home, config, settings):
        d.mkdir(parents=True, exist_ok=True)
    return {"HOME": str(home), "XDG_CONFIG_HOME": str(config), SETTINGS_DIR: str(settings)}


def settings_file(env: Dict[str, str]) -> Path:
    """Where QSettings("sirius", "sirius-app") lives under an isolated_settings() environment."""
    return Path(env[SETTINGS_DIR]) / "sirius" / "sirius-app.ini"


def secret_store(env: Dict[str, str]) -> Path:
    """The secret store file under an isolated_settings() environment (not on Windows: DPAPI in the settings)."""
    return Path(env[SETTINGS_DIR]) / "secrets.json"


# No proxy between the application and a FakeModelServer.
LOCAL_ONLY = {"http_proxy": "", "HTTP_PROXY": "", "no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost"}


class FakeModelServer:
    """An OpenAI-compatible model server on 127.0.0.1, for the assistant scenarios.

    It lists one model ("foo") and answers the chats in turn with the messages
    in `chats` (the last one from then on; "hello" by default), as plain JSON,
    which the client takes even when it asked for a stream. It keeps the
    method, path and Authorization header of every request it saw, and the
    body of every chat request.
    """

    def __init__(self, chats: Optional[List[Dict[str, Any]]] = None) -> None:
        seen: List[Tuple[str, str, Optional[str]]] = []
        bodies: List[Dict[str, Any]] = []
        replies = chats or [{"role": "assistant", "content": "hello"}]
        self.requests = seen
        self.bodies = bodies

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def answer(self, body: Dict[str, Any]) -> None:
                seen.append((self.command, self.path, self.headers.get("Authorization")))
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
                self.answer({"data": [{"id": "foo"}], "models": [{"name": "foo"}]})

            def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                bodies.append(body)
                message = replies[min(len(bodies), len(replies)) - 1]
                self.answer({"choices": [{"message": message, "finish_reason": "stop"}]})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self) -> FakeModelServer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


def tool_results(output: str) -> Dict[str, List[Any]]:
    """Every `tool <name> -> {json}` the run printed, by tool name."""
    out: Dict[str, List[Any]] = {}
    for m in _TOOL.finditer(output):
        rest = output[m.end() :]
        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(rest)
        except ValueError:
            continue
        out.setdefault(m.group(1), []).append(value)
    return out


def only(results: Dict[str, List[Any]], name: str) -> Any:
    values = results.get(name)
    if not values:
        raise Failure(f"the run printed no result for '{name}'")
    return values[-1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def image_is_not_blank(path: Path) -> None:
    check(path.is_file() and path.stat().st_size > 5000, f"{path} missing or suspiciously small")
    try:
        from PIL import Image  # noqa: PLC0415 - optional, the size check stands without it
    except ImportError:
        return
    with Image.open(path) as im:
        colours = im.convert("RGB").getcolors(maxcolors=1 << 20)
    check(colours is not None and len(colours) > 32, f"{path} has almost no colours: a blank window")


# --- the scenarios ---------------------------------------------------------


def test_ortho_view_shows_the_dataset(app: Path, tmp: Path) -> None:
    shot = tmp / "ortho.png"
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"set_view","args":{"mode":"ortho"}}',
            "--tool",
            '{"name":"get_state","args":{}}',
            "--screenshot",
            str(shot),
            "--settle",
            "900",
            "--quit-after",
            "6000",
        ],
    )
    state = only(tool_results(out), "get_state")
    check(state["dataset"] is not None, "the dataset did not open")
    check(state["dataset"]["shape"].startswith("c1 t1 z135"), f"unexpected shape {state['dataset']['shape']}")
    image_is_not_blank(shot)


def test_every_view_mode_renders(app: Path, tmp: Path) -> None:
    for mode in ("ortho", "3d", "compare"):
        shot = tmp / f"mode_{mode}.png"
        out = run(
            app,
            [
                "--dataset",
                str(RAW),
                "--tool",
                json.dumps({"name": "set_view", "args": {"mode": mode}}),
                "--tool",
                '{"name":"get_state","args":{}}',
                "--screenshot",
                str(shot),
                "--settle",
                "900",
                "--quit-after",
                "6000",
            ],
        )
        state = only(tool_results(out), "get_state")
        check(state["view"]["mode"].lower() == mode, f"view is {state['view']['mode']}, asked for {mode}")
        # Qt's offscreen platform has no OpenGL widgets, so the 3D pane may be
        # a notice saying so rather than a rendering. Switching to it and
        # drawing the window without crashing is what this can honestly check;
        # the slice views are painted by QPainter and always have content.
        if mode == "3d":
            check(shot.is_file() and shot.stat().st_size > 5000, f"{shot} missing or suspiciously small")
        else:
            image_is_not_blank(shot)


def test_compare_shows_raw_beside_the_result(app: Path, tmp: Path) -> None:
    # the compare pane once drew the raw side at the reconstruction's subsample
    # factor, which made it blurry; both sides must report the same field
    shot = tmp / "compare.png"
    out = run(
        app,
        [
            "--pipeline",
            str(PIPELINE),
            "--tool",
            '{"name":"run","args":{}}',
            "--tool",
            '{"name":"set_view","args":{"mode":"compare"}}',
            "--tool",
            '{"name":"get_state","args":{}}',
            "--screenshot",
            str(shot),
            "--settle",
            "1200",
            "--quit-after",
            "9000",
        ],
    )
    state = only(tool_results(out), "get_state")
    check(state["view"]["mode"].lower() == "compare", "compare mode did not take")
    image_is_not_blank(shot)


def test_painting_reaches_the_labels(app: Path, tmp: Path) -> None:
    # the recording is the machine-readable account of what the widgets did
    log = tmp / "paint.jsonl"
    out = run(
        app,
        [
            "--record",
            str(log),
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"add_step","args":{"kind":"classic"}}',
            "--tool",
            '{"name":"run","args":{}}',
            "--action",
            "Paint labels",
            "--stroke",
            "20,20,30,30,6",
            "--tool",
            '{"name":"get_step","args":{"step":3}}',
            "--settle",
            "900",
            "--quit-after",
            "9000",
        ],
    )
    step = only(tool_results(out), "get_step")
    check(step["kind"] == "classic", f"step 3 is {step['kind']}")
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    check("step_ran" in kinds, "the segmentation did not run")
    paints = [e for e in events if e["event"] == "paint"]
    check(len(paints) >= 3, f"the stroke produced {len(paints)} paint events")
    check(all(p["voxels"] > 0 for p in paints), "a paint event changed nothing")
    check(any(p["x"] != paints[0]["x"] for p in paints), "the stroke never moved")


def test_the_wheel_zooms(app: Path, tmp: Path) -> None:
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--wheel",
            "32,32,3",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--settle",
            "600",
            "--quit-after",
            "5000",
        ],
    )
    state = only(tool_results(out), "get_state")
    check(float(state["view"]["zoom"]) > 1.0, f"zoom is {state['view']['zoom']} after scrolling in")


def test_the_wheel_zooms_about_the_cursor_in_compare(app: Path, tmp: Path) -> None:
    # Compare zoomed about a point of the XY pane, which is hidden there (a
    # stale view, another size), so the image slid out from under the cursor.
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"set_view","args":{"mode":"compare"}}',
            "--wheel",
            "10,10,3",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--settle",
            "600",
            "--quit-after",
            "6000",
        ],
        env=isolated_settings(tmp, "compare-wheel"),
    )
    state = only(tool_results(out), "get_state")
    check(float(state["view"]["zoom"]) > 1.0, f"zoom is {state['view']['zoom']} after scrolling in")
    m = re.search(r"wheel: .* on compareStepPane .*under the cursor now \(([-\d.]+), ([-\d.]+)\)", out)
    check(m is not None, "the wheel did not report the compare pane")
    x, y = float(m.group(1)), float(m.group(2))
    check(abs(x - 10.0) < 0.05 and abs(y - 10.0) < 0.05, f"voxel (10, 10) under the cursor became ({x}, {y})")


def test_a_dropped_file_opens(app: Path, tmp: Path) -> None:
    out = run(app, ["--drop", str(RAW), "--tool", '{"name":"get_state","args":{}}', "--settle", "900", "--quit-after", "6000"])
    state = only(tool_results(out), "get_state")
    check(state["dataset"] is not None, "dropping a TIFF did not open it")
    check(state["dataset"]["name"].startswith("raw"), f"opened {state['dataset']['name']}")


def test_files_named_on_the_command_line_open(app: Path, tmp: Path) -> None:
    # what a file manager's "Open with" (app/linux/sirius-app.desktop, Exec=sirius-app %F) passes
    out = run(app, [str(RAW), "--tool", '{"name":"get_state","args":{}}', "--settle", "900", "--quit-after", "6000"])
    state = only(tool_results(out), "get_state")
    check(state["dataset"] is not None and state["dataset"]["name"].startswith("raw"), f"the dataset is {state['dataset']}")
    out = run(app, [str(PIPELINE), "--tool", '{"name":"get_state","args":{}}', "--settle", "900", "--quit-after", "6000"])
    kinds = [s["kind"] for s in only(tool_results(out), "get_state")["steps"]]
    check("sim" in kinds, f"the pipeline file did not open: steps {kinds}")


def test_an_invalid_step_says_so_in_the_error_colour(app: Path, tmp: Path) -> None:
    # A step whose parameters do not validate shows why in its row, in the
    # error colour. A universal "* { color }" rule in the style sheet used to
    # repaint every palette colour in body text, so the line was there but read
    # like any other summary. The same pipeline with and without a missing OTF
    # differs only by that line, so the red it adds is the line's text.
    try:
        from PIL import Image  # noqa: PLC0415 - optional, as in image_is_not_blank
    except ImportError:
        raise Skip("needs Pillow to read the screenshot") from None
    text = PIPELINE.read_text()
    check("../tests/data/otf.tif" in text, f"{PIPELINE} no longer names ../tests/data/otf.tif")
    data = (ROOT / "tests" / "data").as_posix()
    valid = tmp / "valid.sirius.toml"
    valid.write_text(text.replace("../tests/data/", data + "/"))
    invalid = tmp / "invalid.sirius.toml"
    missing = (tmp / "missing-otf.tif").as_posix()
    invalid.write_text(text.replace("../tests/data/otf.tif", missing).replace("../tests/data/", data + "/"))

    def red_pixels(pipeline: Path) -> int:
        shot = tmp / f"{pipeline.stem}.png"
        run(app, ["--pipeline", str(pipeline), "--screenshot", str(shot), "--settle", "900", "--quit-after", "6000"])
        image_is_not_blank(shot)
        with Image.open(shot) as im:
            return sum(1 for r, g, b in im.convert("RGB").getdata() if r > 140 and g < 100 and b < 80 and r - g > 90)

    added = red_pixels(invalid) - red_pixels(valid)
    check(added > 60, f"the invalid step added {added} red pixels: its error line is not in the error colour")


def test_a_run_that_fails_in_the_worker_ends_a_headless_run(app: Path, tmp: Path) -> None:
    # A step that raises while it runs (as a CUDA error inside a model does)
    # used to leave a headless --run waiting for its 600 s deadline: the "Run
    # failed" message box blocked in the window's runFinished handler, and the
    # handler that ends the headless run is connected after it.
    env = isolated_settings(tmp, "failing")
    plugins = Path(env["HOME"]) / ".sirius" / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    (plugins / "fails_while_running.py").write_text(
        "STEP = {'kind': 'fails_while_running', 'name': 'Fails while running', 'group': 'Intensity', 'params': []}\n"
        "\n"
        "def run(data, params, meta, ctx):\n"
        "    raise RuntimeError('CUDA error: an illegal memory access was encountered')\n"
    )
    pipeline = tmp / "failing.sirius.toml"
    pipeline.write_text(
        "version = 1\n\n"
        '[[steps]]\nkind = "load"\nname = "Load"\n[steps.params]\n'
        f'path = "{RAW.as_posix()}"\n\n'
        '[[steps]]\nkind = "fails_while_running"\nname = "Fails"\n'
    )
    shot = tmp / "failing.png"
    args = ["--pipeline", str(pipeline), "--run", "--tool", '{"name":"get_log","args":{}}', "--screenshot", str(shot)]
    try:
        out = run(app, args, timeout=120, env=env)
    except Failure as e:
        out = str(e)
        if out.startswith("timed out"):
            raise Failure("a run that failed in the worker did not end the headless run (the Run failed box blocked)") from None
        if "Plugins unavailable" in out or "not loaded" in out:
            raise Skip("no Python worker to serve the failing step (set SIRIUS_PYTHON to an interpreter with numpy)") from None
        check(out.startswith("exit 1:"), f"expected the failed run's exit status 1, got: {out[:300]}")
    else:
        raise Failure("the failing step's run exited 0")
    check("illegal memory access" in out, "the worker's error is not in the log")
    check(shot.is_file(), "no screenshot: the run ended without the grab")


def write_moving_blobs(path: Path) -> None:
    """A (t, z, y, x) = (6, 4, 48, 48) uint16 ImageJ hyperstack: three bright
    cubes stepping two pixels a frame, the third missed in frame 3. Pure Python,
    so the scenario needs nothing installed; it is small enough to be quick."""
    t_, z_, y_, x_ = 6, 4, 48, 48
    starts = [(8, 8, 0), (30, 10, 1), (20, 34, -1)]  # y, x, direction along y
    pages = []
    for t in range(t_):
        for z in range(z_):
            plane = bytearray(struct.pack("<H", 100) * (y_ * x_))
            if 1 <= z <= 2:
                for k, (y0, x0, dy) in enumerate(starts):
                    if k == 2 and t == 3:
                        continue
                    cy, cx = y0 + dy * 2 * t, x0 + 2 * t
                    for yy in range(cy - 2, cy + 3):
                        for xx in range(cx - 2, cx + 3):
                            struct.pack_into("<H", plane, 2 * (yy * x_ + xx), 3000)
            pages.append(bytes(plane))
    desc = f"ImageJ=1.11a\nimages={t_ * z_}\nframes={t_}\nslices={z_}\nhyperstack=true\n\0".encode()
    out = bytearray(b"II*\0" + struct.pack("<I", 0))
    link = 4
    for i, page in enumerate(pages):
        data = len(out)
        out += page
        tags = [(256, 4, 1, x_), (257, 4, 1, y_), (258, 3, 1, 16), (259, 3, 1, 1), (262, 3, 1, 1)]
        if i == 0:
            text = len(out)
            out += desc + (b"\0" if len(desc) % 2 else b"")
            tags.append((270, 2, len(desc), text))
        tags += [(273, 4, 1, data), (277, 3, 1, 1), (278, 4, 1, y_), (279, 4, 1, len(page))]
        ifd = len(out)
        struct.pack_into("<I", out, link, ifd)
        out += struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHII", *tag) for tag in tags)
        link = len(out)
        out += struct.pack("<I", 0)
    path.write_bytes(bytes(out))


def test_the_tracks_tab_follows_a_track(app: Path, tmp: Path) -> None:
    # Every dataset in tests/data is one time point, so nothing else here ever
    # produced tracked labels: the Tracks tab, the trajectories and follow mode
    # were never drawn by a headless run. Segment and track a synthetic clip
    # with the built-in tracker (no worker), then drive the table with a key.
    clip = tmp / "blobs.tif"
    write_moving_blobs(clip)
    pipeline = tmp / "tracks.sirius.toml"
    pipeline.write_text(
        "version = 1\n\n"
        '[[steps]]\nkind = "load"\nname = "Load"\n[steps.params]\n'
        f'path = "{clip.as_posix()}"\nvoxel_x = 0.2\nvoxel_y = 0.2\nvoxel_z = 0.5\n\n'
        '[[steps]]\nkind = "classic"\nname = "Segment"\n[steps.params]\nsigma = 0.0\n\n'
        '[[steps]]\nkind = "track"\nname = "Track"\n[steps.params]\nmax_distance = 1.5\n'
    )
    shot = tmp / "tracks.png"
    out = run(
        app,
        [
            "--pipeline",
            str(pipeline),
            "--tool",
            '{"name":"run","args":{}}',
            "--tool",
            '{"name":"view_step","args":{"step":3}}',
            "--tool",
            '{"name":"select_step","args":{"step":3}}',
            "--tool",
            '{"name":"list_tracks","args":{}}',
            "--tool",
            '{"name":"focus_track","args":{"id":1}}',
            "--tool",
            '{"name":"set_view","args":{"t":4,"follow_track":true}}',
            "--tool",
            '{"name":"get_state","args":{}}',
            "--key",
            "trackTable=Down",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--screenshot",
            str(shot),
            "--settle",
            "900",
            "--quit-after",
            "20000",
        ],
    )
    results = tool_results(out)
    ran = only(results, "run")
    check(ran.get("ok") is True, f"the pipeline did not run: {ran}")
    tracks = only(results, "list_tracks")
    check(tracks.get("total") == 3, f"expected 3 tracks, got {tracks}")
    check(tracks.get("with_gaps") == 1, f"the track missed in frame 3 should show one gap: {tracks}")
    check(only(results, "focus_track").get("ok") is True, "focus_track found no track 1")
    before, after = results["get_state"][-2:]
    check(
        before["view"]["selected_label"] == 1 and before["view"]["follow_track"],
        f"track 1 not selected and followed: {before['view']}",
    )
    focused(out, "trackTable=Down")
    check(after["view"]["selected_label"] not in (0, 1), f"Down in the Tracks table chose no other track: {after['view']}")
    image_is_not_blank(shot)


def test_menu_actions_reach_the_view(app: Path, tmp: Path) -> None:
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--action",
            "Labels overlay",
            "--action",
            "Physical z scaling",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--settle",
            "600",
            "--quit-after",
            "5000",
        ],
    )
    view = only(tool_results(out), "get_state")["view"]
    check(view["labels"] is True, "Labels overlay did not turn on")
    check(view.get("physical_z") is False, "Physical z scaling did not turn off")


def test_a_preset_fills_the_fields(app: Path, tmp: Path) -> None:
    # a preset is values, not a mode: the step holds what it wrote and the
    # change is undoable like any other
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"add_step","args":{"kind":"classic"}}',
            "--tool",
            '{"name":"apply_preset","args":{"step":3,"preset":"Filament network"}}',
            "--tool",
            '{"name":"get_step","args":{"step":3}}',
            "--settle",
            "600",
            "--quit-after",
            "6000",
        ],
    )
    params = only(tool_results(out), "get_step")["params"]
    check(params["enhance"] == "Neurites (Meijering)", f"enhance is {params['enhance']}")
    check(params["post"] == "Connected components", f"post is {params['post']}")
    check(abs(float(params["enhance_sigma"]) - 0.8) < 1e-9, f"sigma is {params['enhance_sigma']}")


def test_a_token_the_secret_store_refuses_stays_in_the_settings(app: Path, tmp: Path) -> None:
    # A token still in the plaintext settings moves into the secret store file
    # at start-up (the HPC token is read then). When the store cannot take it,
    # the settings entry is the only copy: deleting it anyway worked for one
    # session and lost the token at the next launch. A store that exists but
    # does not parse must not be written over either -- that dropped every
    # other secret in it.
    if os.name == "nt":
        raise Skip("Windows keeps secrets with DPAPI inside the settings, not in a file with a mode")
    if os.geteuid() == 0:
        raise Skip("root writes through a read-only file mode")
    env = isolated_settings(tmp, "secrets")
    conf = settings_file(env)
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text("[hpc]\ntoken=tok_LEGACY\n")
    store = secret_store(env)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{}\n")
    store.chmod(0o400)
    try:
        for launch in (1, 2):
            out = run(app, ["--quit-after", "1500"], env=env)
            check("tok_LEGACY" in conf.read_text(), f"launch {launch} deleted the plaintext token the store refused")
        check(out.count("could not move 'hpc/token'") == 1, "the refused migration was not reported exactly once in a launch")
    finally:
        store.chmod(0o600)
    corrupt = '{"hub/token": "kept", '
    store.write_text(corrupt)
    run(app, ["--quit-after", "1500"], env=env)
    check(store.read_text() == corrupt, f"a store that does not parse was written over: {store.read_text()!r}")
    check("tok_LEGACY" in conf.read_text(), "the plaintext token went although the store could not take it")


def test_a_run_with_settings_of_its_own_leaves_the_users_alone(app: Path, tmp: Path) -> None:
    # --settings moved QSettings, but the secret store stayed in ~/.sirius: a
    # scripted run read the user's real tokens, and migrated the plaintext token
    # of its own settings into the user's store.
    if os.name == "nt":
        raise Skip("Windows keeps secrets with DPAPI inside the settings, which --settings moves as a whole")
    env = isolated_settings(tmp, "own")
    users_store = Path(env["HOME"]) / ".sirius" / "secrets.json"
    users_store.parent.mkdir(parents=True, exist_ok=True)
    users_store.write_text('{"sentinel": "the user\'s"}\n')
    users_conf = Path(env["XDG_CONFIG_HOME"]) / "sirius" / "sirius-app.conf"
    users_conf.parent.mkdir(parents=True, exist_ok=True)
    users_conf.write_text("[hpc]\ntoken=tok_USERS\n")
    before = (users_store.read_bytes(), users_store.stat().st_mtime_ns, users_conf.read_bytes())
    conf = settings_file(env)
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text("[hpc]\ntoken=tok_RUNS\n")
    out = run(app, ["--quit-after", "1500"], env=env)
    check(f"settings: {conf}" in out, f"the run did not use {conf}")
    after = (users_store.read_bytes(), users_store.stat().st_mtime_ns, users_conf.read_bytes())
    check(after == before, "the run wrote to the user's own ~/.sirius/secrets.json or settings")
    store = secret_store(env)
    check(store.is_file() and "hpc/token" in store.read_text(), f"the run's plaintext token did not move into {store}")
    check("tok_RUNS" not in conf.read_text(), "the run's plaintext token stayed in its settings once stored")


def focused(out: str, spec: str) -> None:
    """The --key press `spec` ("Z plane=Right") reached the widget it named."""
    target, key = spec.split("=", 1)
    check(f"key {key} to {target}: focus yes" in out, f"--key {spec} did not get the focus onto {target}")


def test_arrow_keys_reach_the_focused_control(app: Path, tmp: Path) -> None:
    # Left / Right are also Segment > Previous / Next flagged label. The
    # shortcut stood back for a focused slider or slice pane only after Qt had
    # already given it the key press, so the arrows moved nothing at all.
    env = isolated_settings(tmp, "arrows")
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"get_state","args":{}}',
            "--key",
            "Z plane=Right",
            "--key",
            "xyPane=Right",
            "--tool",
            '{"name":"get_state","args":{}}',
            "--settle",
            "600",
            "--quit-after",
            "6000",
        ],
        env=env,
    )
    focused(out, "Z plane=Right")
    focused(out, "xyPane=Right")
    before, after = tool_results(out)["get_state"][-2:]
    z0, z1 = before["view"]["z"], after["view"]["z"]
    check(z1 == z0 + 1, f"Right on the Z slider: z {z0} -> {z1}")
    x0, x1 = before["view"]["crosshair_x"], after["view"]["crosshair_x"]
    check(x1 == x0 + 1, f"Right on the XY pane: crosshair x {x0} -> {x1}")


def test_space_in_a_read_only_view_leaves_the_step_alone(app: Path, tmp: Path) -> None:
    # Space is Edit > Enable / skip step. Pressed in the (read-only) session
    # log, which pages with it, it silently skipped the selected step. With the
    # focus on a widget that has no use for Space it still does.
    env = isolated_settings(tmp, "space")
    out = run(
        app,
        [
            "--dataset",
            str(RAW),
            "--tool",
            '{"name":"add_step","args":{"kind":"classic"}}',
            "--tool",
            '{"name":"select_step","args":{"step":3}}',
            "--key",
            "Session log=Space",
            "--tool",
            '{"name":"get_step","args":{"step":3}}',
            "--key",
            "xyPane=Space",
            "--tool",
            '{"name":"get_step","args":{"step":3}}',
            "--settle",
            "600",
            "--quit-after",
            "6000",
        ],
        env=env,
    )
    focused(out, "Session log=Space")
    focused(out, "xyPane=Space")
    in_log, on_pane = tool_results(out)["get_step"][-2:]
    check(in_log["enabled"] is True, "Space in the session log skipped the selected step")
    check(on_pane["enabled"] is False, "Space on the XY pane no longer skips the selected step (the shortcut is gone)")


def test_ollama_never_gets_the_api_key(app: Path, tmp: Path) -> None:
    # One stored key serves OpenRouter and custom servers. With the provider
    # switched to Ollama (whose key field is disabled) every request still
    # carried it -- to a local server, or a remote one over plain http.
    env = isolated_settings(tmp, "ollama-key")
    env.update(LOCAL_ONLY)
    env.update({"OPENROUTER_API_KEY": "", "SIRIUS_LLM_API_KEY": ""})
    with FakeModelServer() as server:
        conf = settings_file(env)
        conf.parent.mkdir(parents=True, exist_ok=True)
        # a plaintext key from before the secret store: read, and migrated, at start-up
        conf.write_text(f"[assistant]\nprovider=ollama\nbaseUrl={server.url}\napiKey=sk-or-STORED\n")
        run(app, ["--ask", "hello", "--quit-after", "6000"], env=env)
        asked = any(method == "POST" for method, _, _ in server.requests)
        check(asked, f"the question never reached the server: {server.requests}")
        sent = [(method, path) for method, path, auth in server.requests if auth]
        check(not sent, f"requests to Ollama carried the API key: {sent}")


def test_an_api_key_from_the_environment_is_not_stored(app: Path, tmp: Path) -> None:
    # OPENROUTER_API_KEY is used when no key is stored, and it was written
    # into ~/.sirius/secrets.json by the first save of any assistant
    # setting -- which start-up does as soon as the server lists its models.
    env = isolated_settings(tmp, "env-key")
    env.update(LOCAL_ONLY)
    env.update({"OPENROUTER_API_KEY": "sk-or-FROMENV"})
    with FakeModelServer() as server:
        conf = settings_file(env)
        conf.parent.mkdir(parents=True, exist_ok=True)
        conf.write_text(f"[assistant]\nprovider=openrouter\nbaseUrl={server.url}\n")
        run(app, ["--ask", "hello", "--quit-after", "6000"], env=env)
        used = ("POST", "/v1/chat/completions", "Bearer sk-or-FROMENV") in server.requests
        check(used, f"the environment's key was not used: {server.requests}")
    check("model=foo" in conf.read_text(), "the model the server listed was not saved (the save this is about never ran)")
    store = secret_store(env)
    check(not store.exists() or "assistant/apiKey" not in store.read_text(), f"the environment's key was written to {store}")
    check("secrets/assistant" not in conf.read_text(), "the environment's key was written to the settings (DPAPI)")


def test_a_cut_off_tool_call_is_answered_not_run(app: Path, tmp: Path) -> None:
    # A reply cut off at the token limit leaves a tool call with arguments that
    # are not JSON. They ran as {} -- a cut-off `run` ran every step -- and the
    # model heard the result as if its call had worked.
    env = isolated_settings(tmp, "cut-off")
    env.update(LOCAL_ONLY)
    env.update({"OPENROUTER_API_KEY": "", "SIRIUS_LLM_API_KEY": ""})
    call = {"id": "call_1", "type": "function", "function": {"name": "set_view", "arguments": '{"mode": "3'}}
    cut = {"role": "assistant", "content": "", "tool_calls": [call]}
    with FakeModelServer(chats=[cut, {"role": "assistant", "content": "done"}]) as server:
        conf = settings_file(env)
        conf.parent.mkdir(parents=True, exist_ok=True)
        conf.write_text(f"[assistant]\nprovider=custom\nbaseUrl={server.url}\nmodel=foo\naskBeforeActing=false\n")
        run(app, ["--dataset", str(RAW), "--ask", "show it in 3D", "--quit-after", "8000"], env=env)
    answers = [m for body in server.bodies for m in body.get("messages", []) if m.get("role") == "tool"]
    check(bool(answers), f"the cut-off call was never answered ({len(server.bodies)} chat requests)")
    check(
        "not a valid JSON object" in answers[0]["content"],
        f"the cut-off call ran with no arguments: {answers[0]['content'][:200]}",
    )


SCENARIOS = [
    test_ortho_view_shows_the_dataset,
    test_every_view_mode_renders,
    test_compare_shows_raw_beside_the_result,
    test_painting_reaches_the_labels,
    test_the_wheel_zooms,
    test_the_wheel_zooms_about_the_cursor_in_compare,
    test_a_dropped_file_opens,
    test_files_named_on_the_command_line_open,
    test_an_invalid_step_says_so_in_the_error_colour,
    test_a_run_that_fails_in_the_worker_ends_a_headless_run,
    test_the_tracks_tab_follows_a_track,
    test_menu_actions_reach_the_view,
    test_a_preset_fills_the_fields,
    test_a_token_the_secret_store_refuses_stays_in_the_settings,
    test_a_run_with_settings_of_its_own_leaves_the_users_alone,
    test_arrow_keys_reach_the_focused_control,
    test_space_in_a_read_only_view_leaves_the_step_alone,
    test_ollama_never_gets_the_api_key,
    test_an_api_key_from_the_environment_is_not_stored,
    test_a_cut_off_tool_call_is_answered_not_run,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path, help="the sirius-app binary")
    parser.add_argument("--only", default="", help="run just the scenarios whose name contains this")
    args = parser.parse_args()
    if not args.app.is_file():
        print(f"no such application: {args.app}", file=sys.stderr)
        return 2
    if problem := check_offscreen_plugin(args.app):
        print(f"cannot run offscreen: {problem}", file=sys.stderr)
        return 2

    chosen = [s for s in SCENARIOS if not args.only or args.only in s.__name__]
    if not chosen:
        print(f"--only {args.only!r} matched none of: {', '.join(s.__name__ for s in SCENARIOS)}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="sirius-gui-"))
    failures = 0
    try:
        for scenario in chosen:
            name = scenario.__name__.removeprefix("test_").replace("_", " ")
            try:
                scenario(args.app, tmp)
            except Failure as e:
                failures += 1
                print(f"FAIL  {name}\n      {e}", file=sys.stderr, flush=True)
            except Skip as e:
                print(f"skip  {name}\n      {e}", flush=True)
            else:
                print(f"ok    {name}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(chosen) - failures}/{len(chosen)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
