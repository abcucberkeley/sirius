"""An agent's side of sirius-cli's protocols, interactive over pipes (stdlib only).

    python tests/cli/agent_client.py --cli <sirius-cli> --data <source dir> [--work <dir>]

Registered as sirius::cli.agent_client by tests/CMakeLists.txt. The transcripts
that tests/cli/check_cli.cmake replays are fixed in advance; this client reacts
to what comes back, as an agent does: it waits for each answer before it sends
the next request, cancels a run it has just started, decodes the images it is
sent, and closes stdin while a run is still going.

MCP (the legacy handshake, 2025-06-18):
  1. initialize, notifications/initialized;
  2. tools/list: every schema is a closed object with a workspace argument,
     there are no view tools, and the hints say what each tool may do;
  3. open_dataset;
  4. render: the PNG's IHDR agrees with the caption, and the file is the one in
     the server's scratch directory;
  5. load_pipeline(examples/sim_bundled.sirius.toml), run {wait_s: 0} -> running;
  6. run_status {wait_s: -1} with a progress token: strictly increasing
     progress, none after the response, then succeeded;
  7. run {force: true} cancelled at once: no response for it, and run_status
     reports it cancelled (or the previous run succeeded, when the cancel
     dropped it before it started);
  8. stdin closed: exit 0 within 10 s, and the scratch directory is gone.

Session: subscribe {log: true} brings log events; cancel of a running run
brings its run_finished, cancelled; a run that is waited for reports
progress; end of input while a run is still going brings its run_finished,
then exit 0, and the scratch directory is gone.

SIRIUS_PYTHON_ENV points into a scratch directory (tests/CMakeLists.txt sets
it; otherwise --work or a temporary directory is used), so the user's own
Python environment is never looked at. Exit status 0 when every check holds.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# An MSVC Debug build reconstructs the bundled stack in a few seconds; the rest is
# slack, short enough that a hang is reported here before CTest's 600 s limit.
RESPONSE_TIMEOUT_S = 240.0
MCP_EXIT_TIMEOUT_S = 10.0
SESSION_EXIT_TIMEOUT_S = 240.0
# A run this long must have reported progress at least once while it was awaited.
PROGRESS_EXPECTED_AFTER_S = 1.5

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The tools this client calls or checks by name.
REQUIRED_TOOLS = (
    "open_dataset",
    "load_pipeline",
    "get_state",
    "render",
    "statistics",
    "run",
    "run_status",
    "cancel_run",
    "set_backend",
    "setup_worker_env",
)
VIEW_TOOLS = ("view_step", "select_step", "set_view", "focus_track")
HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


class CheckFailed(Exception):
    """A check that did not hold; the message says which."""


def check(condition: object, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def passed(what: str) -> None:
    print(f"ok: {what}", flush=True)


def failure(error: Exception, protocol: str, child: Child) -> CheckFailed:
    """What went wrong in one part, with the tail of sirius-cli's stderr. Called
    in an except block: an error other than a failed check (a member the server
    did not send, a text item that is not JSON) brings its traceback too."""
    text = str(error) if isinstance(error, CheckFailed) else traceback.format_exc().rstrip()
    return CheckFailed(f"{text}\n--- sirius-cli {protocol} stderr (last lines) ---\n{child.stderr_tail()}")


def is_under(path: Path, directory: Path) -> bool:
    # realpath on both sides: a temporary directory may sit behind a symbolic link (/tmp on macOS)
    path = Path(os.path.normcase(os.path.realpath(path)))
    directory = Path(os.path.normcase(os.path.realpath(directory)))
    return directory in path.parents


class Child:
    """sirius-cli on pipes. Requests go to stdin; each stdout line comes back as
    one JSON object through a queue; the tail of stderr is kept for the report."""

    _END = object()

    def __init__(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd
        )
        self._lines: queue.Queue[object] = queue.Queue()
        self._stderr: collections.deque[str] = collections.deque(maxlen=80)
        self.ended = False
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            self._lines.put(raw)
        self._lines.put(Child._END)

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            self._stderr.append(raw.decode("utf-8", "replace").rstrip("\r\n"))

    def send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def receive(self, deadline: float) -> dict[str, Any] | None:
        """The next message, or None once stdout has ended."""
        if self.ended:
            return None
        try:
            raw = self._lines.get(timeout=max(deadline - time.monotonic(), 0.0))
        except queue.Empty:
            raise CheckFailed("sirius-cli sent nothing more within the time allowed") from None
        if raw is Child._END:
            self.ended = True
            return None
        assert isinstance(raw, bytes)
        try:
            message = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise CheckFailed(f"a stdout line is not JSON ({e}): {raw[:400]!r}") from None
        check(isinstance(message, dict), f"a stdout line is not a JSON object: {raw[:400]!r}")
        return message

    def close_input(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()

    def wait(self, timeout: float) -> int:
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            raise CheckFailed(f"sirius-cli did not exit within {timeout:g} s of the end of its input") from None

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)


# --- MCP ------------------------------------------------------------------------


class McpClient:
    """JSON-RPC over a Child: requests carry increasing integer ids, progress
    notifications are kept by token, and responses nobody waited for are kept
    to be checked (a cancelled request must never be answered)."""

    def __init__(self, child: Child) -> None:
        self.child = child
        self.last_id = 0
        self.progress: dict[object, list[dict[str, Any]]] = collections.defaultdict(list)
        self.answered_tokens: set[object] = set()
        self.token_of: dict[int, object] = {}
        self.unexpected: list[dict[str, Any]] = []

    def send(self, method: str, params: dict[str, Any] | None = None, token: object = None) -> int:
        self.last_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self.last_id, "method": method}
        if token is not None:
            params = dict(params or {})
            params["_meta"] = {"progressToken": token}
            self.token_of[self.last_id] = token
        if params is not None:
            message["params"] = params
        self.child.send(message)
        return self.last_id

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.child.send(message)

    def note(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Files a notification; returns a response."""
        check(message.get("jsonrpc") == "2.0", f"not JSON-RPC 2.0: {message}")
        method = message.get("method")
        if method is None:
            return message
        check(method == "notifications/progress", f"the server sent an unexpected {method}: {message}")
        params = message.get("params") or {}
        token = params.get("progressToken")
        check(token in self.token_of.values(), f"progress for a token no request carried: {message}")
        check(token not in self.answered_tokens, f"progress for {token!r} after its request was answered")
        self.progress[token].append(params)
        return None

    def await_response(self, request_id: int, timeout: float = RESPONSE_TIMEOUT_S) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            message = self.child.receive(deadline)
            check(message is not None, f"sirius-cli ended its output without answering request {request_id}")
            assert message is not None
            response = self.note(message)
            if response is None:
                continue
            if type(response.get("id")) is int and response["id"] == request_id:
                if request_id in self.token_of:
                    self.answered_tokens.add(self.token_of[request_id])
                return response
            self.unexpected.append(response)

    def request(self, method: str, params: dict[str, Any] | None = None, token: object = None) -> dict[str, Any]:
        response = self.await_response(self.send(method, params, token))
        check("error" not in response, f"{method} failed: {response.get('error')}")
        check(isinstance(response.get("result"), dict), f"{method}: the result is not an object: {response}")
        return response["result"]

    def call(self, name: str, arguments: dict[str, Any], token: object = None) -> dict[str, Any]:
        result = self.request("tools/call", {"name": name, "arguments": arguments}, token)
        check(isinstance(result.get("isError"), bool), f"{name}: no isError: {result}")
        content = result.get("content")
        check(isinstance(content, list) and content, f"{name}: no content: {result}")
        check(content[0].get("type") == "text", f"{name}: the first content item is not text: {content[0]}")
        check(all(item.get("type") in ("text", "image") for item in content), f"{name}: unexpected content: {content}")
        return result

    def call_ok(self, name: str, arguments: dict[str, Any], token: object = None) -> dict[str, Any]:
        result = self.call(name, arguments, token)
        check(result["isError"] is False, f"{name} failed: {result['content'][0].get('text')}")
        value = result.get("structuredContent")
        check(isinstance(value, dict), f"{name}: no structuredContent object: {result}")
        if len(result["content"]) == 1:
            text = json.loads(result["content"][0]["text"])
            check(text == value, f"{name}: the text content differs from structuredContent")
        return value


def check_progress(notes: list[dict[str, Any]], what: str) -> None:
    values = [n.get("progress") for n in notes]
    check(all(isinstance(v, (int, float)) and 0 <= v <= 100 for v in values), f"{what}: progress out of range: {values}")
    check(all(b > a for a, b in zip(values, values[1:])), f"{what}: progress is not strictly increasing: {values}")
    check(all(n.get("total") == 100 for n in notes), f"{what}: progress without total 100: {notes}")


def check_tools(tools: dict[str, dict[str, Any]]) -> None:
    for name in REQUIRED_TOOLS:
        check(name in tools, f"tools/list has no {name}")
    for name in VIEW_TOOLS:
        check(name not in tools, f"tools/list offers the view tool {name}")
    for name, tool in tools.items():
        check(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name), f"a tool name MCP clients reject: {name!r}")
        check(0 < len(tool.get("description", "")) <= 2048, f"{name}: the description is empty or too long")
        check(isinstance(tool.get("title"), str), f"{name}: no title (2025-06-18 has them)")
        check("outputSchema" not in tool, f"{name}: has an outputSchema")
        schema = tool.get("inputSchema") or {}
        check(schema.get("type") == "object", f"{name}: the inputSchema is not an object schema")
        check(schema.get("additionalProperties") is False, f"{name}: the inputSchema allows unknown arguments")
        properties = schema.get("properties") or {}
        check("workspace" in properties, f"{name}: no workspace argument")
        check("out" not in properties, f"{name}: an out argument (images go to the scratch directory only)")
        hints = tool.get("annotations") or {}
        check(all(isinstance(hints.get(h), bool) for h in HINTS), f"{name}: annotations incomplete: {hints}")
        # open world: what reaches past this machine (an index, Hugging Face, the HPC worker)
        check(hints["openWorldHint"] == (name in ("setup_worker_env", "run")),
              f"{name}: openWorldHint is {hints['openWorldHint']}")
    check(tools["render"]["annotations"]["readOnlyHint"] is True, "render is not read-only")
    # D27: an agent picks a backend, never an endpoint or a token, whatever the argument is called
    backend_args = set(tools["set_backend"]["inputSchema"]["properties"])
    check(backend_args <= {"backend", "cuda_device", "hpc_device", "workspace"},
          f"set_backend takes more: {sorted(backend_args)}")
    setup = tools["setup_worker_env"]
    check(setup["annotations"]["destructiveHint"] is True, "setup_worker_env is not flagged destructive")
    check(
        (setup.get("_meta") or {}).get("anthropic/requiresUserInteraction") is True,
        "setup_worker_env does not ask for the user's interaction",
    )


def mcp_part(cli: list[str], data: Path, env: dict[str, str], cwd: Path) -> None:
    child = Child([*cli, "--backend", "cpu", "--plugins", "off", "mcp"], env, cwd)
    try:
        mcp = McpClient(child)

        # 1. the handshake
        init = mcp.request(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "agent_client", "version": "1.0"}},
        )
        check(init.get("protocolVersion") == "2025-06-18", f"initialize did not echo 2025-06-18: {init}")
        check(init.get("capabilities") == {"tools": {}}, f"capabilities other than tools: {init.get('capabilities')}")
        server = init.get("serverInfo") or {}
        check(server.get("name") == "sirius" and server.get("title") and server.get("version"), f"serverInfo: {server}")
        check(0 < len(init.get("instructions", "")) <= 2048, "the instructions are empty or longer than 2048 characters")
        mcp.notify("notifications/initialized")
        passed("mcp: initialize")

        # 2. the tool list
        listed = mcp.request("tools/list")
        check("nextCursor" not in listed, "tools/list is paged")
        tools = {tool.get("name"): tool for tool in listed.get("tools", [])}
        check_tools(tools)
        passed(f"mcp: tools/list ({len(tools)} tools)")

        # 3. a dataset
        opened = mcp.call_ok("open_dataset", {"path": (data / "tests" / "data" / "raw.tif").as_posix()})
        check(opened["dims"]["z"] > 0, f"open_dataset: dims {opened.get('dims')}")
        workspace = opened.get("workspace", "")
        check(re.fullmatch(r"ws_[0-9a-f]{12}", workspace), f"open_dataset: workspace {workspace!r}")
        passed("mcp: open_dataset")

        # 4. an image the agent can see, which is also the file in scratch
        scratch = Path(mcp.call_ok("get_state", {})["scratch"])
        check(scratch.is_dir(), f"get_state.scratch is not a directory: {scratch}")
        result = mcp.call("render", {"plane": "xy"})
        check(result["isError"] is False, f"render failed: {result['content'][0].get('text')}")
        content = result["content"]
        check(len(content) == 2 and content[1].get("type") == "image", f"render: content {[c.get('type') for c in content]}")
        caption = json.loads(content[0]["text"])
        check(content[1].get("mimeType") == "image/png", f"render: mimeType {content[1].get('mimeType')}")
        png = base64.b64decode(content[1]["data"], validate=True)
        check(png[:8] == PNG_SIGNATURE and png[12:16] == b"IHDR", "render: the image is not a PNG")
        size = struct.unpack(">II", png[16:24])
        check(size == (caption["width"], caption["height"]), f"render: IHDR {size}, caption {caption}")
        rendered = Path(caption["path"])
        check(is_under(rendered, scratch / "renders"), f"render: {rendered} is not under {scratch / 'renders'}")
        check(rendered.is_file() and rendered.read_bytes() == png, f"render: {rendered} is not the image that was sent")
        result = mcp.call("render", {"plane": "mip", "format": "jpeg"})
        check(result["isError"] is False and result["content"][1].get("mimeType") == "image/jpeg", "render: no JPEG")
        check(base64.b64decode(result["content"][1]["data"])[:2] == b"\xff\xd8", "render: the JPEG has no SOI marker")
        passed(f"mcp: render ({size[0]} x {size[1]} PNG, and a JPEG)")

        # 5. a pipeline, run without waiting
        bundled = (data / "examples" / "sim_bundled.sirius.toml").as_posix()
        loaded = mcp.call_ok("load_pipeline", {"path": bundled})
        check(len(loaded.get("steps", [])) == 4, f"load_pipeline: {len(loaded.get('steps', []))} steps")
        check(loaded.get("workspace") == workspace, "load_pipeline changed the workspace id")
        started = mcp.call_ok("run", {"wait_s": 0}, token="run-1")
        check(started.get("status") == "running", f"run {{wait_s: 0}}: {started.get('status')}")
        run_id = started.get("run_id")
        passed(f"mcp: run {run_id} started")

        # 6. polled to the end, with progress
        outcome = mcp.call_ok("run_status", {"wait_s": -1}, token="status-1")
        check(outcome.get("status") == "succeeded", f"run_status: {outcome.get('status')}: {outcome}")
        check(outcome.get("run_id") == run_id, f"run_status reports {outcome.get('run_id')}, not {run_id}")
        states = {step.get("kind"): step.get("state") for step in outcome.get("steps", [])}
        check(states.get("sim") == "ran", f"run_status: the SIM step {states.get('sim')}")
        notes = mcp.progress.get("status-1", [])
        check_progress(notes, "run_status")
        check_progress(mcp.progress.get("run-1", []), "run")
        seconds = float(outcome.get("seconds") or 0)
        check(notes or seconds < PROGRESS_EXPECTED_AFTER_S, f"no progress while a run of {seconds:.1f} s was awaited")
        passed(f"mcp: run_status -> succeeded in {seconds:.1f} s, {len(notes)} progress notifications")

        # 7. a run cancelled at once is never answered
        cancelled_id = mcp.send("tools/call", {"name": "run", "arguments": {"force": True}})
        mcp.notify("notifications/cancelled", {"requestId": cancelled_id, "reason": "agent_client changed its mind"})
        after = mcp.call_ok("run_status", {"wait_s": -1})
        check(mcp.request("ping") == {}, "ping")
        late = [r for r in mcp.unexpected if r.get("id") == cancelled_id]
        if after.get("run_id") == run_id:
            # The cancel dropped the forced run from the queue before it started, so
            # the last run is still the one of step 5.
            check(after.get("status") == "succeeded", f"run_status after the cancel: {after}")
            check(not late, f"the cancelled request was answered: {late}")
            how = "dropped before it started"
        elif after.get("status") == "cancelled":
            # It started: then it must have stopped, not run on with its answer suppressed.
            check(not late, f"the cancelled request was answered: {late}")
            how = f"{after.get('run_id')} cancelled"
        else:
            # The run finished before the server read the cancel. The response it had
            # already sent by then is allowed (MCP: a cancel may arrive too late), but
            # only that one.
            check(after.get("status") == "succeeded", f"the cancelled run was not cancelled: {after}")
            check(len(late) <= 1, f"the cancelled request was answered {len(late)} times: {late}")
            how = f"{after.get('run_id')} finished before the cancel arrived"
        others = [r for r in mcp.unexpected if r.get("id") != cancelled_id]
        check(not others, f"responses nobody waited for: {others}")
        passed(f"mcp: a cancelled run is not answered ({how})")

        # 8. end of input
        child.close_input()
        code = child.wait(MCP_EXIT_TIMEOUT_S)
        deadline = time.monotonic() + 10
        while (message := child.receive(deadline)) is not None:
            response = mcp.note(message)
            check(response is None, f"a response after the end of input: {response}")
        check(code == 0, f"sirius-cli mcp exited with {code}")
        check(not scratch.exists(), f"the scratch directory was left behind: {scratch}")
        passed("mcp: end of input -> exit 0, scratch removed")
    except Exception as e:
        raise failure(e, "mcp", child) from None
    finally:
        child.stop()


# --- session ------------------------------------------------------------------------


class SessionClient:
    """sirius-session/1 over a Child: one request at a time, events kept in order."""

    def __init__(self, child: Child) -> None:
        self.child = child
        self.last_id = 0
        self.events: list[dict[str, Any]] = []

    def ready(self) -> dict[str, Any]:
        message = self.child.receive(time.monotonic() + RESPONSE_TIMEOUT_S)
        check(message is not None and message.get("event") == "ready", f"the first line is not the ready event: {message}")
        assert message is not None
        return message

    def note(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Files an event; returns a response."""
        event = message.get("event")
        if event is None:
            return message
        if event == "log":
            check(message.get("source") in ("workbench", "worker"), f"a log event from {message.get('source')!r}")
            check(isinstance(message.get("line"), str), f"a log event without a line: {message}")
        elif event == "progress":
            check(isinstance(message.get("fraction"), (int, float)), f"a progress event without a fraction: {message}")
        elif event == "run_finished":
            check(isinstance(message.get("run_id"), str), f"run_finished without a run_id: {message}")
            check(isinstance(message.get("result"), dict), f"run_finished without a result: {message}")
        else:
            raise CheckFailed(f"an unexpected event: {message}")
        self.events.append(message)
        return None

    def send(self, method: str, params: dict[str, Any] | None = None) -> int:
        self.last_id += 1
        message: dict[str, Any] = {"id": self.last_id, "method": method}
        if params is not None:
            message["params"] = params
        self.child.send(message)
        return self.last_id

    def await_response(self, request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + RESPONSE_TIMEOUT_S
        while True:
            message = self.child.receive(deadline)
            check(message is not None, f"sirius-cli ended its output without answering request {request_id}")
            assert message is not None
            response = self.note(message)
            if response is not None:
                check(response.get("id") == request_id, f"expected the answer to {request_id}, got {response}")
                return response

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.await_response(self.send(method, params))
        check(response.get("ok") is True, f"{method} failed: {response.get('error')}")
        check(isinstance(response.get("result"), dict), f"{method}: the result is not an object: {response}")
        return response["result"]

    def wait_event(
        self, matches: Callable[[dict[str, Any]], bool], what: str, timeout: float = RESPONSE_TIMEOUT_S
    ) -> dict[str, Any]:
        for event in self.events:
            if matches(event):
                return event
        deadline = time.monotonic() + timeout
        while True:
            message = self.child.receive(deadline)
            check(message is not None, f"sirius-cli ended its output before {what}")
            assert message is not None
            response = self.note(message)
            check(response is None, f"a response nobody waited for: {response}")
            if matches(message):
                return message

    def finished(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == "run_finished" and e["run_id"] == run_id]


def session_part(cli: list[str], data: Path, env: dict[str, str], cwd: Path) -> None:
    child = Child([*cli, "--backend", "cpu", "--plugins", "off", "session"], env, cwd)
    try:
        session = SessionClient(child)
        ready = session.ready()
        check(ready.get("protocol") == "sirius-session/1", f"ready: protocol {ready.get('protocol')}")
        workspace = ready.get("workspace", "")
        check(re.fullmatch(r"ws_[0-9a-f]{12}", workspace), f"ready: workspace {workspace!r}")
        check(isinstance(ready.get("tools"), int) and ready["tools"] > 0, f"ready: tools {ready.get('tools')}")
        session.request("ping")
        scratch = Path(session.request("get_state")["scratch"])
        check(scratch.is_dir(), f"get_state.scratch is not a directory: {scratch}")
        session.request("subscribe", {"progress": True, "log": True})
        loaded = session.request("load_pipeline", {"path": (data / "examples" / "sim_bundled.sirius.toml").as_posix()})
        check(loaded.get("workspace") == workspace, "load_pipeline changed the workspace id")
        passed("session: ready, subscribed, pipeline loaded")

        # a running run, cancelled
        first = session.request("run", {"wait_s": 0})
        check(first.get("status") == "running", f"run {{wait_s: 0}}: {first.get('status')}")
        cancel = session.request("cancel", {})
        check(isinstance(cancel.get("cancelled"), bool), f"cancel: {cancel}")
        finished = session.wait_event(
            lambda e: e["event"] == "run_finished" and e["run_id"] == first["run_id"], f"run_finished for {first['run_id']}"
        )
        # A run that had already ended when the cancel came has succeeded. One that
        # was still going may also succeed: its last step can be past its final
        # cancel check when the cancel is answered, and a finished job counts as
        # succeeded. It can only have been cancelled if the cancel said so.
        expected = ("cancelled", "succeeded") if cancel["cancelled"] else ("succeeded",)
        check(finished["result"].get("status") in expected, f"after cancel {cancel}: {finished['result']}")
        passed(f"session: cancel -> run_finished ({finished['result'].get('status')})")

        # a run waited for, with progress
        waited_id = session.send("run", {"wait_s": -1, "force": True})
        response = session.await_response(waited_id)
        check(response.get("ok") is True, f"run {{wait_s: -1}} failed: {response.get('error')}")
        outcome = response["result"]
        check(outcome.get("status") == "succeeded", f"run {{wait_s: -1}}: {outcome.get('status')}")
        fractions = [e["fraction"] for e in session.events if e["event"] == "progress" and e.get("id") == waited_id]
        check(all(0 <= f <= 1 for f in fractions), f"progress fractions out of range: {fractions}")
        check(all(b >= a for a, b in zip(fractions, fractions[1:])), f"progress goes backwards: {fractions}")
        seconds = float(outcome.get("seconds") or 0)
        check(fractions or seconds < PROGRESS_EXPECTED_AFTER_S, f"no progress events during a run of {seconds:.1f} s")
        logs = [e for e in session.events if e["event"] == "log"]
        check(logs, "no log events although subscribed to them")
        passed(f"session: a run waited for, {len(fractions)} progress and {len(logs)} log events")

        # end of input while a run is going: the session waits for it
        last = session.request("run", {"wait_s": 0, "force": True})
        check(last.get("status") == "running", f"run {{wait_s: 0}}: {last.get('status')}")
        status = session.request("status")
        check(isinstance(status.get("running"), bool), f"status: {status}")
        check(status.get("workspace") == workspace, f"status: workspace {status.get('workspace')}")
        child.close_input()
        finished = session.wait_event(
            lambda e: e["event"] == "run_finished" and e["run_id"] == last["run_id"],
            f"run_finished for {last['run_id']} after the end of input",
        )
        check(finished["result"].get("status") == "succeeded", f"after the end of input: {finished['result']}")
        deadline = time.monotonic() + SESSION_EXIT_TIMEOUT_S
        while (message := child.receive(deadline)) is not None:
            check(session.note(message) is None, f"a response after the end of input: {message}")
        code = child.wait(SESSION_EXIT_TIMEOUT_S)
        check(code == 0, f"sirius-cli session exited with {code}")
        check(not session.finished(outcome.get("run_id", "")), "run_finished for a run whose call already returned it")
        check(len(session.finished(last["run_id"])) == 1, f"run_finished for {last['run_id']} more than once")
        check(not scratch.exists(), f"the scratch directory was left behind: {scratch}")
        passed("session: end of input waited for the run, exit 0, scratch removed")
    except Exception as e:
        raise failure(e, "session", child) from None
    finally:
        child.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", nargs="+", required=True, help="sirius-cli (a wrapper and its arguments may come first)")
    parser.add_argument("--data", type=Path, required=True, help="the source tree, for tests/data and examples")
    parser.add_argument("--work", type=Path, help="a scratch directory (default: a temporary one)")
    args = parser.parse_args(argv)
    # sirius-cli runs in the scratch directory, not in the one the client was started
    # from, so relative paths are made absolute here. A part of --cli that names no
    # file is left alone: it may be a program found on the PATH, such as a wrapper.
    cli = [str(Path(part).resolve()) if Path(part).is_file() else part for part in args.cli]
    data = args.data.resolve()
    if args.work:
        args.work = args.work.resolve()

    with tempfile.TemporaryDirectory(prefix="sirius-agent-client-") as temporary:
        work = args.work if args.work else Path(temporary)
        work.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if args.work or not env.get("SIRIUS_PYTHON_ENV"):
            env["SIRIUS_PYTHON_ENV"] = str(work / "pyenv")
        try:
            mcp_part(cli, data, env, work)
            session_part(cli, data, env, work)
        except (CheckFailed, OSError) as e:
            print(f"FAILED: {e}", flush=True)
            return 1
    print("agent_client: every check passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
