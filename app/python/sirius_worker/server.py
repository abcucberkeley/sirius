"""The compute worker: one TCP listener, one client at a time, one job at a
time, streamed progress, cancellation.

Requests (see protocol.py for the framing):

    hello       {token, protocol_version}   -> capabilities (incl. protocol_version)
    ping        {}                          -> {}
    list_plugins   {}                       -> {plugins: [spec + file (+ error)], dirs}
    reload_plugins {}                       -> the same, after re-importing every plugin file
    model_info  {path | spec}               -> format, input_shape, output_shape, dtype, size_bytes, channels_out;
                                               cellpose: / microsam: specs -> {format, available, install_hint}
    hub_search  {query, limit, filter?}     -> {models: [{id, downloads, likes, tags, last_modified, pipeline_tag}]}
    hub_files   {repo}                      -> {repo, files: [{name, size, model}]}
    hub_download {repo, file}               -> "progress"* then {path, bytes, spec} (cancellable like a run)
                                               (hub_* take an optional `token` for gated / private repositories)
    models_list {}                          -> {cache, models: [{spec, path, bytes}]}
    models_delete {path}                    -> {path, bytes, removed_directories}  (cache only)
    install     {family, dry_run?}          -> "progress"* (one frame per output line) then
                                               {ok, returncode, available, command, tail}: pip / conda installs
                                               the family's package into the worker's Python
    model_prepare {spec}                    -> "progress"* then {spec, path, cached}: fetches the weights now
    run         {kind, params} + tensors    -> "progress"* then "result" (+ tensors)
    cancel      {id}                        -> {} (the cancelled run replies with an error "cancelled")
    shutdown    {}                          -> {} and the server exits

Model specs (params.model of torch_segment, model_info): a local .pt / .pts /
.pth / .onnx path; ``hf:<repo>[:<file>]`` (downloaded into $SIRIUS_MODEL_CACHE
or ~/.sirius/models); ``cellpose:<model>``; ``microsam:<model_type>`` -- see
models.py.

Run kinds and their tensors:

    torch_segment  in  "input" (z, y, x) float32
                   out "prob"  (C, z, y, x) float32, result {channels, seconds}   (file / hf models)
                   out "labels" (z, y, x) uint32 [+ "prob" (1, z, y, x)], result {labels, format}
                                                                              (cellpose / micro-sam)
    sim            in  "input" (sections, y, x) or (c, t, sections, y, x)
                   out "output" reconstructed, same rank; result {fits, seconds}
    <numpy kinds>  in  "input" (c, t, z, y, x) [+ "labels" (t, z, y, x) uint32]
                   out "output" (+ "labels", "prob"), result {meta, info, seconds}
    plugin         params {plugin: kind, params, meta}; in "input" (c, t, z, y, x) [+ "labels"]
                   out "output" (+ "labels", "image<i>"), result {meta, diagnostics, seconds}

The reader loop runs on the connection's thread and the job on a worker
thread, so a cancel request is read while a run is in progress. Every reply
carries the request's id.

Trust model (app/python/SECURITY.md): whoever completes `hello` can run code
here, so the listener refuses a non-loopback address without a token, the
token is compared in constant time, and everything a peer sends before its
`hello` is capped at protocol.MAX_PREAUTH_FRAME. `hello` also exchanges
protocol.PROTOCOL_VERSION and refuses a peer that speaks another one.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import platform
import select
import socket
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import __version__
from . import models as model_hub
from . import plugins as plugin_registry
from .protocol import MAX_PREAUTH_FRAME, PROTOCOL_VERSION, ProtocolError, encode_frame, read_frame
from .steps import workbench

log = logging.getLogger("sirius_worker")

# kinds served through run_step plus the two with their own tensor contracts
_SPECIAL_KINDS = ("torch_segment", "sim", "btrack", "skimage_seg")


class _Cancelled(Exception):
    pass


class WorkerServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, token: str = "", device: str = "auto",
                 max_clients: int = 1) -> None:
        self.host = host
        self.port = port
        self.token = token or ""
        self.device = device
        self.max_clients = max_clients
        self._listener: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._job_lock = threading.Lock()
        self._job: Optional[Dict[str, Any]] = None

    # --- lifecycle ------------------------------------------------------------

    def bind(self) -> int:
        # Reaching this port is the whole of the authorisation model, so a
        # port anyone on the network can reach must at least need the token.
        if not self.token and not is_loopback(self.host):
            raise ValueError(
                f"refusing to listen on {self.host or '0.0.0.0'} without a token: any host that can reach this "
                f"port could run code as {_username()}. Pass --token (or set $SIRIUS_TOKEN), for example "
                "SIRIUS_TOKEN=$(openssl rand -hex 16); or bind 127.0.0.1 and reach the worker through an SSH "
                "tunnel (app/python/SECURITY.md)")
        if not self.token:
            log.warning("no token: every client that can connect to %s:%s is served. That is only safe on a "
                        "machine you are the only user of; pass --token to require a shared secret.",
                        self.host, self.port or "<auto>")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # On Winsock SO_REUSEADDR lets a second socket bind a port that is
        # already listening (and receive the client's hello, token included);
        # SO_EXCLUSIVEADDRUSE is the option that means what POSIX's does.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(4)
        s.settimeout(0.5)
        self._listener = s
        self.port = s.getsockname()[1]
        return self.port

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        assert self._listener is not None
        log.info("listening on %s:%d (device %s)", self.host, self.port, self.resolved_device())
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = self._listener.accept()
                except (socket.timeout, TimeoutError):  # noqa: UP041 -- distinct before 3.10
                    # socket.timeout is its own OSError subclass before 3.10
                    continue
                except OSError:
                    break
                peer = _peer(addr)
                log.info("client %s connected", peer)
                try:
                    self._serve_client(conn, peer)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    log.info("client %s disconnected", peer)
        finally:
            self.close()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

    # --- capabilities ----------------------------------------------------------

    def resolved_device(self) -> str:
        return workbench().resolve_device(self.device)

    def capabilities(self) -> Dict[str, Any]:
        wb = workbench()
        methods = ["hello", "ping", "model_info", "run", "cancel", "shutdown", "list_plugins", "reload_plugins",
                   "hub_search", "hub_files", "hub_download", "models_list", "models_delete", "install",
                   "model_prepare"]
        kinds = list(_SPECIAL_KINDS) + [k for k in wb.step_kinds() if k not in _SPECIAL_KINDS] + ["plugin"]
        methods += [f"run:{k}" for k in kinds]
        cuda = False
        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                cuda = True
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                device = f"cuda:{idx} · {props.name} · {props.total_memory / 2**30:.0f} GB"
        except Exception:  # noqa: BLE001 - torch is optional
            pass
        if not cuda:
            try:
                import sirius  # type: ignore

                if sirius.cuda_available():
                    cuda = True
                    p = sirius.device_properties(sirius.Device.cuda(0))
                    device = f"cuda:0 · {p.name} · {p.total_memory_bytes / 2**30:.0f} GB"
            except Exception:  # noqa: BLE001
                pass
        if self.resolved_device() == "cpu" or not cuda:
            device = f"cpu · {os.cpu_count() or 1} threads"
        return {
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "methods": methods,
            "cuda": cuda and self.resolved_device().startswith("cuda"),
            "device": device,
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": _module_version("torch"),
            "sirius": _module_version("sirius"),
            "workbench": getattr(wb, "__source_file__", getattr(wb, "__file__", "")),
        }

    def _handshake(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Check a `hello`: the token first (constant time, so a wrong one
        leaks nothing through timing), then the protocol version. Returns
        (ok, message); the message is what the client is told and logged."""
        if self.token:
            supplied = params.get("token", "")
            supplied = supplied if isinstance(supplied, str) else ""
            if not hmac.compare_digest(supplied.encode("utf-8"), self.token.encode("utf-8")):
                return False, "authentication failed: bad token"
        # Same version required on both ends. A peer that does not send the
        # field predates the handshake and counts as version 0.
        raw = params.get("protocol_version", 0)
        theirs = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
        if theirs != PROTOCOL_VERSION:
            fix = ("update the SIRIUS application that connects to this worker"
                   if theirs < PROTOCOL_VERSION else
                   "update sirius_worker on this machine (app/python)")
            return False, (f"protocol version mismatch: this worker speaks version {PROTOCOL_VERSION}, "
                           f"the client speaks version {theirs}; {fix}")
        return True, ""

    # --- one connection ----------------------------------------------------------

    # A peer that connects and never completes `hello` is dropped after this
    # long: serving is serial, so a silent connection would otherwise hold
    # the port for everyone else.
    HELLO_TIMEOUT = 15.0
    # How often the connection loop looks at the stop flag while idle.
    IDLE_POLL = 0.5

    def _serve_client(self, conn: socket.socket, peer: str = "?") -> None:
        conn.settimeout(None)
        send_lock = threading.Lock()
        hello_deadline = time.monotonic() + self.HELLO_TIMEOUT
        # Nothing is served, with or without a token, until `hello` has agreed
        # on the protocol version -- and until then the peer's frames are held
        # to MAX_PREAUTH_FRAME.
        authenticated = False

        def send(header: Dict[str, Any], tensors=None) -> None:
            data = encode_frame(header, tensors)
            with send_lock:
                conn.sendall(data)

        def reply(rid, result: Dict[str, Any], tensors=None) -> None:
            send({"id": rid, "type": "result", "result": result}, tensors)

        def error(rid, message: str) -> None:
            send({"id": rid, "type": "error", "message": message})

        while not self._stop.is_set():
            # Wait for the next frame without blocking in recv: a blocked recv
            # ignores stop() (SIGTERM, the launcher) for as long as the client
            # stays silent, and a silent pre-hello peer would hold the worker
            # forever. Once bytes are on the wire the frame is read in full.
            try:
                readable, _, _ = select.select([conn], [], [], self.IDLE_POLL)
            except (OSError, ValueError):
                break
            if not readable:
                if not authenticated and time.monotonic() > hello_deadline:
                    log.warning("client %s sent no hello within %.0f s; dropped", peer, self.HELLO_TIMEOUT)
                    break
                continue
            try:
                if authenticated:
                    header, tensors = read_frame(conn)
                else:
                    header, tensors = read_frame(conn, MAX_PREAUTH_FRAME, MAX_PREAUTH_FRAME)
            except ConnectionError:
                break
            except ProtocolError as e:
                log.warning("protocol error from %s: %s", peer, e)
                try:
                    error(None, f"protocol error: {e}")
                except OSError:
                    pass
                break
            except OSError as e:
                log.info("connection error: %s", e)
                break
            except Exception as e:  # noqa: BLE001 - a decoder bug ends the connection, not the worker
                log.exception("unexpected error reading a frame from %s", peer)
                try:
                    error(None, f"internal error: {_message(e)}")
                except OSError:
                    pass
                break
            rid = header.get("id")
            method = str(header.get("method", ""))
            params = header.get("params")
            if not isinstance(params, dict):
                params = {}
            if header.get("type", "request") != "request":
                error(rid, f"unexpected frame type '{header.get('type')}'")
                continue
            try:
                if method == "hello":
                    ok, message = self._handshake(params)
                    if not ok:
                        error(rid, message)
                        log.warning("%s: %s", peer, message)
                        break
                    authenticated = True
                    reply(rid, self.capabilities())
                elif not authenticated:
                    error(rid, "not authenticated: send 'hello' with the worker token first")
                elif method == "ping":
                    reply(rid, {"time": time.time()})
                elif method == "shutdown":
                    # Privileged: it ends every job on this worker. Left
                    # reachable from a tunnelled (non-loopback) client because
                    # that is how the application stops the worker it started
                    # on a cluster node; logged so the log says who did it.
                    log.warning("privileged request: shutdown, from %s", peer)
                    reply(rid, {})
                    self.stop()
                    break
                elif method == "model_info":
                    reply(rid, self.model_info(str(params.get("spec") or params.get("path") or params.get("model") or "")))
                elif method == "hub_search":
                    model_hub.set_hub_token(str(params.get("token", "") or ""))
                    reply(rid, {"models": model_hub.hub_search(str(params.get("query", "")),
                                                               int(params.get("limit", 25) or 25),
                                                               str(params.get("filter", "") or ""))})
                elif method == "hub_files":
                    model_hub.set_hub_token(str(params.get("token", "") or ""))
                    repo = str(params.get("repo", ""))
                    reply(rid, {"repo": repo, "files": model_hub.hub_files(repo)})
                elif method == "hub_download":
                    # a job like "run": progress frames stream while the reader keeps taking cancel
                    model_hub.set_hub_token(str(params.get("token", "") or ""))
                    repo = str(params.get("repo", ""))
                    filename = str(params.get("file") or params.get("filename") or "")
                    self._start_job(rid, f"hub_download {repo}", send,
                                    lambda progress, cancel, repo=repo, filename=filename:
                                        self._download(repo, filename, progress, cancel))
                elif method == "install":
                    family = str(params.get("family", ""))
                    dry_run = bool(params.get("dry_run", False))
                    # Privileged: a real install runs pip / conda in this
                    # interpreter, i.e. arbitrary package code as this user.
                    # models.install refuses it unless --allow-install was given.
                    log.warning("privileged request: install '%s'%s, from %s (allow_install=%s)", family,
                                " (dry run)" if dry_run else "", peer, model_hub.ALLOW_INSTALL)
                    self._start_job(rid, f"install {family}", send,
                                    lambda progress, cancel, family=family, dry_run=dry_run: (model_hub.install(
                                        family, progress, cancelled=cancel.is_set, dry_run=dry_run), None))
                elif method == "model_prepare":
                    model_hub.set_hub_token(str(params.get("token", "") or ""))
                    spec = str(params.get("spec", ""))
                    self._start_job(rid, f"model_prepare {spec}", send,
                                    lambda progress, cancel, spec=spec:
                                        (model_hub.prepare(spec, progress, cancelled=cancel.is_set), None))
                elif method == "models_list":
                    reply(rid, {"cache": str(model_hub.cache_dir()), "models": model_hub.list_cached_models()})
                elif method == "models_delete":
                    reply(rid, model_hub.delete_cached_model(str(params.get("path", ""))))
                elif method in ("list_plugins", "reload_plugins"):
                    reply(rid, self.plugin_list(reload=method == "reload_plugins", extra=params.get("dirs")))
                elif method == "cancel":
                    target = params.get("id", header.get("target"))
                    self._cancel(target)
                    reply(rid, {"cancelled": target})
                elif method == "run":
                    # a step fetching a gated model sends the token with the request
                    model_hub.set_hub_token(str(params.get("token", "") or ""))
                    self._start_run(rid, params, tensors, send)
                else:
                    error(rid, f"unknown method '{method}'")
            except Exception as e:  # noqa: BLE001 - every failure is reported to the client
                log.error("%s failed: %s", method, e)
                log.debug("%s", traceback.format_exc())
                try:
                    error(rid, _message(e))
                except OSError:
                    break
        # the connection is gone: cancel whatever is still running
        self._cancel(None)
        job = self._current_job()
        if job is not None:
            job["thread"].join(timeout=30)

    # --- jobs ----------------------------------------------------------------------

    def _current_job(self) -> Optional[Dict[str, Any]]:
        with self._job_lock:
            return self._job

    def _cancel(self, rid) -> None:
        with self._job_lock:
            job = self._job
        if job is None:
            return
        if rid is None or job["id"] == rid:
            job["cancel"].set()
            log.info("cancel requested for %s", job["id"])

    def _start_run(self, rid, params: Dict[str, Any], tensors: Dict[str, np.ndarray], send) -> None:
        self._start_job(rid, str(params.get("kind", "")), send,
                        lambda progress, cancel: self._execute(rid, params, tensors, cancel, progress))

    def _start_job(self, rid, label: str, send, work) -> None:
        """Run `work(progress, cancel_event) -> (result, tensors)` on its own
        thread; `send(header, tensors)` is the connection's locked sender,
        shared by progress frames and the reply."""
        with self._job_lock:
            if self._job is not None and self._job["thread"].is_alive():
                send({"id": rid, "type": "error", "message": f"busy: request {self._job['id']} is still running"})
                return
            cancel = threading.Event()
            job: Dict[str, Any] = {"id": rid, "cancel": cancel, "thread": None}
            self._job = job

        def progress(fraction: float, message: str = "") -> None:
            try:
                send({"id": rid, "type": "progress", "fraction": float(fraction), "message": str(message)})
            except OSError:
                pass

        token = model_hub.current_request_token()   # the request's, read on the connection thread

        def run() -> None:
            t0 = time.time()
            model_hub.set_hub_token(token or "")   # this thread's copy: the connection may move on
            try:
                result, out = work(progress, cancel)
                if cancel.is_set():
                    send({"id": rid, "type": "error", "message": "cancelled"})
                else:
                    result["seconds"] = time.time() - t0
                    send({"id": rid, "type": "result", "result": result}, out)
            except Exception as e:  # noqa: BLE001 - every failure is reported to the client
                if cancel.is_set() or isinstance(e, _Cancelled) or e.__class__.__name__ == "Cancelled":
                    message = "cancelled"
                else:
                    log.error("%s failed: %s", label, e)
                    log.debug("%s", traceback.format_exc())
                    message = _message(e)
                try:
                    send({"id": rid, "type": "error", "message": message})
                except OSError:
                    pass
            finally:
                with self._job_lock:
                    if self._job is job:
                        self._job = None

        thread = threading.Thread(target=run, name=f"sirius-run-{rid}", daemon=True)
        job["thread"] = thread
        thread.start()

    # --- models ----------------------------------------------------------------------

    def model_info(self, spec: str) -> Dict[str, Any]:
        """Facts about a model spec. Family specs report availability; an hf:
        file not in the cache yet is described without downloading it (the
        first run, or hub_download, fetches it)."""
        ms = model_hub.parse_spec(spec)
        if ms.family in ("cellpose", "microsam"):
            return model_hub.family_info(spec)
        if ms.family == "hf":
            have = model_hub.cached_path(ms.name, ms.filename)
            if have is None:
                available, hint = model_hub.family_available("hf")
                return {"spec": ms.text(), "format": "hf", "repo": ms.name, "file": ms.filename, "cached": False,
                        "available": available, "install_hint": hint, "path": ""}
            info = workbench().model_info(have, self.resolved_device())
            info.update({"spec": ms.text(), "repo": ms.name, "file": ms.filename, "cached": True})
            return info
        return workbench().model_info(ms.name, self.resolved_device())

    def _download(self, repo: str, filename: str, progress, cancel: threading.Event):
        def report(fraction: float, message: str = "") -> None:
            if cancel.is_set():
                raise _Cancelled()
            progress(fraction, message)

        path = model_hub.hub_download(repo, filename, report)
        return {"path": path, "bytes": os.path.getsize(path), "repo": repo, "file": filename or os.path.basename(path),
                "spec": f"hf:{repo}:{filename or os.path.basename(path)}"}, None

    # --- plugins ---------------------------------------------------------------------

    def plugin_list(self, reload: bool = False, extra=None) -> Dict[str, Any]:
        if reload:
            job = self._current_job()
            if job is not None and job["thread"].is_alive():
                # re-importing a plugin file while a job may be executing it
                raise RuntimeError(f"busy: request {job['id']} is still running; reload the plugins afterwards")
        with self._job_lock:
            cached = getattr(self, "_plugins", None)
        if cached is None or reload:
            plugins, dirs = plugin_registry.load_all(list(extra or []))
            with self._job_lock:
                self._plugins = plugins
                self._plugin_dirs = dirs
            for pl in plugins:
                if pl.error:
                    log.warning("plugin %s: %s", pl.file, pl.error.splitlines()[0])
                else:
                    log.info("plugin %s from %s", pl.kind, pl.file)
        return {"plugins": [pl.describe() for pl in self._plugins], "dirs": list(self._plugin_dirs)}

    def _plugin(self, kind: str):
        if getattr(self, "_plugins", None) is None:
            self.plugin_list()
        for pl in self._plugins:
            if pl.kind == kind and not pl.error:
                return pl
        for pl in self._plugins:
            if pl.kind == kind:
                raise ValueError(f"plugin '{kind}' failed to load: {pl.error}")
        raise ValueError(f"unknown plugin '{kind}' (Process ▸ Reload plugins after adding it)")

    def _execute(self, rid, params: Dict[str, Any], tensors: Dict[str, np.ndarray], cancel: threading.Event,
                 progress):
        wb = workbench()
        kind = str(params.get("kind", ""))
        p = params.get("params") or {}
        device = self.resolved_device()

        def cancelled() -> bool:
            return cancel.is_set()

        def check() -> None:
            if cancel.is_set():
                raise _Cancelled()

        if kind == "plugin":
            plugin = self._plugin(str(params.get("plugin", "")))
            arr = tensors.get("input")
            if arr is None:
                raise ValueError("run plugin: missing tensor 'input'")
            out, out_labels, diagnostics, meta_out = plugin_registry.run_plugin(
                plugin, arr, p, params.get("meta") or {}, tensors.get("labels"), progress=progress, cancelled=cancelled)
            check()
            tensors_out: Dict[str, np.ndarray] = {"output": out}
            if out_labels is not None:
                tensors_out["labels"] = out_labels
            images = []
            for i, im in enumerate(diagnostics.pop("images", []) or []):
                data = np.ascontiguousarray(np.asarray(im.get("data"), dtype=np.float32))
                if data.ndim != 2:
                    continue
                tensors_out[f"image{i}"] = data
                images.append({"title": str(im.get("title", f"image {i}")), "meta": str(im.get("meta", "")),
                               "log": bool(im.get("log", False)), "tensor": f"image{i}"})
            diagnostics["images"] = images
            return {"meta": _jsonable(meta_out), "diagnostics": _jsonable(diagnostics), "device": device}, tensors_out

        if kind == "torch_segment":
            volume = _tensor(tensors, "input", 3)
            spec = str(p.get("model") or p.get("model_path") or "")
            if model_hub.is_family_spec(spec):
                # cellpose / micro-SAM produce instance labels themselves; the
                # application skips its threshold / watershed stage for these
                labels, prob = model_hub.run_family(spec, volume, p, device, progress=progress, cancelled=cancelled)
                check()
                out_t: Dict[str, np.ndarray] = {"labels": np.ascontiguousarray(labels, dtype=np.uint32)}
                if prob is not None:
                    out_t["prob"] = np.ascontiguousarray(prob, dtype=np.float32)
                return {"labels": int(labels.max()) if labels.size else 0, "model": spec,
                        "format": model_hub.parse_spec(spec).family, "device": device}, out_t
            _, path = model_hub.resolve(spec, progress)   # hf: specs download on first use
            model = wb.load_model(path, device)
            tile = _triple(p.get("tile"), (32, 256, 256))
            ov = p.get("overlap", 32)
            overlap = _triple(ov, (4, 32, 32)) if isinstance(ov, (list, tuple, str)) else (max(1, int(ov) // 8), int(ov), int(ov))
            prob = wb.tiled_inference(volume, model, tile, overlap, device, int(p.get("pad_to", 1) or 1),
                                      str(p.get("activation", "auto")), bool(p.get("normalize", True)),
                                      progress=progress, cancelled=cancelled)
            check()
            return {"channels": int(prob.shape[0]), "device": device}, {"prob": prob}

        if kind == "skimage_seg":
            # the scikit-image methods the application does not implement
            # natively: they take a volume and hand back instance labels
            from . import skimage_seg

            volume = _tensor(tensors, "input", 3)
            labels, info = skimage_seg.run(volume, p, progress=progress, cancelled=cancelled)
            check()
            return {**_jsonable(info), "device": "cpu"}, {"labels": np.ascontiguousarray(labels, dtype=np.uint32)}

        if kind == "btrack":
            # Bayesian tracking: the labels go over as they are and come back
            # renumbered by track, so every btrack specific stays on this side.
            from . import tracking as tracking_backends

            marks = tensors.get("labels")
            if marks is None:
                raise ValueError("run btrack: missing tensor 'labels'")
            shape = marks.shape
            if marks.ndim == 3:
                marks = marks[:, np.newaxis]   # (t, y, x) -> (t, 1, y, x)
            voxel = p.get("voxel_um") or (params.get("meta") or {}).get("voxel_um") or (1.0, 1.0, 1.0)
            out, info = tracking_backends.run_btrack(marks, tuple(float(v) for v in voxel), p, progress=progress)
            check()
            return info, {"labels": np.ascontiguousarray(out.reshape(shape), dtype=np.uint32)}

        if kind == "sim":
            raw = tensors.get("input")
            if raw is None:
                raise ValueError("run sim: missing tensor 'input'")
            rank = raw.ndim
            meta = params.get("meta") or {}
            res = wb.run_step("sim", p, raw, meta, progress=progress, cancelled=cancelled, device=device)
            check()
            out = res.array
            if rank == 3:
                out = out[0, 0]
            return {"meta": _jsonable(res.meta), "info": _jsonable(res.info), "device": device}, {"output": out}

        if kind not in wb.step_kinds() and kind not in wb._KIND_ALIASES:  # noqa: SLF001 - same package family
            supported = ", ".join(list(_SPECIAL_KINDS) + [k for k in wb.step_kinds() if k not in _SPECIAL_KINDS])
            raise ValueError(f"unknown run kind '{kind}'; supported: {supported}")
        arr = tensors.get("input")
        if arr is None:
            raise ValueError(f"run {kind}: missing tensor 'input'")
        labels = tensors.get("labels")
        res = wb.run_step(kind, p, arr, params.get("meta") or None, labels, progress=progress,
                          cancelled=cancelled, device=device)
        check()
        out = {"output": res.array}
        if res.labels is not None:
            out["labels"] = np.ascontiguousarray(res.labels, dtype=np.uint32)
        if res.prob is not None:
            out["prob"] = res.prob
        return {"meta": _jsonable(res.meta), "info": _jsonable(res.info), "device": device}, out


# --- helpers -----------------------------------------------------------------------


def is_loopback(host: str) -> bool:
    """True for an address that only this machine can reach. "" and "0.0.0.0"
    are every interface, and a name that is not literally loopback is treated
    as public: the point is to be wrong on the safe side."""
    h = (host or "").strip()
    if not h:
        return False
    if h.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _peer(addr) -> str:
    try:
        return f"{addr[0]}:{addr[1]}"
    except (IndexError, TypeError):
        return str(addr)


def _username() -> str:
    for key in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    return "the user running it"


def _tensor(tensors: Dict[str, np.ndarray], name: str, ndim: int) -> np.ndarray:
    a = tensors.get(name)
    if a is None:
        raise ValueError(f"missing tensor '{name}'")
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > ndim and a.shape[0] == 1:
        a = a[0]
    if a.ndim != ndim:
        raise ValueError(f"tensor '{name}' must have {ndim} dimensions, got shape {a.shape}")
    return np.ascontiguousarray(a)


def _triple(v, default) -> tuple:
    if v is None:
        return tuple(default)
    if isinstance(v, str):
        parts = [x for x in v.replace("×", ",").replace("x", ",").split(",") if x.strip()]
        v = [int(float(x)) for x in parts]
    if isinstance(v, (int, float)):
        return (int(v),) * 3
    v = [int(x) for x in v]
    return tuple(v + list(default)[len(v):])[:3]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        # tolist() gives Python floats: scrubbed like any other (encode_frame
        # refuses NaN, and one inside a diagnostics table lost whole replies)
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def _message(e: BaseException) -> str:
    text = str(e).strip() or e.__class__.__name__
    return f"{e.__class__.__name__}: {text}" if not text.startswith(e.__class__.__name__) else text


def _module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def announce(server: WorkerServer, stream=None) -> None:
    """Print the one JSON line the launching application waits for."""
    stream = stream or sys.stdout
    stream.write(json.dumps({"port": server.port, "pid": os.getpid(), "host": server.host,
                             "device": server.resolved_device()}) + "\n")
    stream.flush()
