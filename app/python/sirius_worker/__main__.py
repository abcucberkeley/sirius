"""Command line of the SIRIUS compute worker.

    python -m sirius_worker [--host H] [--port P] [--token T] [--device auto|cuda|cpu]
                            [--allow-install] [--max-clients N] [--log-level L]
    python -m sirius_worker --check

Listens on host:port (port 0 picks a free one), prints one JSON line
``{"port": N, "pid": ..., "host": ..., "device": ...}`` to stdout once it is
ready -- the launching application reads exactly that -- and logs to stderr.
The token, when given, must be sent with the client's ``hello``, together
with the protocol version both ends have to agree on.

When a package the worker cannot start without (``REQUIRED``, numpy) is not
installed in this interpreter, it prints one JSON line instead,
``{"error": "missing_packages", "missing": ["numpy"], "python": ..., "version": ...}``,
logs which, and exits with code 3; the application then offers to set up its
own Python environment. Exit code 2 is a configuration mistake (the step
library not found, a refused bind).

``--check`` prints one JSON line about this interpreter -- its version, whether
it is a venv, pip and ensurepip, which required packages are missing and which
optional ones are installed -- and exits 0. It imports none of those packages,
so it works in an interpreter that lacks them; SIRIUS runs it to verify the
environment it set up.

Anyone holding the token can run code here; see ../SECURITY.md before binding
to anything but 127.0.0.1. Binding a non-loopback address without a token is
refused outright.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys

from . import OPTIONAL, REQUIRED, __version__

# The exit code for "a required package is missing", which the launching
# application recognises (together with the missing_packages line).
EXIT_MISSING_PACKAGES = 3


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _missing_required() -> list:
    """Import names of REQUIRED that this interpreter cannot import.

    find_spec locates a top-level module without importing it, so this costs
    nothing and cannot fail on a broken package. Tests replace it."""
    import importlib.util

    missing = []
    for name in REQUIRED:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _distribution_version(name: str):
    """The installed version of distribution `name`, or None."""
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:  # not installed, or metadata that cannot be read
        return None


def _check_report() -> dict:
    """What --check prints: this interpreter, as the worker would run in it."""
    import importlib.util
    import sysconfig

    def has(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    optional = {dist: _distribution_version(dist) for dist in OPTIONAL.values()}
    packages = {}
    for dist in [*REQUIRED.values(), *OPTIONAL.values(), "pip"]:
        version = _distribution_version(dist)
        if version is not None:
            packages[dist] = version
    stdlib = sysconfig.get_path("stdlib") or ""
    return {
        "python": sys.executable,
        "executable": sys.executable,
        "base_executable": getattr(sys, "_base_executable", None) or sys.executable,
        "version": _python_version(),
        "venv": sys.prefix != sys.base_prefix,
        "externally_managed": os.path.isfile(os.path.join(stdlib, "EXTERNALLY-MANAGED")),
        "pip": has("pip"),
        "ensurepip": has("ensurepip"),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "bits": 64 if sys.maxsize > 2**32 else 32,
        "missing": _missing_required(),
        "optional": optional,
        "packages": packages,
        "worker": __version__,
    }


def _device(text: str) -> str:
    """--device: auto, cpu, cuda or cuda:N. The application names the GPU it was
    told to use ("cuda:1"); with only the three bare words accepted, a worker
    started on the CUDA backend exited at once with a usage error."""
    value = text.strip().lower()
    if value in ("auto", "cpu", "cuda"):
        return value
    kind, _, index = value.partition(":")
    if kind == "cuda" and index.isdigit():
        return value
    raise argparse.ArgumentTypeError(f"{text!r} is not auto, cpu, cuda or cuda:N")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sirius_worker", description="SIRIUS compute worker")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (0.0.0.0 for a cluster node; then a token is required)")
    parser.add_argument("--port", type=int, default=0, help="TCP port; 0 picks a free port")
    parser.add_argument("--token", default=os.environ.get("SIRIUS_TOKEN", ""),
                        help="shared secret the client must present (default: $SIRIUS_TOKEN)")
    parser.add_argument("--device", default="auto", type=_device,
                        help="where models run: auto (cuda when torch sees a GPU), cpu, cuda, or cuda:N for one GPU")
    # Package installation (the `install` method: pip / conda in this
    # interpreter) is a privileged operation, so it is opt-in. The desktop
    # application passes this for the worker it starts on the user's own
    # machine; the cluster job script (slurm/sirius_worker.sbatch) does not.
    parser.add_argument("--allow-install", action="store_true",
                        help="let clients install model packages with pip / conda in this interpreter")
    parser.add_argument("--log-level", default="INFO", help="stderr log level")
    # The desktop application passes this: it holds the worker's stdin open,
    # so end-of-file there means the application is gone (crashed, killed)
    # and the worker must not live on holding the GPU and a valid token.
    # Not for a terminal or a batch job, where stdin is closed from the start.
    parser.add_argument("--exit-with-parent", action="store_true",
                        help="stop when stdin reaches end-of-file (the launching process went away)")
    # The cluster job passes this: the application keeps a connection for its
    # status and one for the dataset on screen besides each run's.
    parser.add_argument("--max-clients", type=int, default=int(os.environ.get("SIRIUS_MAX_CLIENTS", "1") or 1),
                        help="connections served at once (default 1, or $SIRIUS_MAX_CLIENTS); jobs still run one at a time")
    parser.add_argument("--check", action="store_true",
                        help="print what this interpreter has for the worker as one JSON line and exit")
    args = parser.parse_args(argv)

    if args.check:
        print(json.dumps(_check_report()), flush=True)
        return 0

    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Before anything imports numpy: without it the imports below end in a
    # traceback, which the application can only show as it is. This line says
    # what is missing and where, so that it can offer the fix instead.
    missing = _missing_required()
    if missing:
        print(json.dumps({"error": "missing_packages", "missing": missing, "python": sys.executable,
                          "version": _python_version()}), flush=True)
        logging.getLogger("sirius_worker").error("missing packages: %s (not installed in %s)", ", ".join(missing), sys.executable)
        return EXIT_MISSING_PACKAGES

    from . import models
    from .server import WorkerServer, announce
    from .steps import workbench

    models.ALLOW_INSTALL = bool(args.allow_install)
    if models.ALLOW_INSTALL:
        logging.getLogger("sirius_worker").warning(
            "--allow-install: a client presenting the token may install packages into %s", sys.executable)

    try:
        wb = workbench()
        # With forward slashes, as every other path SIRIUS reports.
        source = str(getattr(wb, "__source_file__", wb.__file__)).replace("\\", "/")
        logging.getLogger("sirius_worker").info("step library: %s", source)
    except ImportError as e:
        logging.getLogger("sirius_worker").error("%s", e)
        return 2

    server = WorkerServer(args.host, args.port, args.token, args.device, max(1, args.max_clients))
    try:
        server.bind()
    except (OSError, ValueError) as e:
        # A refused bind is a configuration mistake, not a crash: say what to
        # change and exit, without the traceback.
        logging.getLogger("sirius_worker").error("%s", e)
        return 2
    announce(server)

    def on_signal(signum, frame):  # noqa: ARG001
        server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass
    if args.exit_with_parent:
        watch_parent(server)
    server.serve_forever()
    return 0


def _wait_for_stdin_eof_windows() -> None:
    """Return once the pipe on stdin is closed at its other end, on Windows.

    A thread blocked in a read of the pipe is not an option there: while that
    synchronous read is pending, anything else that touches the handle waits
    for it, and loading an extension module does (the C runtime of each DLL
    looks at the standard handles as it starts). The first `import numpy` of a
    step then hangs for good. Peeking at the pipe never blocks: it fails with
    ERROR_BROKEN_PIPE once the parent's end is closed."""
    import ctypes
    import msvcrt
    import threading
    import time
    from ctypes import wintypes

    error_broken_pipe = 109
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    peek = kernel32.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    peek.restype = wintypes.BOOL
    fd = sys.stdin.fileno()
    handle = msvcrt.get_osfhandle(fd)
    available = wintypes.DWORD(0)
    while True:
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error != error_broken_pipe:
                # Not a pipe (a console, a file): there is no end to wait for,
                # and the worker runs until it is stopped some other way.
                logging.getLogger("sirius_worker").warning(
                    "--exit-with-parent: stdin is not a pipe (error %d); not watching it", error)
                threading.Event().wait()
            return
        if available.value:
            # Whatever the parent writes is drained, so the pipe never fills;
            # the bytes are there, so this read returns at once.
            os.read(fd, available.value)
        time.sleep(0.25)


def watch_parent(server) -> None:
    """Stop `server` once stdin reaches end-of-file, from a daemon thread.

    The parent holds the other end of the pipe for as long as it lives; one
    that exits or crashes closes it. A job in flight is given a moment to
    notice the stop flag, then the process leaves regardless: nobody is
    waiting for its answer any more."""
    import threading
    import time

    def run() -> None:
        try:
            if sys.platform == "win32":
                _wait_for_stdin_eof_windows()
            else:
                stream = getattr(sys.stdin, "buffer", sys.stdin)
                while stream.read(1):
                    pass
        except (OSError, ValueError):
            pass
        logging.getLogger("sirius_worker").info("the launching process went away; stopping")
        server.stop()
        time.sleep(2.0)
        os._exit(0)

    threading.Thread(target=run, name="sirius-parent-watch", daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
