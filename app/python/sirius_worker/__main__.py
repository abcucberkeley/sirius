"""Command line of the SIRIUS compute worker.

    python -m sirius_worker [--host H] [--port P] [--token T] [--device auto|cuda|cpu]
                            [--allow-install] [--log-level L]

Listens on host:port (port 0 picks a free one), prints one JSON line
``{"port": N, "pid": ..., "host": ..., "device": ...}`` to stdout once it is
ready -- the launching application reads exactly that -- and logs to stderr.
The token, when given, must be sent with the client's ``hello``, together
with the protocol version both ends have to agree on.

Anyone holding the token can run code here; see ../SECURITY.md before binding
to anything but 127.0.0.1. Binding a non-loopback address without a token is
refused outright.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sirius_worker", description="SIRIUS compute worker")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (0.0.0.0 for a cluster node; then a token is required)")
    parser.add_argument("--port", type=int, default=0, help="TCP port; 0 picks a free port")
    parser.add_argument("--token", default=os.environ.get("SIRIUS_TOKEN", ""),
                        help="shared secret the client must present (default: $SIRIUS_TOKEN)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="where models run; auto = cuda when torch sees a GPU")
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
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from . import models
    from .server import WorkerServer, announce
    from .steps import workbench

    models.ALLOW_INSTALL = bool(args.allow_install)
    if models.ALLOW_INSTALL:
        logging.getLogger("sirius_worker").warning(
            "--allow-install: a client presenting the token may install packages into %s", sys.executable)

    try:
        wb = workbench()
        logging.getLogger("sirius_worker").info("step library: %s", getattr(wb, "__source_file__", wb.__file__))
    except ImportError as e:
        logging.getLogger("sirius_worker").error("%s", e)
        return 2

    server = WorkerServer(args.host, args.port, args.token, args.device)
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


def watch_parent(server) -> None:
    """Stop `server` once stdin reaches end-of-file, from a daemon thread.

    The read blocks for as long as the parent lives; a parent that exits or
    crashes closes its end of the pipe. A job in flight is given a moment to
    notice the stop flag, then the process leaves regardless: nobody is
    waiting for its answer any more."""
    import threading
    import time

    def run() -> None:
        try:
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
