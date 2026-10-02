"""The fake worker's command line: ``python -m sirius_worker [the launcher's arguments]``.

What it does is chosen by $SIRIUS_FAKE_WORKER:

    missing (default)  what the real worker does without numpy: one
                       missing_packages line on stdout, one line on stderr, exit 3
    traceback          what a worker from before that line did: a
                       ModuleNotFoundError traceback, exit 1
    sleep              never prints its port (the launcher's cancel)
    linger             prints something else than its port, then stays (a
                       cancel while the launcher waits for it to leave);
                       "fake worker: lingering" on stderr says it got there
    exit               a start that fails for another reason: a message, exit 1

Every mode first logs its pid, its arguments, $PYTHONHOME and which of the
secret variables the launcher must not pass on it sees (by name, never the
value) to stderr, where the tests read them back from the launcher's log.
"""

import json
import os
import platform
import sys
import time


def main() -> int:
    mode = os.environ.get("SIRIUS_FAKE_WORKER", "missing")
    print(f"fake worker: pid {os.getpid()}", file=sys.stderr, flush=True)
    print("fake worker: argv " + json.dumps(sys.argv[1:]), file=sys.stderr, flush=True)
    print("fake worker: PYTHONHOME " + json.dumps(os.environ.get("PYTHONHOME")), file=sys.stderr, flush=True)
    seen = sorted(k for k in ("SIRIUS_HPC_TOKEN", "SIRIUS_LLM_API_KEY", "OPENAI_API_KEY", "HF_TOKEN") if k in os.environ)
    print("fake worker: secrets " + json.dumps(seen), file=sys.stderr, flush=True)
    if mode == "sleep":
        time.sleep(60)
        return 0
    if mode == "linger":
        print("fake worker: not a port", flush=True)
        # Long enough for the launcher to have read that line before the
        # test, which waits for the next one, cancels.
        time.sleep(0.5)
        print("fake worker: lingering", file=sys.stderr, flush=True)
        time.sleep(60)
        return 0
    if mode == "traceback":
        import sirius_test_absent_module  # noqa: F401

        return 0
    if mode == "exit":
        print("fake worker: failing on purpose", file=sys.stderr, flush=True)
        return 1
    report = {"error": "missing_packages", "missing": ["numpy"], "python": sys.executable, "version": platform.python_version()}
    print(json.dumps(report), flush=True)
    print(f"missing packages: numpy (not installed in {sys.executable})", file=sys.stderr, flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
