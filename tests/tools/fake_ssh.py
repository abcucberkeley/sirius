#!/usr/bin/env python3
"""A stand-in for OpenSSH's `ssh` in the cluster tests (tests/test_app_cluster.cpp).

It never opens a network connection to anything but 127.0.0.1. Started the
way core/remote_host.cpp starts ssh:

    fake_ssh.py -T -o BatchMode=no -o NumberOfPasswordPrompts=1 ... [-D 127.0.0.1:PORT] -- HOST "/bin/bash -l -s"

it does what OpenSSH would, as far as the application can tell:

  * the login: for each prompt in $FAKE_SSH_PROMPTS (a JSON list, e.g.
    ["Password: ", "Verification code: "]) it runs $SSH_ASKPASS with the
    prompt "(USER@HOST) <prompt>" as its one argument -- only when
    SSH_ASKPASS_REQUIRE=force, as the application sets -- and reads the answer
    from its stdout. An answer other than the one $FAKE_SSH_ANSWERS lists ends
    the login: "USER@HOST: Permission denied (keyboard-interactive)." on
    stderr, exit 255. A helper that fails counts as an empty answer, which is
    what OpenSSH would send; the tests check that the application never lets
    it come to that (it stops this process first).
  * -D: a SOCKS5 proxy on that port (no authentication, CONNECT only), which
    connects every host name it is given to 127.0.0.1 -- the "compute node"
    is this machine. A target nothing listens on closes the connection, as
    OpenSSH does.
  * the remote command: `bash -s` (never a login shell: the user's own
    profile is not read) with this process's stdin and stdout, in
    $FAKE_SSH_HOME (HOME as well), with $FAKE_SSH_PATH in front of PATH
    (the fake Slurm tools of tests/tools/fake_slurm).

$FAKE_SSH_LOG, when set, receives one line per event (argv, asking,
answered, response, socks, exit) for the tests to read. $FAKE_SSH_BASH names
the bash to run. With $FAKE_SLURM_DIR and $FAKE_SLURM_KILL_ON_EXIT=1 the fake
jobs' workers are stopped when this process ends.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading


def log(event: str) -> None:
    path = os.environ.get("FAKE_SSH_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(event.replace("\n", " ") + "\n")


def parse(argv):
    options, socks, host, command = [], None, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "-D", "-p", "-l", "-i", "-J", "-F"):
            if i + 1 >= len(argv):
                break
            if a == "-o":
                options.append(argv[i + 1])
            elif a == "-D":
                socks = argv[i + 1]
            i += 2
            continue
        if a == "--":
            host = argv[i + 1] if i + 1 < len(argv) else None
            command = argv[i + 2:]
            break
        if a.startswith("-"):
            i += 1
            continue
        host, command = a, argv[i + 1:]
        break
    return options, socks, host, command


def askpass(prompt: str):
    program = os.environ.get("SSH_ASKPASS", "")
    if not program or os.environ.get("SSH_ASKPASS_REQUIRE", "") != "force":
        return None
    log("asking " + prompt)
    try:
        r = subprocess.run([program, prompt], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"answered error {e}")
        return None
    log(f"answered rc={r.returncode}")
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "replace").rstrip("\r\n")


class _Socks(socketserver.BaseRequestHandler):
    def _exactly(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.request.recv(n - len(data))
            if not chunk:
                raise ConnectionError
            data += chunk
        return data

    def handle(self):
        try:
            ver, n = self._exactly(2)
            self._exactly(n)
            if ver != 5:
                return
            self.request.sendall(b"\x05\x00")
            ver, cmd, _, atyp = self._exactly(4)
            if atyp == 3:
                host = self._exactly(self._exactly(1)[0]).decode("ascii", "replace")
            elif atyp == 1:
                host = socket.inet_ntoa(self._exactly(4))
            else:
                return
            (port,) = struct.unpack(">H", self._exactly(2))
            log(f"socks {host}:{port}")
            try:
                upstream = socket.create_connection(("127.0.0.1", port), timeout=5)
            except OSError:
                sys.stderr.write("channel 3: open failed: connect failed: Connection refused\n")
                sys.stderr.flush()
                return   # OpenSSH closes without a reply
            upstream.settimeout(None)
            self.request.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")

            def pump(a, b):
                try:
                    while True:
                        d = a.recv(65536)
                        if not d:
                            break
                        b.sendall(d)
                except OSError:
                    pass
                finally:
                    for s in (a, b):
                        try:
                            s.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass

            t = threading.Thread(target=pump, args=(upstream, self.request), daemon=True)
            t.start()
            pump(self.request, upstream)
            t.join(timeout=5)
            upstream.close()
        except (ConnectionError, OSError, ValueError):
            pass


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


def find_bash() -> str:
    given = os.environ.get("FAKE_SSH_BASH")
    if given:
        return given
    if os.name == "nt":
        for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if os.path.isfile(p):
                return p
    return shutil.which("bash") or "/bin/bash"


def stop_fake_jobs() -> None:
    d = os.environ.get("FAKE_SLURM_DIR")
    if not d or os.environ.get("FAKE_SLURM_KILL_ON_EXIT") != "1":
        return
    for f in glob.glob(os.path.join(d, "*.winpid")) + glob.glob(os.path.join(d, "*.ospid")):
        try:
            pid = int(open(f, encoding="utf-8").read().strip())
        except (OSError, ValueError):
            continue
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def main(argv) -> int:
    log("argv " + json.dumps(argv))
    options, socks, host, _command = parse(argv)
    if not host:
        sys.stderr.write("usage: ssh [options] destination [command]\n")
        return 255
    user = os.environ.get("FAKE_SSH_USER", "tester")
    name = host.split("@")[-1]
    prompts = json.loads(os.environ.get("FAKE_SSH_PROMPTS", "[]"))
    answers = json.loads(os.environ.get("FAKE_SSH_ANSWERS", "[]"))
    for i, p in enumerate(prompts):
        got = askpass(f"({user}@{name}) {p}")
        if got is None:
            got = ""   # what OpenSSH sends for a prompt its helper gave up on
        ok = i < len(answers) and got == answers[i]
        log("response " + ("ok" if ok else ("empty" if got == "" else "wrong")))
        if not ok:
            sys.stderr.write(f"{user}@{name}: Permission denied (keyboard-interactive).\n")
            sys.stderr.flush()
            log("exit 255")
            return 255
    server = None
    if socks:
        port = int(socks.rsplit(":", 1)[-1])
        try:
            server = _Server(("127.0.0.1", port), _Socks)
        except OSError:
            sys.stderr.write(f"bind [127.0.0.1]:{port}: Address already in use\nCould not request local forwarding.\n")
            sys.stderr.flush()
            if "ExitOnForwardFailure=yes" in options:
                log("exit 255")
                return 255
        if server is not None:
            threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ)
    home = os.environ.get("FAKE_SSH_HOME") or os.getcwd()
    env["HOME"] = home
    extra = os.environ.get("FAKE_SSH_PATH")
    if extra:
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    for k in ("SSH_ASKPASS", "SSH_ASKPASS_REQUIRE", "SIRIUS_ASKPASS_PORT", "SIRIUS_ASKPASS_SECRET"):
        env.pop(k, None)
    try:
        rc = subprocess.call([find_bash(), "--noprofile", "--norc", "-s"], cwd=home, env=env)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        stop_fake_jobs()
    log(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
