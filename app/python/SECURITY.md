# Security model of the SIRIUS compute worker

`sirius_worker` is a TCP service that runs whatever the connected client asks
it to run. This file says exactly what "whatever" covers, so that nobody has to
infer it from the protocol.

## The one sentence to remember

**Holding the worker's token is equivalent to having a shell on the machine the
worker runs on, as the user who started it.**

Everything below is a consequence of that, not a separate risk.

## What a client can make the worker do

| Method | What it reaches |
| --- | --- |
| `reload_plugins` with `dirs` | Every `*.py` in the directories the *client* names is imported and executed. The client chooses the path; the worker imports the file. |
| `install` | Runs `pip install` / `conda install` in the worker's interpreter — arbitrary package code, from the index. Off by default; see below. |
| `run` with `kind: "flatfield"` | Reads the TIFF at the `flat` / `dark` path the client gives. |
| `run` with `kind: "sim"` | Reads the measured OTF file at the `otf` path, and a parameter file at `params_file`. |
| `run` / `model_info` / `model_prepare` with a `file:` model spec | Loads a TorchScript / ONNX file from any path. `torch.jit.load` on an untrusted file is itself code execution. |
| `hub_download`, `model_prepare` | Fetches from Hugging Face into the worker's cache, with the client's HF token when it sends one. |
| `shutdown` | Stops the worker, and with it every job running on it. |

None of these paths is confined to a sandbox, a chroot or a directory
allow-list. Model files, plugin directories and flat-field images are read (and
plugins executed) with the full rights of the account that started the worker.

## Authentication

`--token T` (or `$SIRIUS_TOKEN`) is a shared secret the client must present in
its first `hello`, where it is compared with `hmac.compare_digest` (constant
time, so a wrong token leaks nothing through how long the answer takes).
Nothing else is checked: no user identity, no rate limit, no authorisation
levels. Every client that knows the token can do everything in the table
above.

`--token ""` — the default when neither the flag nor `$SIRIUS_TOKEN` is set —
**disables authentication entirely**: the worker serves any connection that
reaches it. That is only tolerable bound to `127.0.0.1` on a machine you are
the only user of, so:

- **binding a non-loopback address with an empty token is refused at
  startup**, with a message naming the fix (`--host 0.0.0.0`, `--host` of a
  routable interface, and the "every interface" `--host ""` all count). The
  worker exits 2 without opening a socket.
- binding a loopback address with an empty token still works — that is the
  single-user desktop case, and the application starts its local worker that
  way — but logs a warning saying that every client that can connect is
  served.
- `app/python/slurm/sirius_worker.sbatch` refuses to submit a job without
  `SIRIUS_TOKEN` for the same reason, one layer earlier.

Always set a token. Generate it, don't invent it:

```sh
SIRIUS_TOKEN=$(openssl rand -hex 16)
```

Nothing at all is served before a successful `hello`, with or without a token:
the version handshake below happens first either way.

## Frames from an unauthenticated peer

Both lengths in a frame header, and every tensor descriptor in it, are numbers
the peer chose. Both ends check them before they size an allocation or index a
buffer, and the same limits apply on both sides
(`sirius_worker/protocol.py`, `app/core/rpc.cpp`):

| limit | value | what it stops |
| --- | --- | --- |
| `MAX_HEADER` / `kMaxHeaderBytes` | 64 MiB | a header length from a corrupt or hostile stream |
| `MAX_PAYLOAD` / `kMaxPayloadBytes` | 32 GiB | a payload length that would otherwise be believed, waited for, or (in C++) wrapped into a small total |
| `MAX_PREAUTH_FRAME` | 16 KiB | anything an unauthenticated peer sends: a `hello` is a few hundred bytes |

The pre-authentication cap is enforced by the read path — each length is
checked against it *before* the bytes it announces are read — so an anonymous
peer cannot make the worker allocate a buffer or block on a long read by
announcing a large frame. Tensor descriptors are checked the same way on both
sides: `offset` and `nbytes` are compared without forming a sum that could
wrap, and the product of the shape is bounded as it is computed, before it
sizes anything.

## Protocol version

`hello` carries a `protocol_version` in its params and returns one in its
result: `PROTOCOL_VERSION` in `sirius_worker/protocol.py`, `kProtocolVersion`
in `app/core/rpc.hpp`, currently **1**. The rule is *the same version on both
ends*; a peer that sends no field at all predates the handshake and counts as
version 0. Either side refuses a mismatch immediately, naming both numbers and
which end to update — the application raises it out of `RemoteWorker`'s
constructor, so it reaches the user as the reason the worker would not connect
rather than as a strange failure in the middle of a run.

Bump both constants together whenever the framing or the method set changes in
a way an older peer cannot understand.

## There is no transport security

The protocol is length-prefixed JSON and raw tensors over a plain TCP socket.
No TLS, no certificate, no integrity check. The token is sent in the clear in
the first message, and so is every image the worker returns.

The supported deployment is therefore:

- **bind to `127.0.0.1`** (the default `--host`), and
- reach a remote worker **through an SSH tunnel**:

```sh
ssh -N -L 7645:<node>:7645 <login-node>
```

The application then talks to `localhost:7645`, and SSH provides the
encryption and the authentication of the host. `--host 0.0.0.0` puts an
unencrypted, single-secret service on the network; the SLURM script uses it
because a compute node is only reachable from inside the cluster, and even
there the token is what stands between the worker and any other user on the
login node.

## The privileged methods: `install` and `shutdown`

`install` changes the worker's software and `shutdown` ends everyone's work on
it. There is only one privilege level — the token — so neither can be
restricted to a subset of clients, and both are logged at WARNING with the
peer's address before they act:

```
sirius_worker WARNING privileged request: install 'cellpose', from 10.0.0.7:51544 (allow_install=False)
sirius_worker WARNING privileged request: shutdown, from 127.0.0.1:51544
```

`shutdown` is deliberately **not** restricted to loopback peers. The supported
remote deployment reaches the worker through `ssh -L`, so the worker sees the
connection coming from the SSH host, not from `127.0.0.1`; a loopback rule
would leave a cluster worker running until its Slurm job's wall clock expired,
while stopping nobody — the same client could simply `install` instead. The
log line is what makes the shutdown attributable.

`install` is additionally gated:

## `--allow-install`

The `install` method runs `pip` or `conda` inside the worker's environment. A
package install executes the package's own code, so this is the one method
that turns "can talk to the worker" into "can change the worker's software".
It is opt-in:

- `python -m sirius_worker --allow-install …` — the desktop application passes
  this for the worker it starts **on the user's own machine**, where the client
  and the worker are the same person, and installing Cellpose or micro-SAM from
  the model hub dialog is the point.
- Without the flag, `install` is refused with the command the user could run
  themselves. `app/python/slurm/sirius_worker.sbatch` deliberately does not
  pass it: on a shared cluster node the worker's environment is a module or a
  conda prefix that other jobs use, and letting a client mutate it is both a
  security problem and an operational one. Install the model packages when you
  prepare the environment, before submitting the job.
- `install` with `dry_run` is always allowed. It adds `--dry-run` to the
  command, so nothing is written; the model hub dialog uses it to show what an
  install would do.

## Tokens the worker is given

The Hugging Face access token a client sends with `hub_*`, `model_prepare`
and a `run` that fetches a gated model is passed to `huggingface_hub` as a
call argument for that request only (held per thread, so a job in flight and
the connection serving the next request cannot swap each other's). It is
deliberately **not** written to `os.environ`: `HF_TOKEN` there would outlive
the request and be inherited by every subprocess the worker starts, `pip` and
`conda` included -- and the desktop launcher does not put it there either.

On the application side, the worker token, the Hugging Face token and the
assistant's API key are stored through `app/qt/secret_store.hpp` (DPAPI on
Windows, a `0600` file elsewhere) rather than as plain text in `QSettings`.

## Reporting

Security issues in SIRIUS itself: open an issue, or contact the maintainer
listed in `pyproject.toml`.
