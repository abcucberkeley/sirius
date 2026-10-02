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
| `reload_plugins` with `dirs` | Every `*.py` in the directories the *client* names is imported and executed. Only on a worker bound to loopback: one bound to any other address refuses `dirs` and loads plugins only from the folders it was started with (`$SIRIUS_PLUGIN_DIRS`, `~/.sirius/plugins`, the checkout's `app/plugins`). |
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

The worker's token is a shared secret. It reaches the worker, in this order
of preference, from:

- `--token-file F` or `$SIRIUS_TOKEN_FILE`: a file only its owner can read
  (on POSIX the worker refuses one that is group- or world-accessible, a
  symlink or another user's), which the worker reads **and deletes**. This is
  how the application hands the token to a cluster job (below).
- `$SIRIUS_TOKEN`: how the desktop application starts its local worker (the
  environment of a process is readable only by its owner).
- `--token T`: still accepted, with a warning, because a command line is
  visible to every user of the machine (`ps`, `/proc`).

Both variables are removed from the worker's environment once read, so no
step, plugin or `pip` it starts inherits them.

**The token never crosses the wire** (protocol version 2). The handshake is a
challenge-response in which the worker proves itself first:

```
client -> hello {protocol_version, client_nonce}            (16 random bytes, hex)
worker -> {protocol_version, server_nonce, server_proof}
client    checks server_proof; on a mismatch it closes and sends nothing more
client -> auth  {client_proof}
worker -> the capabilities
```

with `proof = hex HMAC-SHA256(token, "sirius-worker-auth/2|" + role + "|" +
client_nonce + "|" + server_nonce)`, role `worker` or `client`
(`protocol.handshake_proof`, `rpc::handshakeProof`; the C++ side's SHA-256 is
`app/core/sha256.cpp`, tested against the FIPS and RFC 4231 vectors). Proofs
are compared in constant time. A process that took the worker's port (or
answers in its place) learns a nonce and nothing it could replay, and the
application never sends it a request; requests no longer carry the token
either. Nothing else is checked: no user identity, no authorisation levels.
Every client that completes the handshake can do everything in the table
above.

What the handshake does not give is a protected channel: after it, frames are
neither encrypted nor authenticated, so a party that can *relay and modify*
the TCP stream between the two (rather than merely listen or impersonate one
end) could inject requests into an authenticated connection. That is what the
SSH tunnel below is for.

An empty token — the default when none of the above is given —
**disables authentication entirely**: the proofs are then computable by
anyone, and the worker serves any connection that reaches it. That is only
tolerable bound to `127.0.0.1` on a machine you are the only user of, so:

- **binding a non-loopback address with an empty token is refused at
  startup**, with a message naming the fix (`--host 0.0.0.0`, `--host` of a
  routable interface, and the "every interface" `--host ""` all count). The
  worker exits 2 without opening a socket.
- binding a loopback address with an empty token still works but logs a
  warning saying that every client that can connect is served.
- `app/python/slurm/sirius_worker.sbatch` refuses to start a job without
  `SIRIUS_TOKEN_FILE` or `SIRIUS_TOKEN` for the same reason, one layer
  earlier.

Always set a token. Generate it, don't invent it (`openssl rand -hex 16`).

### The cluster job (Process ▸ Connect to cluster)

The application never puts the token in the job's environment, where Slurm's
accounting may keep it (`AccountingStoreFlags=job_env`), nor on any command
line. Over the SSH command channel it runs `umask 077`, creates
`~/.sirius/run` mode `0700`, writes the token with the shell's `printf`
builtin to a `mktemp` file there (`0600`), and submits the job with only that
file's name in `$SIRIUS_TOKEN_FILE`; the worker deletes the file as it starts
(token files of jobs that never started are removed after a day). The job's
log goes to `~/.sirius/run/sirius-worker-<job>.log`, not the checkout, and the
worker listens on `--port 0`: it takes a free port and announces it in that
log, which only you can read, so no other user of the node can claim a known
port first. The application reads the port from there.

## Connections before and after the handshake

Every connection is served on a thread of its own, so a peer that never
speaks cannot hold the listener, also when the worker serves one client at a
time. A connection that has not completed the handshake:

- must complete it within **5 s** (`WorkerServer.PREAUTH_TIMEOUT`), the frame
  in flight included — a peer dripping a byte at a time is cut off too;
- is one of at most **8** such connections at once (`MAX_PREAUTH`); one more
  is closed as it is accepted;
- does not take a client slot: `--max-clients` counts authenticated
  connections only, and a single-client worker answers the next client's
  `auth` once the client before it has gone.

An authenticated connection that sends nothing and runs nothing for an hour
(`--idle-timeout`, 0 for never) is closed; the application reconnects when it
needs to.

## Frames from an unauthenticated peer

Both lengths in a frame header, and every tensor descriptor in it, are numbers
the peer chose. Both ends check them before they size an allocation or index a
buffer (`sirius_worker/protocol.py`, `app/core/rpc.cpp`):

| limit | value | what it stops |
| --- | --- | --- |
| `MAX_HEADER` / `kMaxHeaderBytes` | 64 MiB | a header length from a corrupt or hostile stream |
| `MAX_PAYLOAD` (worker) | 32 GiB | a payload length that would otherwise be believed and waited for |
| `rpc::maxPayloadBytes()` (application) | 8 GiB, `$SIRIUS_RPC_MAX_PAYLOAD_GIB` | a reply larger than anything the application asks for |
| `MAX_TENSORS` / `kMaxTensors` | 64 | a frame describing more tensors than any request carries |
| `MAX_PREAUTH_FRAME` | 16 KiB | anything an unauthenticated peer sends: `hello` and `auth` are a few hundred bytes |

The pre-authentication cap is enforced by the read path — each length is
checked against it *before* the bytes it announces are read — so an anonymous
peer cannot make the worker allocate a buffer or block on a long read by
announcing a large frame. Tensor descriptors are checked the same way on both
sides: `offset` and `nbytes` are compared without forming a sum that could
wrap, the product of the shape is bounded as it is computed, before it sizes
anything, and the descriptors must follow one another in increasing offset
order without overlapping (and name each tensor once), so together they never
claim more than the payload and no byte is decoded twice.

A dataset array the worker sends compressed (`dataset_read` / `dataset_view`)
carries its shape: the application checks that shape's product before it
allocates, refuses a description larger than the bytes could inflate to, and
inflates into exactly that many bytes — a stream that would inflate further
(a decompression bomb) is an error, not followed. `datasets.decode` holds the
same rule on the Python side. A step input named by reference (`input_ref`)
may name each channel and time point of the dataset once, and only those that
exist.

## Protocol version

`hello` carries a `protocol_version` in its params and returns one in its
result: `PROTOCOL_VERSION` in `sirius_worker/protocol.py`, `kProtocolVersion`
in `app/core/rpc.hpp`, currently **2** (version 1 sent the token in the clear
in `hello` and in every request; there is no fallback to it — an application
meeting a version 1 worker says to update the worker, and a version 1
application is told by the worker to update itself). The rule is *the same version on both
ends*; a peer that sends no field at all predates the handshake and counts as
version 0. Either side refuses a mismatch immediately, naming both numbers and
which end to update — the application raises it out of `RemoteWorker`'s
constructor, so it reaches the user as the reason the worker would not connect
rather than as a strange failure in the middle of a run.

Bump both constants together whenever the framing or the method set changes in
a way an older peer cannot understand.

## There is no transport security

The protocol is length-prefixed JSON and raw tensors over a plain TCP socket.
No TLS, no certificate, no integrity check after the handshake. The token
itself is never sent, but every image the worker returns is, in the clear.

The supported deployment is therefore:

- **bind to `127.0.0.1`** (the default `--host`), and
- reach a remote worker **through an SSH tunnel**:

```sh
ssh -N -L 7645:<node>:7645 <login-node>
```

The application then talks to `localhost:7645`, and SSH provides the
encryption, the integrity of the stream and the authentication of the host.
(The application's own cluster connection does the same through its SSH
session's SOCKS proxy.) `--host 0.0.0.0` puts an
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
assistant's API key are stored through `app/imgui/secret_store.hpp` (DPAPI
blobs in the application's settings file on Windows, `~/.sirius/secrets.json`
created `0600` elsewhere) rather than as plain text in the settings.

## The application's SSH session (Process ▸ Connect to cluster)

The application logs in with the system's OpenSSH client
(`app/core/remote_host.cpp`):

- ssh runs with `-x -a -o ForwardAgent=no -o ForwardX11=no
  -o PermitLocalCommand=no`, whatever `~/.ssh/config` says: the cluster gets
  neither this machine's display nor its SSH agent, and no `LocalCommand`
  runs here. Forwardings are not cleared, because the session's own `-D` is
  one.
- Its SOCKS proxy (`-D 127.0.0.1:<free port>`) listens on loopback only and
  lives exactly as long as the session. While it lives, **any process on this
  machine** can open connections into the cluster through it, as you: on a
  shared desktop, disconnect when you are done. (Forwarding only the worker's
  port would need the port before the job runs, or a second login.)
- ssh runs in a job object (Windows) or a process group of its own (POSIX)
  and is ended with the session; if the application dies, the job closes with
  it on Windows, and elsewhere ssh's stdin closes, which ends the remote shell
  and ssh with it -- no authenticated connection is left behind.
- Passwords and one-time codes reach ssh through its askpass helper and a
  loopback relay guarded by a per-login secret. The relay serves each
  connection on its own thread, drops one that has not sent its request
  within a second, takes at most 8 at once, uses `SO_EXCLUSIVEADDRUSE` on
  Windows, and stops listening as soon as the login is over. An answer is
  shown in clear only for OpenSSH's own host key question
  (`SSH_ASKPASS_PROMPT=confirm`, or its exact "The authenticity of host …
  Are you sure you want to continue connecting (yes/no" wording), never
  because a server's prompt says "(yes/no)". The password field keeps no
  undo history, and the field's text, Dear ImGui's copies of it, the relay's
  JSON and the helper's buffers are overwritten once the answer is handed on.

## What the worker's children inherit

The local worker the application starts, and the installers it runs to set up
its Python environment, do not inherit the application's other secrets:
`SIRIUS_HPC_TOKEN`, `SIRIUS_LLM_API_KEY` and the model providers' API keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` and the like,
`pyenv::secretEnvironmentNames`) are removed from their environment. The local
worker keeps `HF_TOKEN`, which it downloads models with; the installers do not.
A custom package index reaches pip or uv as `PIP_INDEX_URL` / `UV_INDEX_URL`,
not on the command line, and is shown with any credentials and query values
masked.

## Reporting

Security issues in SIRIUS itself: open an issue, or contact the maintainer
listed in `pyproject.toml`.
