# Running the compute worker under Slurm (the HPC backend)

The application's **HPC** backend sends steps to a running `sirius_worker`.
On a cluster the worker runs once per session as a Slurm job, and the
application reaches it through SSH.

## From the application (the usual way)

*Process ▸ Connect to cluster…* in `sirius-app` does all of the below in one
window, with one SSH login:

- **Profile**: the SSH host (an alias of your `~/.ssh/config`, e.g. `fiona`,
  ProxyJump and all), the SIRIUS checkout on the cluster (default
  `~/dev/sirius`), the Python environment to activate (default
  `~/venvs/sirius`), and the job's partition, account, QoS, time, GPUs, CPUs
  and memory (fiona's test GPUs by default: `abc_a100`, `velatkilic`,
  `abc_debug`, one hour). It is remembered.
- **Connect** starts the system's OpenSSH once
  (`ssh -T -o NumberOfPasswordPrompts=1 -o BatchMode=no -D 127.0.0.1:<port> <host> bash -l -s`):
  a password or one-time code prompt is shown in the application and handed
  to ssh only (never stored); the same connection then serves as the command
  channel (checks, `sbatch --parsable`, `squeue`, `scancel`, folder
  listings) and as a SOCKS proxy through which the application reaches the
  worker on the compute node -- no `ssh -L`, no node name to copy.
- The checklist shows each step: login, checks (sbatch, the checkout, the
  venv and numpy in it, with the command that fixes what is missing),
  submit, the queue (state, reason, time waited), the node, and the
  worker's hello (version, device, steps).
- The status bar keeps saying whether it is connected; a job that ends
  (TIMEOUT, CANCELLED) or a connection that drops is reported with the
  reason. *Disconnect…* (and quitting) asks whether to `scancel` the job.

Install once, on the cluster: the checkout at the profile's path (the same
version as the application), and in the venv
`pip install -r app/python/requirements.txt tifffile` (tifffile lets the
worker read TIFF datasets for *File ▸ Open from cluster…*; torch for
segmentation models). The job runs this directory's `sirius_worker.sbatch`
with the profile's options on the `sbatch` command line and `SIRIUS_VENV`,
`SIRIUS_PORT=0` and `SIRIUS_MAX_CLIENTS` in its environment. The token is not
there (Slurm's accounting may store a job's environment): the application
writes it over the SSH session to a private file in `~/.sirius/run` (a `0700`
directory, the file `0600`) and the job gets only the file's name in
`SIRIUS_TOKEN_FILE`; the worker deletes the file as it starts. The job's log
is `~/.sirius/run/sirius-worker-<jobid>.log`, and the worker takes a free port,
which it announces there (`../SECURITY.md`).

The manual way follows: for `sirius-cli`, or a cluster the dialog does not fit.

## 1. Start the worker on a node

```
umask 077; mkdir -p ~/.sirius/run
TOKEN=$(openssl rand -hex 16)             # keep this: the app needs it
printf '%s' "$TOKEN" > ~/.sirius/run/token
SIRIUS_TOKEN_FILE=~/.sirius/run/token sbatch \
    --output="$HOME/.sirius/run/sirius-worker-%j.log" app/python/slurm/sirius_worker.sbatch
```

The worker reads the token file and deletes it. `SIRIUS_TOKEN=... sbatch`
works too, but then the token is in the job's environment, which Slurm's
accounting may store (`AccountingStoreFlags=job_env`).

The template asks for one GPU and eight cores; edit the `#SBATCH` lines and
the `module load` block for your cluster. It refuses to start without a
token, since the port is open to every user of the node -- and so does the
worker itself: `--host 0.0.0.0` with an empty token is refused at startup
(`../SECURITY.md`). The log
(`sirius-worker-<jobid>.log`) prints the node name and the tunnel command;
the worker takes a free port (`SIRIUS_PORT` to fix one, at the risk of
another user of the node taking it first) and announces it in the log's
`{"port": N, ...}` line.

The worker needs a Python with `numpy`; `torch` for segmentation models and
the `sirius` wheel (`pip install .` from this repository) for SIM
reconstruction on the node. Everything else in the pipeline runs where it is
implemented -- see the list the worker prints in its `hello` reply. Install
what the worker needs into the job's environment (the venv or conda
environment the script activates) once, before you submit:

```
pip install -r app/python/requirements.txt                    # numpy: what it cannot start without
pip install -r app/python/requirements-extra.txt              # optional: scipy, scikit-image
```

From an installed tree the files are `share/sirius/python/requirements*.txt`.
A worker started without numpy prints a `missing_packages` line and exits
with code 3, so the log says at once what is missing
(`../README.md`, *What it needs to start*).

When `sirius-cli` is built on the cluster, it can make that environment
instead: `sirius-cli worker setup --yes` creates SIRIUS's own Python
environment under `~/.local/share/sirius/python-env`, or wherever
`SIRIUS_PYTHON_ENV` points (a project directory, when the home quota is
small); point the script's `source .../bin/activate` line at it.

The template passes `--max-clients ${SIRIUS_MAX_CLIENTS:-8}` to the worker,
so the application can keep a connection for its status and one for the
dataset it shows besides each run's (runs still execute one at a time).

## 2. Tunnel the port

From your workstation:

```
ssh -N -L 7645:<node>:<port> <login-node>
```

`<node>` and `<port>` are the compute node and the port from the log; the
login node forwards the connection. Leave the tunnel running for the
session.

## 3. Point the application at it

Preferences ▸ Compute, under *HPC worker*: host `localhost`, port `7645`,
token as above. Then choose **HPC** as the backend (Process ▸ Backend or the
Backend tiles in the parameters dock) and run a step: the application uploads the step's input
volumes, streams the worker's progress into the status bar and downloads the
result. Steps the worker cannot run (it advertises `run:<kind>` per
supported kind) report that in the log instead of failing silently.

Files referenced by parameters -- Torch models, OTFs, flat fields -- must be
readable **on the node**: give cluster paths in the parameters when the HPC
backend is selected.

`sirius-cli` reaches the same worker with `--hpc localhost:7645` and the
token in `SIRIUS_HPC_TOKEN` (never on its command line), for example
`SIRIUS_HPC_TOKEN=... sirius-cli --hpc localhost:7645 --backend hpc run --pipeline p.sirius.toml`
([../../cli/README.md](../../cli/README.md)).

## Without Slurm

The same worker runs anywhere:

```
SIRIUS_TOKEN=X python -m sirius_worker --host 0.0.0.0 --port 7645 --device cuda
```

The token is not optional here: a non-loopback `--host` without one is a
startup error. The application and the worker must also speak the same
protocol version (`hello` checks it), so update both ends together.

Locally `sirius-app` and `sirius-cli` start one themselves, in SIRIUS's own
Python environment unless an interpreter is named (see
`app/python/README.md`, which also says when).
