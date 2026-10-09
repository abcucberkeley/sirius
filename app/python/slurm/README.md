# Running the compute worker under Slurm (the HPC backend)

The application's **HPC** backend sends steps to a running `sirius_worker`.
On a cluster the worker runs once per session as a Slurm job, and the
application reaches it through SSH.

## Connect to your cluster (the usual way)

SIRIUS works with any Slurm cluster you can reach with `ssh` and that runs
Apptainer (or Singularity). Nothing is installed into your home directory:
the worker runs in a container image, and the only other thing it needs is a
SIRIUS checkout on the cluster (its `app/python` is the worker's code).

1. **Once, on the cluster**: clone this repository (the same version as the
   application) — `git clone <this repository> ~/sirius` — and have a worker
   image (`.sif`). Someone at your site may have one; otherwise the
   application builds it (step 4).
2. **In the application**: the *Cluster* button in the title bar (or
   *Process ▸ Connect to cluster…*), page *1 Connect*: *Cluster*, an alias of your `~/.ssh/config` (with its user,
   ProxyJump and keys) or `user@login.example.org`; *⋯ ▸ New cluster profile*
   for another cluster. **Connect**: one SSH login — a password or one-time
   code is asked in the application and handed to ssh only, never stored.
3. **Start job** (page *2 Job*): a job that only holds a node, with the
   node type (partition), account, QoS, time and resources chosen there:

   ```
   sbatch --parsable --job-name=sirius --output=$HOME/.sirius/run/sirius-job-%j.log \
       [--partition=... --account=... --qos=... --time=...] --gres=gpu:N --cpus-per-task=C --mem=M \
       --wrap='... while :; do sleep 300 & wait $!; done'
   ```

   The login fills what the profile leaves empty from the cluster itself:
   the default partition (`sinfo`'s `*`), your account and QoS for it
   (`sacctmgr`), the checkout as `<home>/sirius`. The job's line follows the
   queue (state, reason, time waited) until the job runs on its node.
4. **Worker image** and **Data folders** (page *2 Job* as well): pick the image with *Browse…*
   (the cluster's files), and the folders your datasets live in with
   *Add folder…* (they are bound into the image; your home folder always is).
   No image yet: *More options… ▸ Build an image* runs `apptainer build --fakeroot` inside the
   job, after checking with a tiny test build that the cluster allows
   unprivileged builds (it says so plainly when it does not — then use an
   image someone built).
5. **Start worker** (page *3 Worker*): the checks (srun, the worker's code,
   the image there and readable, `import sirius, numpy` inside it, the
   launcher — `module load apptainer` is tried — each data folder, SIRIUS's
   engine, the node cache folder), then the worker as a step of the held job:

   ```
   srun --jobid=<job> --overlap --nodes=1 --ntasks=1 --job-name=sirius-worker [--gres=gpu:N] \
       bash ~/.sirius/run/sirius_worker-<build>.sbatch
   ```

   The script is **the application's own** `sirius_worker.sbatch`, compiled
   into it and written over the SSH session to `~/.sirius/run` (`0700`,
   named by the application's build) before every start: the copy in a
   checkout on the cluster may be older than the application and is never
   run. It runs SIRIUS's engine (`sirius-cli serve`, with the Python worker as
   its child) in the image: `<launcher> exec [--nv] --cleanenv --bind ...
   <image> sirius-cli serve ...`. The application reaches it through the SSH
   connection itself (a SOCKS proxy on `ssh -D`): no `ssh -L`, no node name
   to copy.

   **The engine is never optional when it is asked for.** The checks make
   sure it will be there inside the image (the engine build picked from the
   *Engine builds folder*, the *Engine executable*, or the image's own
   `/opt/sirius/bin/sirius-cli`, tested with `<launcher> exec <image> test
   -x ...`) and fail otherwise, with what to set; the launch script stops
   with `{"error": "engine_missing", ...}` rather than start the Python
   worker alone; and a worker whose hello has no engine is a failed *Connect
   to the worker*, not a connection. Without the engine the *Worker* page
   says *Not ready*, the HPC backend is not chosen, and every Run button is
   disabled with the reason (*HPC: no SIRIUS engine on the cluster — open
   Cluster to fix*).

Once connected the title bar says *g0003 · GPU* (or *· CPU*); its tooltip
has the host, the job, the node and the time left. A worker that stops
leaves the job held (*no worker yet*); a job that ends (TIMEOUT, CANCELLED)
or a connection that drops is said in red, with the reason. *Disconnect…*
(and quitting) asks whether to `scancel` the job; one left running is taken
up again — with its worker, and what that holds — by the next *Start job*.

Changes while connected: another image, other data folders or software
settings restart only the worker step, in the same job (*Restart worker*);
another partition, account, QoS, time or size needs a new job (*Change job…*
on page 2 asks before cancelling the running one, and keeps the login).

**The job's choices** (each field has a tooltip): the node type (partition),
account, QoS and time limit are dropdowns — the profile's own choices first,
then what the cluster reports (`sinfo -h -o '%P|%a|%l|%D|%t|%G|%c|%m'`,
`sacctmgr -n -P show assoc user="$USER" format=partition,account,qos,defaultqos`,
`sacctmgr -n -P show qos format=name,maxwall`, `scontrol -o show partition`;
only sinfo has to answer), marked *from the cluster*, with *Add … to my
settings*; a partition whose jobs take whole nodes, or whose GPUs a group
shares (a DGX), shows a warning. *More options…* holds the SIRIUS checkout, the launcher, an
extra `PYTHONPATH`, the C++ engine and its builds folder (below), and the
node cache folder (the engine's `--scratch`: a node's local disk is fastest;
empty is the node's temporary folder).

*Cluster device: GPU | CPU*, beside the backend tiles when HPC is selected
(and in *Preferences ▸ Compute*; `sirius-cli --hpc-device`, `set_backend`'s
`hpc_device`), says where the worker computes. Every step and every dataset
read carries it as `"device": "cuda"` / `"cpu"`, so switching needs no new
job. A job without a GPU greys out GPU; a GPU asked of it anyway fails with
"this worker job has no GPU; choose CPU or reconnect with GPUs >= 1".

### The profiles in the settings file

Each cluster is a `[cluster.<name>]` table in the application's settings file
(`sirius-app.toml`; *Preferences ▸ Edit settings file…* edits it, checked as
you type), with the partitions its dropdowns offer as
`[[cluster.<name>.partitions]]` — `docs/clusters.example.toml` is a commented
example. *Import…* / *Export…* in the dialog share one profile as a small
`.toml` file. No password and no token is ever written there: ssh asks for
the password and forgets it, and the worker's token is made anew for each
worker, written over the SSH session to a private file in `~/.sirius/run` (a
`0700` directory, the file `0600`) whose name is all the step is given; the
worker deletes it as it starts. The logs are
`~/.sirius/run/sirius-job-<job>.log` and `sirius-worker-<job>-<n>.log`, and
the worker takes a free port, which it announces there (`../SECURITY.md`).

### Engine builds

The image is a stable runtime (Python, numpy, torch, the compiled `sirius`
package); SIRIUS's engine changes with every commit. With *Engine builds
folder* set, the engine comes from `<folder>/<commit>/bin/sirius-cli`, each
build with its `<folder>/<commit>/BUILD.json` (`{"build", "commit",
"ops_schema", "ops_generation", "api", ...}`, what `sirius-cli --version
--json` prints). The checks list the builds there and pick the one of the
application's own commit, else the newest whose operations and engine API are
the same (`core/build_info.hpp`'s `engineMismatch`); that folder is bound into
the image. `BUILD.json` may name the schema hash `ops_schema` or
`schema_hash` (or both). When none fits, the checks say so and name the
application's commit.

**What has to match is `ops_generation`, not the schema hash.** The hash
covers help text and labels too, so it moves when nothing about what an
operation means has moved, and an engine refused for such a difference would
have to be rebuilt for nothing. `ops_generation` is bumped only when the
operation set changes incompatibly, so an engine built from a nearby commit
keeps serving. An engine built before `ops_generation` existed reports only a
hash; it is served when that hash is this build's own or one
`kAcceptedOlderOpsSchemas` names (with the reason). Either way a refusal says
which of the two sides is the older one and gives the fix for that side.

The build's `python/` folder (the `sirius_worker` package of that
commit) is the worker's code, so the checkout on the cluster is not needed
then; its `lib/` (nvTIFF, nvCOMP) goes in front of the image's library path.
Building one, inside the worker image on a node (latents'
`scripts/build_sirius_engine.sbatch` does exactly this for a commit):

```
git -C <checkout> archive <commit> | tar -x -C <scratch>/src     # never a working tree
cmake -S <scratch>/src -B <scratch>/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSIRIUS_ENABLE_CLI=ON -DSIRIUS_ENABLE_APP=OFF -DSIRIUS_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DSIRIUS_ENABLE_PYTHON_BINDINGS=OFF -DSIRIUS_ENABLE_TENSORSTORE=OFF -DSIRIUS_ENABLE_TESTS=OFF \
    -DSIRIUS_BUILD_COMMIT=<commit> -DSIRIUS_BUILD_DIRTY=OFF
cmake --build <scratch>/build --target sirius-cli
# then <folder>/<commit>/: bin/sirius-cli, lib/ (its shared libraries), python/ (app/python),
# help/, and BUILD.json = the "build" object of `bin/sirius-cli version`
```

A build of another commit serves the application as long as the operation
schema and the engine API are the same: a new build is needed only when
they changed.

A build is a plain, self-contained folder — `bin/sirius-cli`, `lib/`,
`python/`, `help/`, `BUILD.json` — and nothing else: no symlinks into
`$HOME`, no copy of the executable elsewhere. It may live on a scratch file
system the login node mounts `noexec` (fiona's `/clusterfs/nvme2`): the login
node only *reads* it. There the checks look at files alone — the folder,
`BUILD.json` parsed, `bin/sirius-cli` a file (or a link to one) with an
execute bit in its mode (`ls -lL`, never `[ -x ]`, which asks
`access(X_OK)` and says no on a `noexec` mount), `python/sirius_worker`,
`lib/` — to list the builds and pick one. Whether it runs is checked where it
runs: in the job, on its compute node, inside the image, before the worker
starts (below). `chmod +x bin/sirius-cli` is the fix for a build listed
"without an execute bit".

### The checks in the job

Before every worker start the application runs its own launch script with
`--check` as one step of the job (`srun --jobid=<job> --overlap --ntasks=1
[--gres=gpu:N] bash ~/.sirius/run/sirius_worker-<build>.sbatch --check
<data folder pairs>`), so it runs on the compute node with the same image,
binds, library path and environment as the start, in a couple of seconds:

| check | what | fails when |
|---|---|---|
| launcher | `apptainer` (or `singularity`, `module load` tried) on the node | neither runs there |
| image | the image starts | it is unreadable, or does not start |
| python | `import sirius, numpy` in the image (torch noted) | they do not import |
| worker | the worker's code is visible inside the image | it is not there |
| engine | `<engine> version` in the image, its `build` the same as the picked build's `BUILD.json`, and of this application's operations | it is not there, does not run (its output is shown), or is another build |
| gpu / cuda | `CUDA_VISIBLE_DEVICES` names as many GPUs as the job asked for, `nvidia-smi` sees them, the engine's CUDA finds them | (a warning) |
| data:*n* | each data folder is on the node, and readable inside the image | it is not |
| cache | the node cache folder can be written | it cannot |

Each comes back as a line of the Worker page's checklist with what to do;
one that fails stops the start, and nothing runs (the worker is never started
without the engine it was asked for). By hand:

```
SIRIUS_CONTAINER=<image> SIRIUS_ENGINE=1 SIRIUS_ENGINE_DIR=<builds>/<commit> SIRIUS_WORKER_DIR=<builds>/<commit>/python \
    srun --jobid=<job> --overlap --ntasks=1 bash app/python/slurm/sirius_worker.sbatch --check /data /data
```

### The job's GPUs

The worker's step (and the check's) asks for the job's GPUs
(`--gres=gpu:N`), and Slurm names them in `CUDA_VISIBLE_DEVICES`. The image is
entered with `--cleanenv`, which would leave that behind — CUDA in the image
then sees every GPU of a node that does not confine a job's devices — so the
launch script hands it into the image with the rest of the environment. A
step given more GPUs than the job asked for is held to the first of them,
and the log and the checks say so.

### A job of your own

The Job page lists your jobs that run or wait on the cluster (any of them,
`squeue -u $USER`), each with *Use this job*: SIRIUS's worker then runs in
that job as a step, with its GPUs, after the checks on its node — no new job
is asked for. SIRIUS never cancels such a job: Disconnect and *Let go of this
job* leave it running, and Connect takes it up again while it runs.

Empty, the engine inside the image (`/opt/sirius/bin/sirius-cli`) is used;
*Engine executable* names one outright (testing a build of your own).

### In a container

The **worker image** (an Apptainer/Singularity `.sif` holding the compiled
`sirius` package with CUDA/nvTIFF, numpy and torch) is required: there is no
Python environment mode on the cluster. The worker step gets
`SIRIUS_CONTAINER=<image>` and `SIRIUS_LAUNCHER` (default `apptainer`), and
`sirius_worker.sbatch` runs

```
apptainer exec --nv --bind <checkout> --bind ~/.sirius/run <image> python -m sirius_worker ...
```

`--nv` only when the job has GPUs (not for `SIRIUS_DEVICE=cpu`); the worker's
code comes from the checkout (on `PYTHONPATH`, which the container inherits),
and the token file is read and deleted in `~/.sirius/run` as without a
container. When the launcher is not on `PATH`, `module load` of it (then of
`apptainer`, `singularity`) is tried, and then the other of the two.
`SIRIUS_CONTAINER_BIND` adds binds of your own (comma separated: a data file
system your site does not bind into every container already). The checks
step makes sure the image is there and runs
`<launcher> exec <image> python -c "import sirius, numpy"` on the login node;
when that fails, the image itself has to be rebuilt (or the current one
asked for): nothing is installed into it from the application.

By hand: `SIRIUS_CONTAINER=/path/sirius-worker.sif SIRIUS_TOKEN_FILE=... sbatch ... app/python/slurm/sirius_worker.sbatch`
(the script refuses to start without `SIRIUS_CONTAINER`).

The manual way follows: for `sirius-cli`, or a cluster the dialog does not fit.

## 1. Start the worker on a node

```
umask 077; mkdir -p ~/.sirius/run
TOKEN=$(openssl rand -hex 16)             # keep this: the app needs it
printf '%s' "$TOKEN" > ~/.sirius/run/token
SIRIUS_CONTAINER=/path/sirius-worker.sif SIRIUS_TOKEN_FILE=~/.sirius/run/token sbatch \
    --output="$HOME/.sirius/run/sirius-worker-%j.log" app/python/slurm/sirius_worker.sbatch
```

The worker reads the token file and deletes it. `SIRIUS_TOKEN=... sbatch`
works too, but then the token is in the job's environment, which Slurm's
accounting may store (`AccountingStoreFlags=job_env`).

The template asks for one GPU and eight cores (give `--partition`,
`--account`, `--qos` and `--time` on the command line, or edit the `#SBATCH`
lines). It refuses to start without an image or a token, since the port is open to every user of the node -- and so does the
worker itself: `--host 0.0.0.0` with an empty token is refused at startup
(`../SECURITY.md`). The log
(`sirius-worker-<jobid>.log`) prints the node name and the tunnel command;
the worker takes a free port (`SIRIUS_PORT` to fix one, at the risk of
another user of the node taking it first) and announces it in the log's
`{"port": N, ...}` line.

The image holds what the worker needs: a Python with `numpy`, `torch` for
segmentation models and the compiled `sirius` package (SIRIUS's TIFF reader,
SIM reconstruction on the node); nothing is installed on the cluster. A
worker started without numpy prints a `missing_packages` line and exits with
code 3, so the log says at once what the image lacks
(`../README.md`, *What it needs to start*).

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

## The container

`SIRIUS_CONTAINER=<image.sif>` runs the worker inside that image (apptainer or
singularity). This is the practical way to serve a trained model on a cluster:
torch, scipy and scikit-image are already in the image the model was trained
in, so the node needs no multi-gigabyte install into a home directory, and the
worker runs against exactly the libraries the weights were produced with.

    SIRIUS_CONTAINER             the image
    SIRIUS_CONTAINER_BIND        host paths to mount, apptainer's --bind syntax.
                                 Nothing outside the image and $HOME is visible
                                 without this, so name the data the app browses.
    SIRIUS_CONTAINER_PYTHONPATH  extra entries after the worker's own
    SIRIUS_CONTAINER_ARGS        anything else for `apptainer exec`

The image is entered with `--cleanenv` and the environment is handed over in a
private file, not on the command line: `apptainer --env SIRIUS_TOKEN=...` would
put the shared secret in argv, where every user of the node can read it. A token
given as `$SIRIUS_TOKEN` is written to a 0600 file first and passed by path; the
worker reads that file and deletes it. `PYTHONUNBUFFERED=1` goes in too, because
a container's stdout is a pipe and the `{"port": N}` line the application waits
for would otherwise sit in python's buffer while the worker is already serving.

An example from one cluster (fiona's dgx, verified end to end on 2026-10-02:
the worker started in the latents image on an A100, answered the handshake,
loaded a `.ltb` bundle, and returned a prompted mask in 3.6 s); the paths,
partition, account and QoS are that site's, yours will differ:

    umask 077; mkdir -p ~/.sirius/run
    TOKEN=$(openssl rand -hex 16); printf '%s' "$TOKEN" > ~/.sirius/run/token
    SIRIUS_TOKEN_FILE=~/.sirius/run/token \
    SIRIUS_CONTAINER=/clusterfs/nvme2/Users/velatkilic/containers/latents.sif \
    SIRIUS_CONTAINER_BIND=/clusterfs/vast/velatkilic,/clusterfs/nvme2/Users/velatkilic \
    SIRIUS_CONTAINER_PYTHONPATH=/clusterfs/nvme2/Users/velatkilic/pylibs/lib/python3.12/site-packages \
\
    sbatch --partition=dgx --account=co_abc --qos=abc_high --time=04:00:00 \
        --output="$HOME/.sirius/run/sirius-worker-%j.log" \
        app/python/slurm/sirius_worker.sbatch

(The Foundation step's models are self-contained folders now: the worker imports
a model folder's own `model.py` and needs no latents checkout, so the
`SIRIUS_LATENTS_PATH` that example once carried is gone. Bind the models folder
so the worker sees it.)

A bind is not optional on a cluster: without `SIRIUS_CONTAINER_BIND` the worker
cannot see the data directories, and a path that does not exist (or that you
cannot read) makes `apptainer` fail before the worker ever starts. Name only the
trees you need, and mount read-only anything you must not write: `/data:/data:ro`.
