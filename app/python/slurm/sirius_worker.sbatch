#!/bin/bash
#SBATCH --job-name=sirius-worker
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sirius-worker-%j.log
#
# Runs the SIRIUS compute worker -- SIRIUS's engine, or the Python worker
# alone -- in its container image on a cluster node, for the application's
# HPC backend.
#
# sirius-app starts it for you (the Cluster button), in two steps: Connect
# gets a job that only holds the allocation (sbatch --wrap with the profile's
# partition, account, QoS, time and resources), then Start worker runs this
# script inside that job as a step,
#
#     srun --jobid=<job> --overlap --ntasks=1 [--gres=gpu:N] bash ~/.sirius/run/sirius_worker-<build>.sbatch
#
# with the image, the data folders, the worker's code (SIRIUS_WORKER_DIR) and
# the token file in its environment and its log in ~/.sirius/run. The
# application sends its own copy of this script there (named by its build)
# before every start: a checkout on the cluster may be older than the
# application, and its copy is never run. A new image or new data folders
# restart only this step; the job stays.
#
# By hand it can be submitted as a job of its own. Give the shared secret in
# a file only you can read -- the worker reads it and deletes it -- rather
# than in $SIRIUS_TOKEN, which Slurm's accounting may store with the job's
# environment (AccountingStoreFlags=job_env):
#
#     umask 077; mkdir -p ~/.sirius/run
#     TOKEN=$(openssl rand -hex 16); printf '%s' "$TOKEN" > ~/.sirius/run/token
#     SIRIUS_CONTAINER=/path/sirius-worker.sif SIRIUS_TOKEN_FILE=~/.sirius/run/token sbatch \
#         --output="$HOME/.sirius/run/sirius-worker-%j.log" app/python/slurm/sirius_worker.sbatch
#
# No partition, account or QoS is named above: Slurm's defaults for you apply
# unless you give them on the command line, which also overrides the lines
# above: sbatch --partition=... --account=... --qos=... --time=...
#
# then read the node name and the port from the log (the worker takes a free
# port: the {"port": N, ...} line) and tunnel it to your machine:
#
#     ssh -N -L 7645:<node>:<port> <login-node>
#
# and point the app at localhost:7645 with the same token (Preferences ▸
# Compute, HPC worker; or sirius-cli --hpc localhost:7645 with the token in
# SIRIUS_HPC_TOKEN). See README.md next to this file.
#
# --check runs the checks instead, where the worker would run and as it would
# run (the same image, binds, library path and environment), and starts
# nothing: the launcher, the image, sirius and numpy (and torch) importing in
# it, the worker's code visible inside it, `<engine> version` (SIRIUS's
# engine runs there, and which build it is), the GPUs the step was given, the
# data folders readable inside the image and the node cache folder writable.
# One line each, `check <name> <ok|warn|fail> <words>`, and `@@version <json>`
# for the engine's own answer; exit 0 unless one failed. The application runs
# it in the job, as a step of its own, before every worker start:
#
#     srun --jobid=<job> --overlap --ntasks=1 [--gres=gpu:N] bash sirius_worker-<build>.sbatch --check [<host path> <path in the image>]...
#
# with the data folders as pairs after --check. Nothing is tried on the login
# node: its scratch may be mounted noexec, the nodes' is not.

set -euo pipefail
# whatever the job writes is its owner's alone
umask 077

CHECK=0
if [[ "${1:-}" == --check ]]; then
    CHECK=1
    shift
fi
CHECK_DATA=("$@")

# A reason to stop, on stderr; with --check also as that check's line.
stop() {   # stop <check> <exit code> <words>
    local name="$1" rc="$2"
    shift 2
    if [[ "${CHECK}" == 1 ]]; then printf 'check %s fail %s\n' "${name}" "$*"; fi
    echo "$*" >&2
    exit "${rc}"
}

# --- the image -----------------------------------------------------------------
# The worker always runs in a container image (SIRIUS_CONTAINER, below): the
# node needs nothing installed but apptainer (or singularity), and the worker
# runs against exactly the libraries the image was built with.
if [[ -z "${SIRIUS_CONTAINER:-}" ]]; then
    stop image 1 "SIRIUS_CONTAINER is not set: the worker runs in a container image (an Apptainer/Singularity .sif holding the" \
         "compiled sirius package, numpy and torch); see app/python/slurm/README.md"
fi

# --- worker ------------------------------------------------------------------
# Port 0: a free port, which the worker announces ({"port": N} in the log). A
# fixed port is one another user of the node could take first.
SIRIUS_PORT="${SIRIUS_PORT:-0}"
SIRIUS_DEVICE="${SIRIUS_DEVICE:-cuda}"
if [[ "${CHECK}" == 0 && -z "${SIRIUS_TOKEN_FILE:-}" && -z "${SIRIUS_TOKEN:-}" ]]; then
    echo "neither SIRIUS_TOKEN_FILE nor SIRIUS_TOKEN is set; refusing to start an unauthenticated worker on a shared node" >&2
    exit 1
fi

# The worker package (sirius_worker): SIRIUS_WORKER_DIR, which the application
# sets (the engine build's python/ folder, else the checkout's app/python).
# By hand: the engine build's python/ when it has one, else app/python of the
# checkout the job was submitted from. Slurm runs a copy of this script from
# its spool directory, so the script's own location says nothing.
WORKER_DIR="${SIRIUS_WORKER_DIR:-}"
if [[ -z "${WORKER_DIR}" ]]; then
    ENGINE_PY=()
    [[ "${SIRIUS_ENGINE:-0}" == 1 && -n "${SIRIUS_ENGINE_DIR:-}" ]] && ENGINE_PY=("${SIRIUS_ENGINE_DIR}/python")
    for candidate in ${ENGINE_PY[@]+"${ENGINE_PY[@]}"} "${SLURM_SUBMIT_DIR:-$PWD}/app/python" "${SLURM_SUBMIT_DIR:-$PWD}" \
                     "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
        if [[ -f "${candidate}/sirius_worker/__main__.py" ]]; then WORKER_DIR="${candidate}"; break; fi
    done
fi
if [[ -z "${WORKER_DIR}" || ! -f "${WORKER_DIR}/sirius_worker/__main__.py" ]]; then
    stop worker 1 "cannot find sirius_worker${WORKER_DIR:+ in ${WORKER_DIR}} on $(hostname): submit from the sirius checkout, or set SIRIUS_WORKER_DIR=<checkout>/app/python"
fi
export PYTHONPATH="${WORKER_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${CHECK}" == 0 ]]; then
    echo "sirius worker on $(hostname) (device ${SIRIUS_DEVICE}), job ${SLURM_JOB_ID:-?}; its port is in the {\"port\": ...} line below"
    echo "tunnel:  ssh -N -L 7645:$(hostname):<port> <login-node>"
fi

# --- the job's GPUs ----------------------------------------------------------
# srun --gres=gpu:N gives the step the job's GPUs, and Slurm names them in
# CUDA_VISIBLE_DEVICES (SLURM_STEP_GPUS or SLURM_JOB_GPUS where it sets none).
# The image is entered with --cleanenv, which would leave that behind, and
# CUDA in the image would then see every GPU of a node that does not confine
# a job's devices: it goes in with the environment file below. A step given
# more GPUs than the job asked for (SIRIUS_GPUS, which the application sets)
# is held to the first of them, and says so.
GPU_IDS=""
GPU_SET=0
GPU_NOTE=""
WANT_GPUS=0
if [[ "${SIRIUS_DEVICE}" != cpu ]]; then
    WANT_GPUS="${SIRIUS_GPUS:-}"
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
        GPU_IDS="${CUDA_VISIBLE_DEVICES}"
        GPU_SET=1
    elif [[ -n "${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}" ]]; then
        GPU_IDS="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
        GPU_SET=1
    fi
    if [[ "${GPU_SET}" == 1 && "${WANT_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
        IFS=, read -r -a GIDS <<< "${GPU_IDS}"
        if (( ${#GIDS[@]} > WANT_GPUS )); then
            GPU_NOTE="the step was given ${#GIDS[@]} GPUs (${GPU_IDS}), the job asked for ${WANT_GPUS}: held to "
            GPU_IDS="$(IFS=,; echo "${GIDS[*]:0:WANT_GPUS}")"
            GPU_NOTE+="${GPU_IDS}"
        fi
    fi
    if [[ "${GPU_SET}" == 1 ]]; then export CUDA_VISIBLE_DEVICES="${GPU_IDS}"; fi
    if [[ "${CHECK}" == 0 ]]; then
        if [[ "${GPU_SET}" == 1 ]]; then echo "GPUs: CUDA_VISIBLE_DEVICES=${GPU_IDS}${GPU_NOTE:+ (${GPU_NOTE})}"
        else echo "GPUs: Slurm named none for this step (no CUDA_VISIBLE_DEVICES): CUDA sees every GPU the node shows it"; fi
    fi
fi

# No --allow-install: on a shared node the worker must not run pip or conda
# on a client's request (see ../SECURITY.md). Install the model packages into
# the environment above when you submit the job, not from the application.
# The token is read from the file $SIRIUS_TOKEN_FILE names (then deleted), or
# from $SIRIUS_TOKEN; never passed as an argument: the command line of a job
# is visible to every user of the node.
cd "${WORKER_DIR}"
# --max-clients: the application keeps a connection for its status and one for
# the dataset it shows (read here, on the node) besides each run's.
ARGS=(--host 0.0.0.0 --port "${SIRIUS_PORT}" --device "${SIRIUS_DEVICE}" --max-clients "${SIRIUS_MAX_CLIENTS:-8}")

# --- SIRIUS's engine instead of the Python worker alone ----------------------
# SIRIUS_ENGINE=1 (the application sets it from the profile) runs SIRIUS's C++
# engine, `sirius-cli serve`: every step of a pipeline then runs here, on the
# node, with CUDA where the library has it, and its results stay here until
# the application shows them (at screen size) or downloads them. The engine
# starts the Python worker as its child on 127.0.0.1 (torch models, plugins).
# SIRIUS_ENGINE_BIN names the executable: /opt/sirius/bin/sirius-cli in the
# container image by default. SIRIUS_ENGINE_DIR, when set, is a per-commit
# engine build (<engine_builds>/<commit>, with bin/sirius-cli, lib/, python/
# and BUILD.json), bound into the image so the image stays a stable runtime;
# its lib/ goes in front of the image's library path. The engine reads the
# same token file and prints the same {"port": N} line. Asked for and not
# there, nothing starts: the Python worker alone is never started instead.
CMD=(python -m sirius_worker "${ARGS[@]}")
ENGINE_BIN=""
ENGINE_HOME=""
if [[ "${SIRIUS_ENGINE:-0}" == 1 ]]; then
    ENGINE_BIN="${SIRIUS_ENGINE_BIN:-}"
    if [[ -z "${ENGINE_BIN}" ]]; then
        if [[ -n "${SIRIUS_ENGINE_DIR:-}" ]]; then ENGINE_BIN="${SIRIUS_ENGINE_DIR}/bin/sirius-cli"; else ENGINE_BIN=/opt/sirius/bin/sirius-cli; fi
    fi
    # the build folder the executable sits in (<home>/bin/sirius-cli), when it is on this file system
    if [[ -n "${SIRIUS_ENGINE_DIR:-}" ]]; then
        ENGINE_HOME="${SIRIUS_ENGINE_DIR}"
    elif [[ "${ENGINE_BIN}" != /opt/sirius/bin/sirius-cli && -d "$(dirname "$(dirname "${ENGINE_BIN}")")" ]]; then
        ENGINE_HOME="$(cd "$(dirname "$(dirname "${ENGINE_BIN}")")" && pwd)"
    fi
    CMD=("${ENGINE_BIN}" serve "${ARGS[@]}" --python python --worker-dir "${WORKER_DIR}")
    [[ -n "${SIRIUS_ENGINE_SCRATCH:-}" ]] && CMD+=(--scratch "${SIRIUS_ENGINE_SCRATCH}")
    # its own libraries (nvTIFF, nvCOMP) in front of the image's, set inside the image
    if [[ -n "${ENGINE_HOME}" && -d "${ENGINE_HOME}/lib" ]]; then
        CMD=(/bin/sh -c 'd=$1; shift; LD_LIBRARY_PATH="$d/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH; exec "$@"' \
             sirius-engine "${ENGINE_HOME}" "${CMD[@]}")
    fi
    if [[ "${CHECK}" == 0 ]]; then echo "SIRIUS engine: ${ENGINE_BIN} serve (the Python worker beside it, from ${WORKER_DIR})"; fi
fi

# The engine was asked for and is not there: said on stdout in the form the
# application reads ({"error": ...}; with --check its check line) and on
# stderr, and nothing is started.
engine_missing() {
    local msg="SIRIUS's C++ engine is not at $1 ($2): set Engine builds folder (Job > More options) to the folder holding per-commit builds, or Engine executable to one"
    if [[ "${CHECK}" == 1 ]]; then printf 'check engine-missing fail %s (%s)\n' "$1" "$2"
    else printf '{"error": "engine_missing", "path": "%s"}\n' "$1"; fi
    echo "${msg}" >&2
    exit 3
}

# --- what --check runs inside the image ------------------------------------------
# /bin/sh -c "$CHECK_BODY" sirius-check <worker dir> <engine> <engine build>
#     <node cache> <GPUs asked for> <GPU note> [<n> <data folder in the image>]...
# Each line `check <name> <ok|warn|fail> <words>`; exit 1 when one failed.
read -r -d '' CHECK_BODY <<'SH' || true
W=$1 E=$2 H=$3 S=$4 G=$5 N=$6
shift 6
bad=0
echo "check image ok starts on $(uname -n)"
# the engine's own libraries in front, as the engine (and its Python child) get them
if [ -n "$H" ] && [ -d "$H/lib" ]; then
    LD_LIBRARY_PATH="$H/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi
# a folder named sirius (a checkout in the working directory) imports as an empty namespace: not the package
if out=$(python -c 'import sys, importlib.util as u
import sirius, numpy
assert getattr(sirius, "__file__", None), "sirius is only a folder named so, not the compiled package"
print("check python ok python %d.%d, sirius, numpy" % sys.version_info[:2])
print("check torch ok in the image" if u.find_spec("torch") else "check torch warn not in the image: models will not run")' 2>&1); then
    printf '%s\n' "$out"
else
    printf '%s\n' "$out"
    echo "check python fail sirius and numpy do not import in the image"
    bad=1
fi
if [ -f "$W/sirius_worker/__main__.py" ]; then echo "check worker ok $W"
else echo "check worker fail $W/sirius_worker is not there inside the image"; bad=1; fi
if [ -n "$E" ] && [ ! -f "$E" ]; then
    echo "check engine-missing fail $E is not there inside the image"
    bad=1
elif [ -n "$E" ]; then
    out=$("$E" version 2>&1)
    rc=$?
    if [ "$rc" = 0 ]; then
        printf '@@version %s\n' "$(printf '%s' "$out" | tr -d '\r\n')"
        echo "check engine ok $E"
    else
        printf '%s\n' "$out"
        echo "check engine fail $(basename "$E") version exited with $rc"
        bad=1
    fi
fi
if [ "$G" != 0 ]; then
    k=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
        if [ "$k" -gt 0 ]; then echo "check gpu warn Slurm named no GPU for this step (no CUDA_VISIBLE_DEVICES): CUDA sees all $k GPUs of the node"
        else echo "check gpu warn no GPU is visible in the image (nvidia-smi lists none)"; fi
    else
        n=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
        if [ "$n" = 0 ]; then echo "check gpu warn CUDA_VISIBLE_DEVICES is empty: no GPU is visible"
        elif [ "$k" = 0 ]; then echo "check gpu warn CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, but nvidia-smi sees no GPU in the image"
        elif [ -n "$G" ] && [ "$n" != "$G" ]; then echo "check gpu warn $n visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES), the job asked for $G${N:+; $N}"
        else echo "check gpu ok $n visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)${N:+; $N}"; fi
    fi
fi
while [ $# -ge 2 ]; do
    if [ -d "$2" ] && ls "$2" >/dev/null 2>&1; then echo "check data:$1 ok $2"
    else echo "check data:$1 fail $2 cannot be read inside the image"; bad=1; fi
    shift 2
done
if [ -n "$S" ]; then
    t="$S/.sirius-check.$$"
    if mkdir -p "$S" 2>/dev/null && touch "$t" 2>/dev/null; then rm -f "$t"; echo "check cache ok $S"
    else echo "check cache fail cannot write to $S"; bad=1; fi
fi
exit $bad
SH

# --- the container -------------------------------------------------------------
# SIRIUS_CONTAINER=<image.sif> runs the worker inside that image. This is the
# practical way to serve a model on a cluster: torch, scipy and scikit-image
# are already in the image the models were trained in, so the node needs no
# multi-gigabyte install into a home directory, and the worker runs against
# exactly the libraries the weights were produced with.
#
#   SIRIUS_CONTAINER          the image (apptainer or singularity)
#   SIRIUS_CONTAINER_BIND     host paths to mount, apptainer's --bind syntax,
#                             e.g. "/data:/data:ro,/scratch". Nothing outside
#                             the image and $HOME is visible without this, so
#                             the data the app browses must be named here.
#   SIRIUS_CONTAINER_PYTHONPATH  extra entries appended after the worker's own
#   SIRIUS_CONTAINER_ARGS     anything else to pass to `apptainer exec`
#
# The image is entered with --cleanenv, so the environment is built here and
# handed over in a private file rather than on the command line: `apptainer
# --env SIRIUS_TOKEN=...` would put the shared secret in argv, where every
# user of the node can read it. A token given as $SIRIUS_TOKEN is written to a
# 0600 file first and passed by path; the worker reads that file and deletes it.
{
    # SIRIUS_LAUNCHER (the application sets it from the profile): apptainer by
    # default; a module of that name is loaded when it is not on PATH, and the
    # other runtime is tried when it still cannot be found.
    RUNTIME="${SIRIUS_LAUNCHER:-apptainer}"
    if ! command -v "${RUNTIME}" >/dev/null 2>&1 && type module >/dev/null 2>&1; then
        module load "${RUNTIME}" >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || module load singularity >/dev/null 2>&1 || true
    fi
    RUNTIME="$(command -v "${RUNTIME}" || command -v apptainer || command -v singularity || true)"
    if [[ -z "${RUNTIME}" ]]; then
        stop launcher 1 "neither ${SIRIUS_LAUNCHER:-apptainer} nor singularity can be run on $(hostname) (module load was tried)"
    fi
    if [[ "${CHECK}" == 1 ]]; then printf 'check launcher ok %s\n' "$(basename "${RUNTIME}")"; fi
    if [[ ! -r "${SIRIUS_CONTAINER}" ]]; then
        stop image 1 "SIRIUS_CONTAINER=${SIRIUS_CONTAINER} is not readable on $(hostname)"
    fi
    # the engine, when asked for, is there before anything starts: on this file
    # system (a build folder, bound into the image) or inside the image itself
    # (--check tries the image's own when it runs `<engine> version` there)
    if [[ "${SIRIUS_ENGINE:-0}" == 1 && ! -x "${ENGINE_BIN}" ]]; then
        if [[ -n "${ENGINE_HOME}" ]]; then engine_missing "${ENGINE_BIN}" "no executable file there on $(hostname)"; fi
        if [[ "${CHECK}" == 0 ]]; then
            "${RUNTIME}" exec --cleanenv "${SIRIUS_CONTAINER}" test -x "${ENGINE_BIN}" </dev/null >/dev/null 2>&1 ||
                engine_missing "${ENGINE_BIN}" "not in the image ${SIRIUS_CONTAINER}"
        fi
    fi
    RUNDIR="${SIRIUS_RUNDIR:-$HOME/.sirius/run}"; mkdir -p "${RUNDIR}"
    TOKEN_FILE="${SIRIUS_TOKEN_FILE:-}"
    if [[ "${CHECK}" == 0 && -z "${TOKEN_FILE}" ]]; then   # only $SIRIUS_TOKEN was given
        TOKEN_FILE="$(mktemp "${RUNDIR}/token.XXXXXX")"; chmod 600 "${TOKEN_FILE}"
        printf '%s' "${SIRIUS_TOKEN}" > "${TOKEN_FILE}"
    fi
    ENV_FILE="$(mktemp "${RUNDIR}/env.XXXXXX")"; chmod 600 "${ENV_FILE}"
    {
        printf 'PYTHONPATH=%s\n' "${WORKER_DIR}${SIRIUS_CONTAINER_PYTHONPATH:+:${SIRIUS_CONTAINER_PYTHONPATH}}"
        if [[ -n "${TOKEN_FILE}" ]]; then printf 'SIRIUS_TOKEN_FILE=%s\n' "${TOKEN_FILE}"; fi
        # The application reads the worker's port from the {"port": N} line, and a container's
        # stdout is a pipe, so python block-buffers it and the line never arrives: the worker looks
        # dead while it is serving. Cost one silent start on 2026-10-02.
        printf 'PYTHONUNBUFFERED=1\n'
        # the step's GPUs (above): --cleanenv leaves Slurm's own word for them behind
        if [[ "${GPU_SET}" == 1 ]]; then printf 'CUDA_VISIBLE_DEVICES=%s\n' "${GPU_IDS}"; fi
        for v in SIRIUS_LATENTS_PATH SIRIUS_MODEL_DIR HF_HOME MPLCONFIGDIR TORCH_HOME \
                 CELLPOSE_LOCAL_MODELS_PATH MICROSAM_CACHEDIR; do
            if [[ -n "${!v:-}" ]]; then printf '%s=%s\n' "$v" "${!v}"; fi
        done
    } > "${ENV_FILE}"
    GPU=(); [[ "${SIRIUS_DEVICE}" == cpu ]] || GPU=(--nv)
    BIND=(); [[ -n "${SIRIUS_CONTAINER_BIND:-}" ]] && BIND=(--bind "${SIRIUS_CONTAINER_BIND}")
    # The image sees $HOME, where the checkout and the run folder usually are; a
    # checkout or run folder elsewhere is bound, or the worker's code and its
    # token file would not be there inside.
    CHECKOUT="${WORKER_DIR%/app/python}"
    # the engine's node cache folder (SIRIUS_ENGINE_SCRATCH) too, made first,
    # and the per-commit engine build it runs (SIRIUS_ENGINE_DIR)
    SCRATCH=()
    if [[ "${SIRIUS_ENGINE:-0}" == 1 && -n "${SIRIUS_ENGINE_SCRATCH:-}" ]]; then
        mkdir -p "${SIRIUS_ENGINE_SCRATCH}" || stop cache 1 "cannot make the node cache folder ${SIRIUS_ENGINE_SCRATCH} on $(hostname)"
        SCRATCH=("${SIRIUS_ENGINE_SCRATCH}")
    fi
    [[ "${SIRIUS_ENGINE:-0}" == 1 && -n "${ENGINE_HOME}" ]] && SCRATCH+=("${ENGINE_HOME}")
    for d in "${CHECKOUT}" "${RUNDIR}" ${SCRATCH[@]+"${SCRATCH[@]}"}; do
        case "$(cd "$d" && pwd -P)/" in "$(cd "$HOME" && pwd -P)/"*) ;; *) BIND+=(--bind "$d") ;; esac
    done
    EXTRA=(); [[ -n "${SIRIUS_CONTAINER_ARGS:-}" ]] && read -r -a EXTRA <<< "${SIRIUS_CONTAINER_ARGS}"

    if [[ "${CHECK}" == 1 ]]; then
        # The data folders (pairs after --check): there on this node first --
        # apptainer stops on a bind whose host path is not -- then readable
        # inside the image.
        DATA=()
        missing=0
        i=0
        while (( i + 1 < ${#CHECK_DATA[@]} )); do
            if [[ -e "${CHECK_DATA[i]}" ]]; then DATA+=("$((i / 2))" "${CHECK_DATA[i + 1]}")
            else
                printf 'check data:%d fail %s is not on %s\n' "$((i / 2))" "${CHECK_DATA[i]}" "$(hostname)"
                missing=1
            fi
            i=$((i + 2))
        done
        if [[ "${missing}" == 1 ]]; then rm -f "${ENV_FILE}"; exit 1; fi
        ENGINE_ARG=""
        if [[ "${SIRIUS_ENGINE:-0}" == 1 ]]; then ENGINE_ARG="${ENGINE_BIN}"; fi
        rc=0
        "${RUNTIME}" exec "${GPU[@]}" --cleanenv "${BIND[@]}" "${EXTRA[@]}" \
            --env-file "${ENV_FILE}" --pwd "${WORKER_DIR}" \
            "${SIRIUS_CONTAINER}" /bin/sh -c "${CHECK_BODY}" sirius-check "${WORKER_DIR}" "${ENGINE_ARG}" "${ENGINE_HOME}" \
            "${SIRIUS_ENGINE_SCRATCH:-}" "${WANT_GPUS}" "${GPU_NOTE}" ${DATA[@]+"${DATA[@]}"} < /dev/null 2>&1 || rc=$?
        rm -f "${ENV_FILE}"
        exit "${rc}"
    fi

    echo "container: ${SIRIUS_CONTAINER} via $(basename "${RUNTIME}")${SIRIUS_CONTAINER_BIND:+, binds ${SIRIUS_CONTAINER_BIND}}"
    # The env file holds no secret (the token lives in its own 0600 file, which
    # the worker deletes), so handing the process over with exec is safe.
    exec "${RUNTIME}" exec "${GPU[@]}" --cleanenv "${BIND[@]}" "${EXTRA[@]}" \
        --env-file "${ENV_FILE}" --pwd "${WORKER_DIR}" \
        "${SIRIUS_CONTAINER}" "${CMD[@]}"
}
