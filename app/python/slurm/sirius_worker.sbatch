#!/bin/bash
#SBATCH --job-name=sirius-worker
#SBATCH --partition=abc_a100
#SBATCH --account=velatkilic
#SBATCH --qos=abc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=sirius-worker-%j.log
#
# Runs the SIRIUS compute worker on a cluster node for the application's HPC
# backend. sirius-app submits it for you (Process > Connect to cluster...:
# one SSH login, then this script with the profile's partition, account, QoS,
# time and resources on the sbatch command line, the token in a private file
# whose name is $SIRIUS_TOKEN_FILE and the log in ~/.sirius/run). By hand,
# submit with a shared secret in a file only you can read -- the worker reads
# it and deletes it -- rather than in $SIRIUS_TOKEN, which Slurm's accounting
# may store with the job's environment (AccountingStoreFlags=job_env):
#
#     umask 077; mkdir -p ~/.sirius/run
#     TOKEN=$(openssl rand -hex 16); printf '%s' "$TOKEN" > ~/.sirius/run/token
#     SIRIUS_TOKEN_FILE=~/.sirius/run/token sbatch \
#         --output="$HOME/.sirius/run/sirius-worker-%j.log" app/python/slurm/sirius_worker.sbatch
#
# The partition, account and QoS above are fiona's test GPUs (abc_debug: one
# hour at most); give your own on the command line elsewhere, which overrides
# them: sbatch --partition=... --account=... --qos=... --time=...
#
# then read the node name and the port from the log (the worker takes a free
# port: the {"port": N, ...} line) and tunnel it to your machine:
#
#     ssh -N -L 7645:<node>:<port> <login-node>
#
# and point the app at localhost:7645 with the same token (Preferences ▸
# Compute, HPC worker; or sirius-cli --hpc localhost:7645 with the token in
# SIRIUS_HPC_TOKEN). See README.md next to this file.

set -euo pipefail
# whatever the job writes is its owner's alone
umask 077

# --- environment (Fiona; adapt the modules to your cluster) -----------------
# SIRIUS_CONTAINER names an Apptainer/Singularity image that holds the
# compiled sirius package, numpy and torch: the worker then runs in it (see
# "worker" below) and none of this environment is used.
if [[ -z "${SIRIUS_CONTAINER:-}" ]]; then
module load python
module load cmake 2>/dev/null || true
module load nvhpc 2>/dev/null || true
fi
# A Python with numpy (and torch for segmentation models); the sirius wheel
# enables SIM reconstruction on the node. Install what the worker needs into
# that environment once, before submitting (share/sirius/python/ in an
# installed tree):
#     pip install -r app/python/requirements.txt          # numpy
#     pip install -r app/python/requirements-extra.txt    # optional: scipy, scikit-image
# Without numpy the worker exits with code 3 and a missing_packages line in
# the log. Uncomment for a venv/conda (or SIRIUS's own environment, which
# `sirius-cli worker setup` makes under ~/.local/share/sirius/python-env):
# source "$HOME/venvs/sirius/bin/activate"
# SIRIUS_VENV names one to activate without editing this file.
if [[ -n "${SIRIUS_VENV:-}" && -z "${SIRIUS_CONTAINER:-}" ]]; then
    # shellcheck disable=SC1091
    source "${SIRIUS_VENV}/bin/activate"
fi

# --- worker ------------------------------------------------------------------
# Port 0: a free port, which the worker announces ({"port": N} in the log). A
# fixed port is one another user of the node could take first.
SIRIUS_PORT="${SIRIUS_PORT:-0}"
SIRIUS_DEVICE="${SIRIUS_DEVICE:-cuda}"
if [[ -z "${SIRIUS_TOKEN_FILE:-}" && -z "${SIRIUS_TOKEN:-}" ]]; then
    echo "neither SIRIUS_TOKEN_FILE nor SIRIUS_TOKEN is set; refusing to start an unauthenticated worker on a shared node" >&2
    exit 1
fi

# The worker package is app/python of the checkout the job was submitted from.
# Slurm runs a copy of this script from its spool directory, so the script's
# own location says nothing; SIRIUS_WORKER_DIR overrides the guess.
WORKER_DIR="${SIRIUS_WORKER_DIR:-}"
if [[ -z "${WORKER_DIR}" ]]; then
    for candidate in "${SLURM_SUBMIT_DIR:-$PWD}/app/python" "${SLURM_SUBMIT_DIR:-$PWD}"                      "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
        if [[ -f "${candidate}/sirius_worker/__main__.py" ]]; then WORKER_DIR="${candidate}"; break; fi
    done
fi
if [[ -z "${WORKER_DIR}" ]]; then
    echo "cannot find sirius_worker: submit from the sirius checkout, or set SIRIUS_WORKER_DIR=<checkout>/app/python" >&2
    exit 1
fi
export PYTHONPATH="${WORKER_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "sirius worker on $(hostname) (device ${SIRIUS_DEVICE}), job ${SLURM_JOB_ID:-?}; its port is in the {\"port\": ...} line below"
echo "tunnel:  ssh -N -L 7645:$(hostname):<port> <login-node>   (fiona: fiona.abc.berkeley.edu)"

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
# container image by default, sirius-cli on PATH without one. The engine reads
# the same token file and prints the same {"port": N} line.
CMD=(python -m sirius_worker "${ARGS[@]}")
if [[ "${SIRIUS_ENGINE:-0}" == 1 ]]; then
    ENGINE_BIN="${SIRIUS_ENGINE_BIN:-}"
    if [[ -z "${ENGINE_BIN}" ]]; then
        if [[ -n "${SIRIUS_CONTAINER:-}" ]]; then ENGINE_BIN=/opt/sirius/bin/sirius-cli; else ENGINE_BIN=sirius-cli; fi
    fi
    CMD=("${ENGINE_BIN}" serve "${ARGS[@]}" --python python --worker-dir "${WORKER_DIR}")
    [[ -n "${SIRIUS_ENGINE_SCRATCH:-}" ]] && CMD+=(--scratch "${SIRIUS_ENGINE_SCRATCH}")
    echo "SIRIUS engine: ${ENGINE_BIN} serve (the Python worker beside it, from ${WORKER_DIR})"
fi

# --- a container instead of a venv -------------------------------------------
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
if [[ -n "${SIRIUS_CONTAINER:-}" ]]; then
    # SIRIUS_LAUNCHER (the application sets it from the profile): apptainer by
    # default; a module of that name is loaded when it is not on PATH, and the
    # other runtime is tried when it still cannot be found.
    RUNTIME="${SIRIUS_LAUNCHER:-apptainer}"
    if ! command -v "${RUNTIME}" >/dev/null 2>&1 && type module >/dev/null 2>&1; then
        module load "${RUNTIME}" >/dev/null 2>&1 || module load apptainer >/dev/null 2>&1 || module load singularity >/dev/null 2>&1 || true
    fi
    RUNTIME="$(command -v "${RUNTIME}" || command -v apptainer || command -v singularity || true)"
    if [[ -z "${RUNTIME}" ]]; then
        echo "SIRIUS_CONTAINER is set but neither ${SIRIUS_LAUNCHER:-apptainer} nor singularity can be run here (module load was tried)" >&2; exit 1
    fi
    if [[ ! -r "${SIRIUS_CONTAINER}" ]]; then
        echo "SIRIUS_CONTAINER=${SIRIUS_CONTAINER} is not readable" >&2; exit 1
    fi
    RUNDIR="${SIRIUS_RUNDIR:-$HOME/.sirius/run}"; mkdir -p "${RUNDIR}"
    TOKEN_FILE="${SIRIUS_TOKEN_FILE:-}"
    if [[ -z "${TOKEN_FILE}" ]]; then                 # only $SIRIUS_TOKEN was given
        TOKEN_FILE="$(mktemp "${RUNDIR}/token.XXXXXX")"; chmod 600 "${TOKEN_FILE}"
        printf '%s' "${SIRIUS_TOKEN}" > "${TOKEN_FILE}"
    fi
    ENV_FILE="$(mktemp "${RUNDIR}/env.XXXXXX")"; chmod 600 "${ENV_FILE}"
    {
        printf 'PYTHONPATH=%s\n' "${WORKER_DIR}${SIRIUS_CONTAINER_PYTHONPATH:+:${SIRIUS_CONTAINER_PYTHONPATH}}"
        printf 'SIRIUS_TOKEN_FILE=%s\n' "${TOKEN_FILE}"
        # The application reads the worker's port from the {"port": N} line, and a container's
        # stdout is a pipe, so python block-buffers it and the line never arrives: the worker looks
        # dead while it is serving. Cost one silent start on 2026-10-02.
        printf 'PYTHONUNBUFFERED=1\n'
        for v in SIRIUS_LATENTS_PATH SIRIUS_MODEL_DIR HF_HOME MPLCONFIGDIR TORCH_HOME \
                 CELLPOSE_LOCAL_MODELS_PATH MICROSAM_CACHEDIR; do
            [[ -n "${!v:-}" ]] && printf '%s=%s\n' "$v" "${!v}"
        done
    } > "${ENV_FILE}"
    GPU=(); [[ "${SIRIUS_DEVICE}" == cpu ]] || GPU=(--nv)
    BIND=(); [[ -n "${SIRIUS_CONTAINER_BIND:-}" ]] && BIND=(--bind "${SIRIUS_CONTAINER_BIND}")
    # The image sees $HOME, where the checkout and the run folder usually are; a
    # checkout or run folder elsewhere is bound, or the worker's code and its
    # token file would not be there inside.
    CHECKOUT="${WORKER_DIR%/app/python}"
    for d in "${CHECKOUT}" "${RUNDIR}"; do
        case "$(cd "$d" && pwd -P)/" in "$(cd "$HOME" && pwd -P)/"*) ;; *) BIND+=(--bind "$d") ;; esac
    done
    EXTRA=(); [[ -n "${SIRIUS_CONTAINER_ARGS:-}" ]] && read -r -a EXTRA <<< "${SIRIUS_CONTAINER_ARGS}"
    echo "container: ${SIRIUS_CONTAINER} via $(basename "${RUNTIME}")${SIRIUS_CONTAINER_BIND:+, binds ${SIRIUS_CONTAINER_BIND}}"
    # The env file holds no secret (the token lives in its own 0600 file, which
    # the worker deletes), so handing the process over with exec is safe.
    exec "${RUNTIME}" exec "${GPU[@]}" --cleanenv "${BIND[@]}" "${EXTRA[@]}" \
        --env-file "${ENV_FILE}" --pwd "${WORKER_DIR}" \
        "${SIRIUS_CONTAINER}" "${CMD[@]}"
fi

exec "${CMD[@]}"
