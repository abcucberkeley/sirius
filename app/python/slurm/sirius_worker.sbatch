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
# sirius-app starts it for you (Process > Connect to cluster...), in two
# steps: Connect gets a job that only holds the allocation (sbatch --wrap
# with the profile's partition, account, QoS, time and resources), then
# Start worker runs this script inside that job as a step,
#
#     srun --jobid=<job> --overlap --ntasks=1 [--gres=gpu:N] bash app/python/slurm/sirius_worker.sbatch
#
# from the SIRIUS checkout, with the image, the data folders and the token
# file in its environment and its log in ~/.sirius/run. A new image or new
# data folders restart only this step; the job stays.
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

set -euo pipefail
# whatever the job writes is its owner's alone
umask 077

# --- the image -----------------------------------------------------------------
# The worker always runs in a container image (SIRIUS_CONTAINER, below): the
# node needs nothing installed but apptainer (or singularity), and the worker
# runs against exactly the libraries the image was built with.
if [[ -z "${SIRIUS_CONTAINER:-}" ]]; then
    echo "SIRIUS_CONTAINER is not set: the worker runs in a container image (an Apptainer/Singularity .sif holding the" \
         "compiled sirius package, numpy and torch); see app/python/slurm/README.md" >&2
    exit 1
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
echo "tunnel:  ssh -N -L 7645:$(hostname):<port> <login-node>"

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
# engine build (<engine_builds>/<commit>, with bin/sirius-cli and BUILD.json),
# bound into the image so the image stays a stable runtime. The engine reads
# the same token file and prints the same {"port": N} line.
CMD=(python -m sirius_worker "${ARGS[@]}")
if [[ "${SIRIUS_ENGINE:-0}" == 1 ]]; then
    ENGINE_BIN="${SIRIUS_ENGINE_BIN:-}"
    if [[ -z "${ENGINE_BIN}" ]]; then
        if [[ -n "${SIRIUS_ENGINE_DIR:-}" ]]; then ENGINE_BIN="${SIRIUS_ENGINE_DIR}/bin/sirius-cli"; else ENGINE_BIN=/opt/sirius/bin/sirius-cli; fi
    fi
    CMD=("${ENGINE_BIN}" serve "${ARGS[@]}" --python python --worker-dir "${WORKER_DIR}")
    [[ -n "${SIRIUS_ENGINE_SCRATCH:-}" ]] && CMD+=(--scratch "${SIRIUS_ENGINE_SCRATCH}")
    echo "SIRIUS engine: ${ENGINE_BIN} serve (the Python worker beside it, from ${WORKER_DIR})"
fi

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
        echo "neither ${SIRIUS_LAUNCHER:-apptainer} nor singularity can be run here (module load was tried)" >&2; exit 1
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
    # the engine's node cache folder (SIRIUS_ENGINE_SCRATCH) too, made first,
    # and the per-commit engine build it runs (SIRIUS_ENGINE_DIR)
    SCRATCH=()
    if [[ "${SIRIUS_ENGINE:-0}" == 1 && -n "${SIRIUS_ENGINE_SCRATCH:-}" ]]; then
        mkdir -p "${SIRIUS_ENGINE_SCRATCH}" || { echo "cannot make the node cache folder ${SIRIUS_ENGINE_SCRATCH}" >&2; exit 1; }
        SCRATCH=("${SIRIUS_ENGINE_SCRATCH}")
    fi
    [[ "${SIRIUS_ENGINE:-0}" == 1 && -n "${SIRIUS_ENGINE_DIR:-}" ]] && SCRATCH+=("${SIRIUS_ENGINE_DIR}")
    for d in "${CHECKOUT}" "${RUNDIR}" ${SCRATCH[@]+"${SCRATCH[@]}"}; do
        case "$(cd "$d" && pwd -P)/" in "$(cd "$HOME" && pwd -P)/"*) ;; *) BIND+=(--bind "$d") ;; esac
    done
    EXTRA=(); [[ -n "${SIRIUS_CONTAINER_ARGS:-}" ]] && read -r -a EXTRA <<< "${SIRIUS_CONTAINER_ARGS}"
    echo "container: ${SIRIUS_CONTAINER} via $(basename "${RUNTIME}")${SIRIUS_CONTAINER_BIND:+, binds ${SIRIUS_CONTAINER_BIND}}"
    # The env file holds no secret (the token lives in its own 0600 file, which
    # the worker deletes), so handing the process over with exec is safe.
    exec "${RUNTIME}" exec "${GPU[@]}" --cleanenv "${BIND[@]}" "${EXTRA[@]}" \
        --env-file "${ENV_FILE}" --pwd "${WORKER_DIR}" \
        "${SIRIUS_CONTAINER}" "${CMD[@]}"
}
