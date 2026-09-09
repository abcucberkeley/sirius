#!/bin/bash
#SBATCH --job-name=sirius-worker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=sirius-worker-%j.log
#
# Runs the SIRIUS compute worker on a cluster node for the application's HPC
# backend. Submit with a shared secret:
#
#     SIRIUS_TOKEN=$(openssl rand -hex 16) sbatch app/python/slurm/sirius_worker.sbatch
#
# then read the node name from the log and tunnel the port to your machine:
#
#     ssh -N -L 7645:<node>:7645 <login-node>
#
# and point the app at localhost:7645 with the same token (Preferences ▸ HPC).
# See README.md next to this file.

set -euo pipefail

# --- environment (Fiona; adapt the modules to your cluster) -----------------
module load python
module load cmake 2>/dev/null || true
module load nvhpc 2>/dev/null || true
# A Python with numpy (and torch for segmentation models); the sirius wheel
# enables SIM reconstruction on the node. Uncomment for a venv/conda:
# source "$HOME/venvs/sirius/bin/activate"

# --- worker ------------------------------------------------------------------
SIRIUS_PORT="${SIRIUS_PORT:-7645}"
SIRIUS_DEVICE="${SIRIUS_DEVICE:-cuda}"
if [[ -z "${SIRIUS_TOKEN:-}" ]]; then
    echo "SIRIUS_TOKEN is not set; refusing to start an unauthenticated worker on a shared node" >&2
    exit 1
fi

# The worker package lives beside this script (app/python) in a checkout or a
# build tree; SIRIUS_WORKER_DIR overrides that.
WORKER_DIR="${SIRIUS_WORKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${WORKER_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "sirius worker on $(hostname) port ${SIRIUS_PORT} (device ${SIRIUS_DEVICE}), job ${SLURM_JOB_ID:-?}"
echo "tunnel:  ssh -N -L ${SIRIUS_PORT}:$(hostname):${SIRIUS_PORT} <login-node>"

# No --allow-install: on a shared node the worker must not run pip or conda
# on a client's request (see ../SECURITY.md). Install the model packages into
# the environment above when you submit the job, not from the application.
# The token is read from $SIRIUS_TOKEN (exported above), not passed as an
# argument: the command line of a job is visible to every user of the node.
exec python -m sirius_worker --host 0.0.0.0 --port "${SIRIUS_PORT}" --device "${SIRIUS_DEVICE}"
