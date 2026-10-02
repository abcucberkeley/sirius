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
# time and resources on the sbatch command line). By hand, submit with a
# shared secret:
#
#     SIRIUS_TOKEN=$(openssl rand -hex 16) sbatch app/python/slurm/sirius_worker.sbatch
#
# The partition, account and QoS above are fiona's test GPUs (abc_debug: one
# hour at most); give your own on the command line elsewhere, which overrides
# them: sbatch --partition=... --account=... --qos=... --time=...
#
# then read the node name from the log and tunnel the port to your machine:
#
#     ssh -N -L 7645:<node>:7645 <login-node>
#
# and point the app at localhost:7645 with the same token (Preferences ▸
# Compute, HPC worker; or sirius-cli --hpc localhost:7645 with the token in
# SIRIUS_HPC_TOKEN). See README.md next to this file.

set -euo pipefail

# --- environment (Fiona; adapt the modules to your cluster) -----------------
module load python
module load cmake 2>/dev/null || true
module load nvhpc 2>/dev/null || true
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
if [[ -n "${SIRIUS_VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${SIRIUS_VENV}/bin/activate"
fi

# --- worker ------------------------------------------------------------------
SIRIUS_PORT="${SIRIUS_PORT:-7645}"
SIRIUS_DEVICE="${SIRIUS_DEVICE:-cuda}"
if [[ -z "${SIRIUS_TOKEN:-}" ]]; then
    echo "SIRIUS_TOKEN is not set; refusing to start an unauthenticated worker on a shared node" >&2
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

echo "sirius worker on $(hostname) port ${SIRIUS_PORT} (device ${SIRIUS_DEVICE}), job ${SLURM_JOB_ID:-?}"
echo "tunnel:  ssh -N -L ${SIRIUS_PORT}:$(hostname):${SIRIUS_PORT} <login-node>   (fiona: fiona.abc.berkeley.edu)"

# No --allow-install: on a shared node the worker must not run pip or conda
# on a client's request (see ../SECURITY.md). Install the model packages into
# the environment above when you submit the job, not from the application.
# The token is read from $SIRIUS_TOKEN (exported above), not passed as an
# argument: the command line of a job is visible to every user of the node.
cd "${WORKER_DIR}"
# --max-clients: the application keeps a connection for its status and one for
# the dataset it shows (read here, on the node) besides each run's.
exec python -m sirius_worker --host 0.0.0.0 --port "${SIRIUS_PORT}" --device "${SIRIUS_DEVICE}" --max-clients "${SIRIUS_MAX_CLIENTS:-8}"
