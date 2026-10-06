#!/bin/bash
#SBATCH --job-name=veros-evaluate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
# Submit from the repository root; add your site's partition/QOS on sbatch.
CONFIG="${1:?Pass the configuration JSON path}"
export VEROS_BACKEND=jax VEROS_DEVICE=gpu VEROS_LINEAR_SOLVER=scipy_jax
export VEROS_DISKLESS_MODE=true JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN_CACHE="$PWD/outputs/cache/${SLURM_JOB_ID}"
mkdir -p "$RUN_CACHE"/{tmp,cache,cuda,matplotlib}
export TMPDIR="$RUN_CACHE/tmp" XDG_CACHE_HOME="$RUN_CACHE/cache"
export CUDA_CACHE_PATH="$RUN_CACHE/cuda" MPLCONFIGDIR="$RUN_CACHE/matplotlib"
python -m veros_control.evaluation.infer "$CONFIG"
