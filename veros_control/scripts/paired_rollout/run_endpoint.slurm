#!/bin/bash
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=paired-lambda1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/Odyssey/private/e25cheve/control/veros_control/scripts/paired_rollout/slurm-%j.out
set -euo pipefail
cd /Odyssey/private/e25cheve/control/veros_control
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VEROS_ASSET_DIR=/Odyssey/private/e25cheve/veros/veros_assets
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export VEROS_CONTROL_DISKLESS_MODE=1
export REPLAY_OUT="$PWD/scripts/paired_rollout/job_${SLURM_JOB_ID}"
mkdir -p "$REPLAY_OUT"/{tmp,cache,cuda_cache,matplotlib}
export TMPDIR="$REPLAY_OUT/tmp"
export XDG_CACHE_HOME="$REPLAY_OUT/cache"
export CUDA_CACHE_PATH="$REPLAY_OUT/cuda_cache"
export MPLCONFIGDIR="$REPLAY_OUT/matplotlib"
PYTHON=/Odyssey/private/e25cheve/miniconda3/envs/veros/bin/python
"$PYTHON" -B scripts/paired_rollout/run.py --checkpoint "$PWD/scripts/basin_endpoint_training/job_55286/train.pkl" --output "$REPLAY_OUT"
"$PYTHON" -B scripts/paired_rollout/plot.py "$REPLAY_OUT"
"$PYTHON" -B scripts/paired_rollout/plot_surface_basin_improvement.py "$REPLAY_OUT"
