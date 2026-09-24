#!/bin/bash
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=nn-mem-debug
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/Odyssey/private/e25cheve/control/veros_control/scripts/memory_debug/slurm-%j.out
set -euo pipefail
cd /Odyssey/private/e25cheve/control/veros_control/scripts
export OMP_NUM_THREADS=1
export VEROS_ASSET_DIR=/Odyssey/private/e25cheve/veros/veros_assets
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_ENABLE_X64=1
export JAX_TRACEBACK_FILTERING=off
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MEMORY_DEBUG_OUT="$PWD/memory_debug/job_${SLURM_JOB_ID}"
mkdir -p "$MEMORY_DEBUG_OUT/tmp" "$MEMORY_DEBUG_OUT/matplotlib"
export TMPDIR="$MEMORY_DEBUG_OUT/tmp"
export MPLCONFIGDIR="$MEMORY_DEBUG_OUT/matplotlib"
export JAX_LOG_COMPILES=1
export MEMORY_CLEAR_AT="${MEMORY_CLEAR_AT:-3}"
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true
nvidia-smi
/Odyssey/private/e25cheve/miniconda3/envs/veros/bin/python -B memory_debug/profile_training.py \
  --run-mode train --control-mode nn_param --method-pair vjp-only \
  --sample-steps 27 \
  --glorys-file '/Odyssey/private/e25cheve/veros/veros_assets/global_1deg_realistic/glorys12_initial_conditions_40_*.nc' \
  --glorys-index 0 --glorys-month-index 0 --glorys-start-month 1 \
  --loss-mode glorys_target --glorys-linear-interpolation --loss-reduction mean \
  --keys-loss temp salt --epsilon 1e-6 --train-n-steps 27 --train-updates 4 --train-lr 5e-3 \
  --feature-stats-file /Odyssey/private/e25cheve/control/veros_control/checkpoints/glorys_feature_stats_meanstd.pkl \
  --use-lon-lat --use-forcings --train-overwrite --balance-loss-terms --diskless-mode \
  --output-folder "$MEMORY_DEBUG_OUT" --train-output-file "$MEMORY_DEBUG_OUT/train.pkl"
