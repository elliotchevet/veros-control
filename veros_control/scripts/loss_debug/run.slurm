#!/bin/bash
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=nn-loss-debug
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/Odyssey/private/e25cheve/control/veros_control/scripts/loss_debug/slurm-%j.out
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
export LOSS_DEBUG_OUT="$PWD/loss_debug/job_${SLURM_JOB_ID}"
mkdir -p "$LOSS_DEBUG_OUT/tmp" "$LOSS_DEBUG_OUT/matplotlib"
export TMPDIR="$LOSS_DEBUG_OUT/tmp"
export MPLCONFIGDIR="$LOSS_DEBUG_OUT/matplotlib"
export JAX_LOG_COMPILES=0
/opt/slurm/bin/scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true
nvidia-smi
/Odyssey/private/e25cheve/miniconda3/envs/veros/bin/python -B loss_debug/check_rollouts.py \
  --run-mode train --control-mode nn_param --method-pair vjp-only \
  --sample-steps 81 \
  --glorys-file '/Odyssey/private/e25cheve/veros/veros_assets/global_1deg_realistic/glorys12_initial_conditions_40_*.nc' \
  --glorys-index 0 --glorys-month-index 0 --glorys-start-month 1 \
  --loss-mode glorys_target --glorys-linear-interpolation --loss-reduction mean \
  --keys-loss temp salt --epsilon 1e-6 --train-n-steps 81 --train-updates 1 --train-lr 5e-3 \
  --feature-stats-file /Odyssey/private/e25cheve/control/veros_control/checkpoints/glorys_feature_stats_meanstd.pkl \
  --use-lon-lat --use-forcings --train-overwrite --balance-loss-terms --diskless-mode \
  --output-folder "$LOSS_DEBUG_OUT" --train-output-file "$LOSS_DEBUG_OUT/train.pkl"
