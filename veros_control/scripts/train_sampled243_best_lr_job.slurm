#!/bin/bash
#
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=sampled243_best_lr
#SBATCH --output=/Odyssey/private/e25cheve/control/veros_control/scripts/slurm-%j.out
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail
cd /Odyssey/private/e25cheve/control/veros_control/scripts

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VEROS_ASSET_DIR=/Odyssey/private/e25cheve/veros/veros_assets
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export JAX_TRACEBACK_FILTERING=off

export TRAIN_OUTPUT_DIR="$PWD/basin_sampled243_alllevels_training/job_${SLURM_JOB_ID}"
mkdir -p "$TRAIN_OUTPUT_DIR"/{tmp,cache,cuda_cache,matplotlib}
export TMPDIR="$TRAIN_OUTPUT_DIR/tmp"
export XDG_CACHE_HOME="$TRAIN_OUTPUT_DIR/cache"
export CUDA_CACHE_PATH="$TRAIN_OUTPUT_DIR/cuda_cache"
export MPLCONFIGDIR="$TRAIN_OUTPUT_DIR/matplotlib"
cp -- "$0" "$TRAIN_OUTPUT_DIR/submitted_job.slurm"
cp ../setup/global_1deg_control/global_1deg_control.py ../setup/global_1deg_control/forcing_window.py "$TRAIN_OUTPUT_DIR/"
cp ../training/best_checkpoint.py ../training/ad_benchmark_root.py ../training/basin_loss.py ../training/stability.py split_epsilon/continue_training.py "$TRAIN_OUTPUT_DIR/"
export SOURCE_CHECKPOINT="$PWD/basin_sampled243_alllevels_training/job_58653/train.pkl"
sha256sum "$SOURCE_CHECKPOINT" > "$TRAIN_OUTPUT_DIR/source_checkpoint.sha256"
export GPU_LOG="$TRAIN_OUTPUT_DIR/gpu_memory.log"

(
  while true; do
    date
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
    sleep 5
  done
) > "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

/Odyssey/private/e25cheve/miniconda3/envs/veros/bin/python -B split_epsilon/continue_training.py \
  --total-updates 100 \
  --run-mode train \
  --control-mode nn_param \
  --method-pair vjp-only \
  --sample-steps 27 \
  --glorys-file '/Odyssey/private/e25cheve/veros/veros_assets/global_1deg_realistic/glorys12_initial_conditions_40_*.nc' \
  --glorys-index 0 \
  --glorys-month-index 0 \
  --glorys-start-month 1 \
  --loss-mode glorys_target \
  --glorys-linear-interpolation \
  --glorys-sampled-interpolation \
  --loss-reduction mean \
  --keys-loss temp salt \
  --momentum-epsilon 1e-7 \
  --tracer-epsilon 1e-6 \
  --train-n-steps 243 \
  --train-updates 20 \
  --train-lr 1e-3 \
  --train-keep-best \
  --nn-checkpoint "$SOURCE_CHECKPOINT" \
  --use-lon-lat \
  --use-forcings \
  --balance-loss-terms \
  --bound-corrections \
  --validate-candidates \
  --basin-mean-loss \
  --basin-levels 0 \
  --basin-pattern-lambda 1 \
  --diskless-mode \
  --output-folder "$TRAIN_OUTPUT_DIR" \
  --train-output-file "$TRAIN_OUTPUT_DIR/train.pkl"
