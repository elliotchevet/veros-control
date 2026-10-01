#!/bin/bash
#
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=endpoint81
#SBATCH --gres=gpu:h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elliot.chevet@imt-atlantique.fr

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

export TRAIN_OUTPUT_DIR="$PWD/basin_endpoint81_training/job_${SLURM_JOB_ID}"
mkdir -p "$TRAIN_OUTPUT_DIR"/{tmp,cache,cuda_cache,matplotlib}
export TMPDIR="$TRAIN_OUTPUT_DIR/tmp"
export XDG_CACHE_HOME="$TRAIN_OUTPUT_DIR/cache"
export CUDA_CACHE_PATH="$TRAIN_OUTPUT_DIR/cuda_cache"
export MPLCONFIGDIR="$TRAIN_OUTPUT_DIR/matplotlib"
cp -- "$0" "$TRAIN_OUTPUT_DIR/submitted_job.slurm"
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
  --fresh-training \
  --total-updates 80 \
  --run-mode train \
  --control-mode nn_param \
  --method-pair vjp-only \
  --sample-steps 81 \
  --glorys-file '/Odyssey/private/e25cheve/veros/veros_assets/global_1deg_realistic/glorys12_initial_conditions_40_*.nc' \
  --glorys-index 0 \
  --glorys-month-index 0 \
  --glorys-start-month 1 \
  --loss-mode glorys_target \
  --glorys-linear-interpolation \
  --loss-reduction mean \
  --keys-loss temp salt \
  --momentum-epsilon 1e-7 \
  --tracer-epsilon 1e-6 \
  --train-n-steps 81 \
  --train-updates 80 \
  --train-lr 5e-3 \
  --nn-checkpoint /Odyssey/private/e25cheve/control/veros_control/scripts/basin_endpoint_training/job_55286/train.pkl \
  --use-lon-lat \
  --use-forcings \
  --balance-loss-terms \
  --bound-corrections \
  --validate-candidates \
  --basin-mean-loss \
  --basin-levels 5 \
  --basin-pattern-lambda 1 \
  --correction-depth-taper 5 5 \
  --diskless-mode \
  --output-folder "$TRAIN_OUTPUT_DIR" \
  --train-output-file "$TRAIN_OUTPUT_DIR/train.pkl"
