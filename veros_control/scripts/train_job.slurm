#!/bin/bash 
#
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=control
#SBATCH --gres=gpu:h100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elliot.chevet@imt-atlantique.fr

# load module dependencies
# module load petsc4py mpi4py h5py ...


export OMP_NUM_THREADS=1
export VEROS_ASSET_DIR=/Odyssey/private/e25cheve/veros/veros_assets
export HOME=/Odyssey/private/e25cheve/
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_ENABLE_X64=1

source /Odyssey/private/e25cheve/miniconda3/etc/profile.d/conda.sh
conda activate /Odyssey/private/e25cheve/miniconda3/envs/veros

export GPU_LOG=/Odyssey/private/e25cheve/gpu_mem_${SLURM_JOB_ID}.log
export PYTHONUNBUFFERED=1

(
  while true; do
    date
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
    echo "---"
    sleep 0.5
  done
) > "$GPU_LOG" 2>&1 &

python - <<'PY'
import jax
import jax.numpy as jnp

print("jax_enable_x64:", jax.config.jax_enable_x64)
print("float default:", jnp.asarray(1.0).dtype)
print("float64 requested:", jnp.asarray(1.0, dtype=jnp.float64).dtype)
PY

GPU_MONITOR_PID=$!
python ../training/ad_benchmark_root.py \
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
  --loss-reduction mean \
  --keys-loss temp salt \
  --epsilon 1e-6 \
  --train-n-steps 27 \
  --train-updates 20 \
  --train-lr 5e-3 \
  --feature-stats-file /Odyssey/private/e25cheve/control/veros_control/checkpoints/glorys_feature_stats_meanstd.pkl \
  --use-lon-lat \
  --use-forcings \
  --balance-loss-terms \
  --diskless-mode
#  --nn-checkpoint '/Odyssey/private/e25cheve/control/veros_control/scripts/output_folder_nn_param_glorys_target_mean/ad_comparison_nn_param_glorys_target_mean_param_l0_i0_vjp_only_curriculum_train_NVIDIA_A100-SXM4-80GB.pkl' \
#  --train-output-file '/Odyssey/private/e25cheve/control/veros_control/scripts/output_folder_nn_param_glorys_target_mean/ad_comparison_nn_param_glorys_target_mean_param_l0_i0_vjp_only_curriculum_train_NVIDIA_A100-SXM4-80GB.pkl'  
#srun python train_launcher.py \

kill "$GPU_MONITOR_PID"
