#!/bin/bash 
#
#SBATCH -p Odyssey_GPU
#SBATCH --qos=high
#SBATCH --job-name=TS_glo
#SBATCH --gres=gpu:a100
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elliot.chevet@imt-atlantique.fr

# load module dependencies
# module load petsc4py mpi4py h5py ...


export OMP_NUM_THREADS=1
export VEROS_ASSET_DIR=/Odyssey/private/e25cheve/veros/veros_assets
export HOME=/Odyssey/private/e25cheve/
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export TF_GPU_ALLOCATOR=cuda_malloc_async

source /Odyssey/private/e25cheve/miniconda3/etc/profile.d/conda.sh
conda activate /Odyssey/private/e25cheve/miniconda3/envs/veros

srun python train_launcher.py
