#!/bin/bash
#SBATCH -A hpcteszt
#SBATCH --job-name=alexnet-train-ddp
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=alexnet-train-ddp-1.out

module load cray-python/3.11.7
module load cuda/12.9

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

export TORCH_HOME=/home/itamas/hpcteszt_scratch/itamas/COVID19-classification
export HF_HOME=/home/itamas/hpcteszt_scratch/itamas/COVID19-classification/

source ../.venv/bin/activate

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    alexnet-ddp.py 