#!/bin/bash
#SBATCH -A hpcteszt
#SBATCH --job-name=densenet-121
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

module load cray-python/3.11.7
module load cuda/12.9

export TORCH_HOME=/home/itamas/hpcteszt_scratch/itamas/COVID19-classification
export HF_HOME=/home/itamas/hpcteszt_scratch/itamas/COVID19-classification/

source ../.venv/bin/activate

srun python3 ./densenet_121.py