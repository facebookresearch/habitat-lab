#!/bin/bash
### Copyright (c) Meta Platforms, Inc. All Rights Reserved
#SBATCH --job-name=galactic
#SBATCH --output=***
#SBATCH --error=***
#SBATCH --gpus-per-task 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=***
#SBATCH --time=20
#SBATCH --open-mode=append
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR
set -x
srun bash slurm_training_task.sh
