#!/bin/bash
#SBATCH --job-name=gala_kinematic
#SBATCH --output=/checkpoint/%u/jobs/gala_kinematic_job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/gala_kinematic_job.%j.err
#SBATCH --gpus-per-task 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=devlab
#SBATCH --time=20
#SBATCH --open-mode=append
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR
set -x
srun bash slurm_training_task.sh
