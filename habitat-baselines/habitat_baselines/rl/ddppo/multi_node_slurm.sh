#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=logs.ddppo.out
#SBATCH --error=logs.ddppo.err
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=60GB
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=dev

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

set -x
srun python -u -m habitat_baselines.run \
    --exp-config habitat-baselines/habitat_baselines/config/pointnav/ddppo_pointnav.yaml \
    --run-type train
