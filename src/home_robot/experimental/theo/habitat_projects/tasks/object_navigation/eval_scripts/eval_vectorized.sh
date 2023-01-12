#!/bin/bash

#SBATCH --partition=learnfair
#SBATCH --job-name=eval
#SBATCH --output=slurm_logs/eval-%j.out
#SBATCH --error=slurm_logs/eval-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --constraint=volta32gb

python home_robot/experimental/theo/habitat_projects/tasks/object_navigation/eval_scripts/eval_vectorized.py "$@"