#!/bin/bash

#SBATCH --job-name=heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics
#SBATCH --output=/fsx-siro/jtruong/repos/vla-physics/habitat-lab/videos_vla_eval/%x/logs/%x_%J.log
#SBATCH --error=/fsx-siro/jtruong/repos/vla-physics/habitat-lab/videos_vla_eval/%x/logs/%x_%J.err
#SBATCH --time=7200
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=500G
#SBATCH --account=siro
#SBATCH --qos=siro_high

export WANDB_DISABLED=1 ; export WANDB_MODE=offline
export HABITAT_REARRANGE_LOG=1 && export HABITAT_ENV_DEBUG=1
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export TRANSFORMERS_CACHE=/fsx-siro/jtruong/data/weights
export VLA_DATA_DIR=/fsx-siro/jtruong/data/vla_data
export VLA_LOG_DIR=/fsx-siro/jtruong/repos/robot-skills/results/vla_wb_log
export VLA_WANDB_ENTITY=joanne
export skill_vlm_dir=/fsx-siro/jtruong/repos/robot-skills
export hab_dir=/fsx-siro/jtruong/repos/habitat-lab

cd /fsx-siro/jtruong/repos/vla-physics/habitat-lab

for i in {0..99}; do
    /data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id $i --exp $1 --ckpt $2
done


# for i in $(seq 5000 5000 150000); do 
#     sbatch --job-name=heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract_30 eval_vla_sbatch.sh heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract_30 $i
# done