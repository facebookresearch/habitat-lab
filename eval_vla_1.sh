#!/usr/bin/env bash

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

/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 0 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 1 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 2 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 3 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 4 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 5 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 6 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 7 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 8 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 9 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 10 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 11 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 12 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 13 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 14 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 15 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 16 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 17 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 18 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 19 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 20 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 21 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 22 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 23 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 24 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 25 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 26 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 27 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 28 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 29 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 30 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 31 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 32 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 33 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 34 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 35 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 36 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 37 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 38 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 39 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 40 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 41 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 42 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 43 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 44 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 45 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 46 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 47 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 48 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 49 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 50 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 51 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 52 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 53 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 54 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 55 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 56 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 57 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 58 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 59 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 60 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 61 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 62 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 63 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 64 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 65 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 66 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 67 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 68 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 69 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 70 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 71 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 72 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 73 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 74 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 75 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 76 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 77 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 78 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 79 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 80 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 81 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 82 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 83 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 84 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 85 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 86 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 87 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 88 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 89 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 90 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 91 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 92 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 93 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 94 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 95 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 96 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 97 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 98 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1
/data/home/jtruong/miniconda3/envs/vla-physics-hab/bin/python eval_vla_isaac_spot.py --ep-id 99 --exp heuristic_expert_pick_joints_base_vel_fremont_livingroom_physics_no_retract --ckpt $1