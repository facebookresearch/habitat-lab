#!/bin/bash

base_dir="multirun"
# shellcheck disable=SC2162
# read -p "Enter sweep directory name (example: pop_play/2023-04-17/16-19-04):" SWEEP_SUBDIR

export SWEEP_SUBDIR="2023-06-22/00-34-44"
SWEEP_DIR="$base_dir/$SWEEP_SUBDIR"
echo "$SWEEP_DIR"
#SWEEP_DIR="/checkpoint/akshararai/hab3/pop_play/2023-04-17/16-19-04/"
echo "Sweep directories:"
# for dir in "$SWEEP_DIR"/*/; do
# 	subdir=$(basename "$dir")
#     echo "$subdir"
#     wandb_name="eval_$SWEEP_SUBDIR"
#     # python create_
#     python habitat-baselines/habitat_baselines/run.py \
#         -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi.yaml \
#         habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
#         habitat_baselines.video_dir="${dir}"/video \
#         habitat_baselines.wb.run_name="${wandb_name}" \
#         habitat.environment.max_episode_steps=500 \
#         habitat_baselines.writer_type=tb \
#         habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
#         habitat_baselines.rl.policy.agent_0.hierarchical_policy.defined_skills.nav_to_obj.apply_postconds=False \
#         habitat_baselines.rl.policy.agent_1.hierarchical_policy.defined_skills.nav_to_obj.apply_postconds=False \
#         habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
#         habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
#         habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
#         habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \
#         hydra/launcher=aws_submitit_habitat_eval


# done

datasets=("microtrain_eval_small_1scene_2objs_30epi_45degree" "microtrain_eval_medium_1scene_2objs_30epi_45degree" "microtrain_eval_large_1scene_2objs_30epi_45degree")
for dataset in "${datasets[@]}"; do

    listruns=(0 1 2)
    for runid in "${listruns[@]}"; do
        subdir=$(basename "$dir")
        echo "$subdir"
        seed_id=$((101+$runid))
        wandb_name="eval_planner"
        dir="$base_dir/planner_robot_eval_2/$runid"
        echo "${dir}"
        python habitat-baselines/habitat_baselines/run.py \
            -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi_planner_planner.yaml \
            habitat_baselines.video_dir="${dir}"/video_data_train_${dataset}_pop \
            habitat_baselines.episode_data_dir="${dir}"/episode_data_${dataset}_pop \
            habitat_baselines.wb.run_name="${wandb_name}" \
            habitat.environment.max_episode_steps=750 \
            habitat_baselines.writer_type=wb \
            habitat.seed=$seed_id \
            habitat_baselines.num_environments=1 \
            habitat_baselines.eval.should_load_ckpt=False \
            habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
            habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
            habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
            habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
            habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/"${dataset}.json.gz" \
            hydra/launcher=aws_submitit_habitat_eval &

    done
done
