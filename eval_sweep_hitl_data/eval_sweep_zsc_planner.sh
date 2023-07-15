#!/bin/bash

base_dir="multirun"
# shellcheck disable=SC2162

export SWEEP_SUBDIR="2023-06-21/00-58-01"
# export SWEEP_SUBDIR="2023-06-22/00-34-44"


#read -p "Enter sweep directory name (example: pop_play/2023-04-17/16-19-04):" SWEEP_SUBDIR
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
    plan_idxes=(-1 -1 -1 -2 -2 -2 -3 -3 -3 -4 -4 -4)
    for dir in "$SWEEP_DIR"/*/; do
        subdir=$(basename "$dir")
        #echo "$subdir"
        wandb_name="eval_$SWEEP_SUBDIR"
        run_id=$((subdir))
        curr_plan_idx=${plan_idxes[$run_id]}
        echo "${dir}"
        echo $curr_plan_idx
        # python habitat-baselines/habitat_baselines/run.py \
        #     -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi_planner.yaml \
        #     habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
        #     habitat_baselines.video_dir="${dir}"/video_train_pop \
        #     habitat_baselines.episode_data_dir="${dir}"/episode_data_train_pop \
        #     habitat_baselines.wb.run_name="${wandb_name}" \
        #     habitat.environment.max_episode_steps=750 \
        #     habitat_baselines.writer_type=tb \
        #     habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
        #     habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
        #     habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
        #     habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
        #     habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=$curr_plan_idx \
        #     habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
        #     habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \
        #     hydra/launcher=aws_submitit_habitat_eval &

        # echo data/datasets/floorplanner/rearrange/scratch/train/"${dataset}.json.gz"

        python habitat-baselines/habitat_baselines/run.py \
            -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi_planner.yaml \
            habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
            habitat_baselines.video_dir="${dir}"/video_val_${dataset}_pop \
            habitat_baselines.episode_data_dir="${dir}"/episode_data_val_${dataset}_pop \
            habitat_baselines.wb.run_name="${wandb_name}" \
            habitat.environment.max_episode_steps=750 \
            habitat_baselines.writer_type=tb \
            habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
            habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
            habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
            habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
            habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=-1 \
            habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
            habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/"${dataset}.json.gz" \
            hydra/launcher=aws_submitit_habitat_eval &
    done
done
    # python habitat-baselines/habitat_baselines/run.py \
    #     -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi_planner.yaml \
    #     habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    #     habitat_baselines.video_dir="${dir}"/video_train_pop \
    #     habitat_baselines.episode_data_dir="${dir}"/episode_data_train_pop \
    #     habitat_baselines.wb.run_name="${wandb_name}" \
    #     habitat.environment.max_episode_steps=750 \
    #     habitat_baselines.writer_type=tb \
    #     habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
    #     habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
    #     habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
    #     habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
    #     habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=$curr_plan_idx \
    #     habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
    #     habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \
    #     hydra/launcher=aws_submitit_habitat_eval &


# done



    # python habitat-baselines/habitat_baselines/run.py \
    #     -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi_planner.yaml \
    #     habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    #     habitat_baselines.video_dir="${dir}"/video_val_pop \
    #     habitat_baselines.episode_data_dir="${dir}"/episode_data_val_pop \
    #     habitat_baselines.wb.run_name="${wandb_name}" \
    #     habitat.environment.max_episode_steps=750 \
    #     habitat_baselines.writer_type=tb \
    #     habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
    #     habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
    #     habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
    #     habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
    #     habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
    #     habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \
    #     hydra/launcher=aws_submitit_habitat_eval &


    # MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet  python habitat-baselines/habitat_baselines/run.py \
    #     -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi.yaml \
    #     habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    #     habitat_baselines.video_dir="${dir}"/video_data_train_pop \
    #     habitat_baselines.video_dir="${dir}"/episode_data_train_pop \
    #     habitat_baselines.wb.run_name="${wandb_name}" \
    #     habitat.environment.max_episode_steps=750 \
    #     habitat_baselines.writer_type=wb \
    #     habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
    #     habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
    #     habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
    #     habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
    #     habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
    #     habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/microtrain_small_size_v0.3.2.json.gz


    # MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet HABITAT_ENV_DEBUG=1  python habitat-baselines/habitat_baselines/run.py \
    #     -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi.yaml \
    #     habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    #     habitat_baselines.video_dir="${dir}"/video_data_val_pop \
    #     habitat_baselines.video_dir="${dir}"/episode_data_val_pop \
    #     habitat_baselines.wb.run_name="${wandb_name}" \
    #     habitat.environment.max_episode_steps=750 \
    #     habitat_baselines.writer_type=tb \
    #     habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
    #     habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
    #     habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
    #     habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
    #     habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
    #     habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \


    # MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet HABITAT_ENV_DEBUG=1  python habitat-baselines/habitat_baselines/run.py \
    #     -m --config-name experiments_hab3/eval_zsc_kinematic_oracle_fp_xavi.yaml \
    #     habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    #     habitat_baselines.video_dir="${dir}"/video \
    #     habitat_baselines.video_dir="${dir}"/episode_data_train_pop \
    #     habitat_baselines.wb.run_name="${wandb_name}" \
    #     habitat.environment.max_episode_steps=750 \
    #     habitat_baselines.writer_type=tb \
    #     habitat_baselines.evaluate=True habitat_baselines.eval.save_summary_data=True \
    #     habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.agents_dist_thresh=1.5 \
    #     habitat.task.lab_sensors.agent_0_should_replan.x_len=-1.0 \
    #     habitat.task.lab_sensors.agent_1_should_replan.x_len=-1.0 \
    #     habitat_baselines.num_environments=1 habitat_baselines.eval.should_load_ckpt=True \
    #     habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \





datasets=("microtrain_eval_small_1scene_2objs_30epi_45degree" "microtrain_eval_medium_1scene_2objs_30epi_45degree" "microtrain_eval_large_1scene_2objs_30epi_45degree")
for dataset in "${datasets[@]}"; do

    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python habitat-baselines/habitat_baselines/run.py -m --config-name experiments_hab3/single_agent_kinematic_oracle_humanoid_fp.yaml \
    habitat.simulator.kinematic_mode=True habitat.simulator.step_physics=False \
    habitat.seed=101 habitat_baselines.eval.save_summary_data=True habitat_baselines.writer_type=tb  \
    habitat_baselines.evaluate=True \
    habitat_baselines.num_environments=1 \
    habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_5s_500epi_2obj.json.gz \
    habitat.task.actions.oracle_nav_action.lin_speed=100 \
    habitat.task.actions.oracle_nav_action.ang_speed=40 \
    habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.plan_idx=3 \
    habitat.task.task_spec=rearrange_easy_fp habitat.task.pddl_domain_def=fp
done
