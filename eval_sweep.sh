#!/bin/bash

# base_dir="/checkpoint/akshararai/hab3"
base_dir="/fsx-siro/akshararai/hab3"
# shellcheck disable=SC2162
read -p "Enter sweep directory name (example: pop_play/2023-04-17/16-19-04):" SWEEP_SUBDIR
SWEEP_DIR="$base_dir/$SWEEP_SUBDIR"
echo "$SWEEP_DIR"
#SWEEP_DIR="/checkpoint/akshararai/hab3/pop_play/2023-04-17/16-19-04/"
echo "Sweep directories:"
plan_idxs="-4 -3 -2 -1"
for dir in "$SWEEP_DIR"/*/; do
	subdir=$(basename "$dir")
    # choose plan_idx based on directory name
    # plan_idx = -4 for 0, 4, 8
    # plan_idx = -3 for 1, 5, 9
    # plan_idx = -2 for 2, 6, 10
    if [ "$subdir" = "0" ] || [ "$subdir" = "4" ] || [ "$subdir" = "8" ]; then
        plan_idx=-4
    elif [ "$subdir" = "1" ] || [ "$subdir" = "5" ] || [ "$subdir" = "9" ]; then
        plan_idx=-3
    elif [ "$subdir" = "2" ] || [ "$subdir" = "6" ] || [ "$subdir" = "10" ]; then
        plan_idx=-2
    elif [ "$subdir" = "3" ] || [ "$subdir" = "7" ] || [ "$subdir" = "11" ]; then
        plan_idx=-1
    fi
    echo "$subdir"
    echo "plan_idx: $plan_idx"

    wandb_name="$SWEEP_SUBDIR_$subdir"
    # python habitat-baselines/habitat_baselines/run.py \
    # -m --config-name experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp.yaml \
    # habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    # habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
    # habitat_baselines.video_dir="${dir}"/video_multi_ep_speed_10 \
    # habitat_baselines.episode_data_dir="${dir}"/eval_data_multi_ep_speed_10 \
    # habitat_baselines.eval.save_summary_data=True \
    # habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.lin_speed=10.0 \
    # habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.ang_speed=10.0 \
    # habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
    # habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
    # habitat.environment.max_episode_steps=1500 \
    # habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
    # habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
    # habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
    # habitat.task.measurements.cooperate_subgoal_reward.collide_penalty=2.0 \
    # habitat.task.slack_reward=-0.001 \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='eval_GTCoord' &

    # python habitat-baselines/habitat_baselines/run.py \
    # -m --config-name experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp.yaml \
    # habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    # habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
    # habitat_baselines.video_dir="${dir}"/video_multi_ep_speed_10 \
    # habitat_baselines.episode_data_dir="${dir}"/eval_data_multi_ep_speed_10 \
    # habitat_baselines.eval.save_summary_data=True \
    # habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.lin_speed=10.0 \
    # habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.ang_speed=10.0 \
    # habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
    # habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
    # habitat.environment.max_episode_steps=1500 \
    # habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
    # habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
    # habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
    # habitat.task.measurements.cooperate_subgoal_reward.collide_penalty=2.0 \
    # habitat.task.slack_reward=-0.001 \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='eval_pp8' &

    python habitat-baselines/habitat_baselines/run.py \
    -m --config-name experiments_hab3/planner_learn_kinematic_oracle_humanoid_spot_fp.yaml \
    habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
    habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
    habitat_baselines.video_dir="${dir}"/video_multi_ep_speed_10 \
    habitat_baselines.episode_data_dir="${dir}"/eval_data_multi_ep_speed_10 \
    habitat_baselines.eval.save_summary_data=True \
    habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.lin_speed=10.0 \
    habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.ang_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
    habitat.environment.max_episode_steps=1500 \
    habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
    habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
    habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
    habitat.task.measurements.cooperate_subgoal_reward.collide_penalty=2.0 \
    habitat.task.slack_reward=-0.001 \
    habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx="${plan_idx}" \
    hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='eval_plan_play_'$plan_idx &

    # python habitat-baselines/habitat_baselines/run.py \
    # -m --config-name config.yaml --config-path "${dir}"/.hydra/ \
    # habitat_baselines.checkpoint_folder="${dir}"/checkpoints \
    # habitat_baselines.wb.run_name="${wandb_name}" habitat_baselines.load_resume_state_config=True \
    # habitat_baselines.num_environments=24 habitat_baselines.writer_type=tb \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='resume_plan_play'

done
