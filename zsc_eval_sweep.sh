#!/bin/bash

# base_dir="/checkpoint/akshararai/hab3"
base_dir="/fsx-siro/akshararai/hab3"
# shellcheck disable=SC2162
read -p "Enter checkpoint sweep directory name (example: pop_play/2023-04-17/16-19-04):" SWEEP_SUBDIR
SWEEP_DIR="$base_dir/$SWEEP_SUBDIR"
echo "$SWEEP_DIR"
#SWEEP_DIR="/checkpoint/akshararai/hab3/pop_play/2023-04-17/16-19-04/"
plan_idxs="-4 -3 -2 -1"
zsc_ckpt_dir="/fsx-siro/akshararai/hab3/zsc_eval/zsc_ckpts"
learned_agents="ckpt.0.pth  ckpt.1.pth  ckpt.2.pth  ckpt.3.pth  ckpt.4.pth  ckpt.5.pth"
# learned_agents="/fsx-siro/akshararai/hab3/GTCoord/2023-08-19/00-07-24/0/checkpoints/latest.pth"
zsc_data_dir="/fsx-siro/akshararai/hab3/zsc_eval/zsc_eval_data/speed_5/"$SWEEP_SUBDIR

dirs="8 9 10 11"
# for dir in "$SWEEP_DIR"/*/; do
for dir in $dirs; do
    eval_ckpt_path_dir="$SWEEP_DIR/$dir"
    # eval_ckpt_path_dir="$dir"
	subdir=$(basename "$dir")
    echo "evaluating checkpoint: $subdir"
    for plan_idx in $plan_idxs; do
        echo "evaluating plan $plan_idx"
        wandb_name="zsc_eval_${SWEEP_SUBDIR}/${subdir}_${plan_idx}"
        echo "wandb_name: $wandb_name"
        python habitat-baselines/habitat_baselines/rl/multi_agent/scripts/zsc_eval.py \
        --plan_idx "${plan_idx}" habitat_baselines.eval_ckpt_path_dir="${eval_ckpt_path_dir}"/checkpoints/latest.pth \
        habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
        habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
        habitat_baselines.video_dir="${zsc_data_dir}"/"${subdir}"/video_plan_"${plan_idx}" \
        habitat_baselines.episode_data_dir="${zsc_data_dir}"/"${subdir}"/eval_data_plan_"${plan_idx}" \
        habitat_baselines.eval.save_summary_data=True \
        habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.lin_speed=10.0 \
        habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.ang_speed=10.0 \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
        habitat.environment.max_episode_steps=1500 \
        habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
        habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
        habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
        hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='zsc_eval_plan_'$subdir$plan_idx &
    done
    for learned_agent in $learned_agents; do
        ckpt_path="$zsc_ckpt_dir/$learned_agent"
        echo "evaluating agent $ckpt_path"
        wandb_name="zsc_eval_${SWEEP_SUBDIR}/${subdir}_${learned_agent}"
        echo "wandb_name: $wandb_name"
        learned_agent_path="$zsc_ckpt_dir/$learned_agent"
        # learned_agent_path=$learned_agent
        python habitat-baselines/habitat_baselines/rl/multi_agent/scripts/zsc_eval.py \
        --learned-agents "${learned_agent_path}" habitat_baselines.eval_ckpt_path_dir="${eval_ckpt_path_dir}"/checkpoints/latest.pth \
        habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
        habitat_baselines.video_dir="${zsc_data_dir}"/"${subdir}"/video_"${learned_agent}" \
        habitat_baselines.episode_data_dir="${zsc_data_dir}"/"${subdir}"/eval_data_"${learned_agent}" \
        habitat_baselines.eval.save_summary_data=True \
        habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.lin_speed=10.0 \
        habitat.task.actions.agent_0_oracle_nav_with_backing_up_action.ang_speed=10.0 \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
        habitat.environment.max_episode_steps=1500 \
        habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
        habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
        habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
        habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
        hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='zsc_eval_'$subdir$learned_agent &
    done
done
