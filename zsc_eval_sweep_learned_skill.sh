#!/bin/bash


# plan_idx=-2



dirs="2 6 10"
plan_idxs="-4 -3 -2 -1"
zsc_ckpt_dir="/fsx-siro/akshararai/hab3/zsc_eval/zsc_ckpts"

learned_agents="ckpt.0.pth  ckpt.1.pth  ckpt.2.pth  ckpt.3.pth  ckpt.4.pth  ckpt.5.pth"


for dir in $dirs; do
    dir="/fsx-siro/akshararai/hab3/plan_play/2023-08-25/18-19-41/"$dir"/"
    subdir=$(basename "$dir")
    dir2="${dir/"/fsx-siro/akshararai/hab3"/"multirun/learned_skills_iclr3/zsc_pop_learned_skill_learned_nav"}"
    for plan_idx in $plan_idxs; do

        python habitat-baselines/habitat_baselines/rl/multi_agent/scripts/zsc_eval.py \
        --plan_idx "${plan_idx}" \
        habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
        habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
        habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largevalsubset_12s_200epi_2obj.json.gz \
        habitat_baselines.video_dir="${dir2}"/video_plan_"${plan_idx}" \
        habitat_baselines.episode_data_dir="${dir2}"/eval_data_plan_"${plan_idx}" \
        habitat_baselines.eval.save_summary_data=True \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
        habitat.environment.max_episode_steps=1500 \
        habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=False \
        habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
        habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
        hydra/launcher=aws_submitit_habitat  hydra.job.name='evalPOP_learned_plan'$plan_idx &
    done


    for learned_agent in $learned_agents; do
        ckpt_path="$zsc_ckpt_dir/$learned_agent"
        echo "evaluating agent $ckpt_path"
        wandb_name="zsc_eval_${SWEEP_SUBDIR}/${subdir}_${learned_agent}"
        echo "wandb_name: $wandb_name"
        learned_agent_path="$zsc_ckpt_dir/$learned_agent"
        # learned_agent_path=$learned_agent
        python habitat-baselines/habitat_baselines/rl/multi_agent/scripts/zsc_eval.py \
        --learned-agents "${learned_agent_path}" \
        habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
        habitat_baselines.writer_type=tb habitat_baselines.num_environments=12 \
        habitat_baselines.video_dir="${dir2}"/video_learn_"${learned_agent}" \
        habitat_baselines.episode_data_dir="${dir2}"/eval_data_learn_"${learned_agent}" \
        habitat_baselines.eval.save_summary_data=True \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.lin_speed=10.0 \
        habitat.task.actions.agent_1_oracle_nav_with_backing_up_action.ang_speed=10.0 \
        habitat.environment.max_episode_steps=1500 \
        habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
        habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] habitat_baselines.evaluate=True \
        habitat_baselines.eval.should_load_ckpt=True habitat_baselines.eval.evals_per_ep=3 \
        habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largevalsubset_12s_200epi_2obj.json.gz \
        hydra/launcher=aws_submitit_habitat hydra.job.name='evalPOPT_learned_learn'$learned_agent &
    done
done
