#!/bin/bash

# base_dir="/checkpoint/akshararai/hab3"
base_dir="/fsx-siro/akshararai/hab3"
# shellcheck disable=SC2162
read -p "Enter sweep directory name (example: pop_play/2023-04-17/16-19-04):" SWEEP_SUBDIR
SWEEP_DIR="$base_dir/$SWEEP_SUBDIR"
echo "$SWEEP_DIR"
#SWEEP_DIR="/checkpoint/akshararai/hab3/pop_play/2023-04-17/16-19-04/"
echo "Sweep directories:"
for dir in "$SWEEP_DIR"/*/; do
	subdir=$(basename "$dir")
    echo "$subdir"
    # skip_dirs="0 1 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"

    # # Check if the current directory should be skipped
    # skip=false
    # for skip_dir in $skip_dirs; do
    #     if [ "$subdir" = "$skip_dir" ]; then
    #         skip=true
    #         echo "skipping $subdir"
    #         break
    #     fi
    # done

    # # Skip the directory if needed
    # if [ "$skip" = true ]; then
    #     continue
    # fi

    wandb_name="$SWEEP_SUBDIR_$subdir"
    # python habitat-baselines/habitat_baselines/run.py \
    # -m --config-name experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp.yaml \
    # habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    # habitat_baselines.video_dir="${dir}"/video \
    # habitat_baselines.evaluate=True \
    # habitat_baselines.num_environments=12 \
    # habitat_baselines.eval.should_load_ckpt=True \
    # habitat_baselines.writer_type=tb \
    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
    # habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1] \
    # habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
    # habitat.task.measurements.cooperate_subgoal_reward.collide_penalty=2.0 \
    # habitat.task.slack_reward=-0.001 \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='eval_GTCoord' &

    # python habitat-baselines/habitat_baselines/run.py \
    # -m --config-name experiments_hab3/pop_play_kinematic_oracle_humanoid_spot_fp.yaml \
    # habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    # habitat_baselines.video_dir="${dir}"/video \
    # habitat_baselines.evaluate=True \
    # habitat_baselines.num_environments=12 \
    # habitat_baselines.eval.should_load_ckpt=True \
    # habitat_baselines.writer_type=tb \
    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
    # habitat_baselines.rl.agent.num_pool_agents_per_type=[1,8] \
    # habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True \
    # habitat.task.measurements.cooperate_subgoal_reward.collide_penalty=2.0 \
    # habitat.task.slack_reward=-0.001 \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='eval_pp8' &

    # python habitat-baselines/habitat_baselines/run.py \
    # -m --config-name config.yaml --config-path "${dir}"/.hydra/ \
    # habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth \
    # habitat_baselines.video_dir="${dir}"/video \
    # habitat_baselines.wb.run_name="${wandb_name}" habitat_baselines.evaluate=True \
    # habitat_baselines.num_environments=12 \
    # habitat_baselines.test_episode_count=200 \
    # habitat_baselines.eval.should_load_ckpt=True \
    # +habitat_baselines.eval.save_summary_data=False \
    # habitat.dataset.data_path=data/datasets/floorplanner/rearrange/scratch/train/largeval_12s_1200epi_2obj.json.gz \
    # habitat_baselines.grouped_scenes=False habitat.simulator.force_soft_reset=False \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='eval_plan_play'

    python habitat-baselines/habitat_baselines/run.py \
    -m --config-name config.yaml --config-path "${dir}"/.hydra/ \
    habitat_baselines.checkpoint_folder="${dir}"/checkpoints \
    habitat_baselines.wb.run_name="${wandb_name}" habitat_baselines.load_resume_state_config=True \
    habitat_baselines.num_environments=24 habitat_baselines.writer_type=tb \
    # hydra/launcher=aws_submitit_habitat hydra/output=aws_path hydra.job.name='resume_plan_play'

done
