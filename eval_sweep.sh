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
    python habitat-baselines/habitat_baselines/run.py -m --config-name config.yaml --config-path "${dir}"/.hydra/ habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=True habitat_baselines.video_dir="${dir}"/video_end_on_collide_True habitat_baselines.writer_type=tb habitat_baselines.evaluate=True habitat_baselines.test_episode_count=25 habitat_baselines.num_environments=16 habitat_baselines.eval.should_load_ckpt=True
done
