#!/bin/bash

base_dir="/checkpoint/akshararai/hab3"
# shellcheck disable=SC2162
read -p "Enter sweep directory name (example: pop_play/2023-04-17/16-19-04):" SWEEP_SUBDIR
SWEEP_DIR="$base_dir/$SWEEP_SUBDIR"
echo "$SWEEP_DIR"
#SWEEP_DIR="/checkpoint/akshararai/hab3/pop_play/2023-04-17/16-19-04/"
echo "Sweep directories:"
for dir in "$SWEEP_DIR"/*/; do
	subdir=$(basename "$dir")
    echo "$subdir"
    wandb_name="eval_$SWEEP_SUBDIR"
    python habitat-baselines/habitat_baselines/run.py -m --config-name config.yaml --config-path "${dir}"/.hydra/ habitat_baselines.eval_ckpt_path_dir="${dir}"/checkpoints/latest.pth habitat_baselines.video_dir="${dir}"/video habitat_baselines.wb.run_name="${wandb_name}" habitat_baselines.evaluate=True habitat_baselines.test_episode_count=20 habitat_baselines.num_environments=16 habitat_baselines.eval.should_load_ckpt=True
done

python examples/siro_sandbox/sandbox_app.py \
--disable-inverse-kinematics \
--gui-controlled-agent-index 1 \
--never-end \
--save-filepath-base my_session \
--episodes-filter "0:5" \
--cfg experiments_hab3/pop_play_kinematic_oracle_humanoid_spot.yaml \
--cfg-opts \
habitat.environment.iterator_options.shuffle=False \
habitat_baselines.evaluate=True \
--cfg-opts \
habitat_baselines.num_environments=1 \
habitat.task.measurements.cooperate_subgoal_reward.end_on_collide=False \
habitat.simulator.habitat_sim_v0.allow_sliding=True \
habitat.task.actions.agent_1_oracle_nav_action.lin_speed=0.0 \
habitat.task.actions.agent_0_oracle_nav_action.lin_speed=0.0 \
habitat.task.actions.agent_0_oracle_nav_action.ang_speed=20.0


