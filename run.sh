CUDA_VISIBLE_DEVICES=1 python examples/siro_sandbox/sandbox_app.py \
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

python habitat-lab/habitat/utils/convert_smplx_poses.py
python habitat-lab/habitat/utils/convert_ems_poses.py
