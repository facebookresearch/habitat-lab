baselines
==============================
### Installation

The `habitat_baselines` sub-package is NOT included upon installation by default. To install `habitat_baselines`, use the following command instead:
```bash
pip install -e habitat-lab
pip install -e habitat-baselines
```
This will also install additional requirements for each sub-module in `habitat_baselines/`, which are specified in `requirements.txt` files located in the sub-module directory.


### Reinforcement Learning (RL)

**Proximal Policy Optimization (PPO)**

**paper**: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

**code**: The PPO implementation is based on
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

**dependencies**: A recent version of pytorch, for installing refer to [pytorch.org](https://pytorch.org/)

For training on sample data please follow steps in the repository README. You should download the sample [test scene data](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip), extract it under the main repo (`habitat-lab/`, extraction will create a data folder at `habitat-lab/data`) and run the below training command.

**train**:
```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml
```

You can reduce training time by changing the trainer from the default implement to [VER](rl/ver/README.md) by
setting `trainer_name` to `"ver"` in either the config or via the command line.

```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml \
  habitat_baselines.trainer_name=ver
```

**test**:
```bash
python -u -m habitat_baselines.run \
  --config-name=pointnav/ppo_pointnav_example.yaml \
  habitat_baselines.evaluate=True
```

We also provide trained RGB, RGBD, and Depth PPO  models for MatterPort3D and Gibson.
To use them download pre-trained pytorch models from [link](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/habitat_baselines_v2.zip) and unzip and specify model path [here](agents/ppo_agents.py#L151).

The `habitat_baselines/config/pointnav/ppo_pointnav.yaml` config has better hyperparameters for large scale training and loads the [Gibson PointGoal Navigation Dataset](/README.md#datasets) instead of the test scenes.
Change the `/benchmark/nav/pointnav: pointnav_gibson` in `habitat_baselines/config/pointnav/ppo_pointnav.yaml` to `/benchmark/nav/pointnav: pointnav_mp3d` in the defaults list for training on [MatterPort3D PointGoal Navigation Dataset](/README.md#datasets).

### Hierarchical Reinforcement Learning (HRL)

We provide a two-layer hierarchical policy class, consisting of a low-level skill that moves the robot, and a high-level policy that reasons about which low-level skill to use in the current state. This can be especially powerful in long-horizon mobile manipulation tasks, like those introduced in [Habitat2.0](https://arxiv.org/abs/2106.14405). Both the low- and high- level can be either learned or an oracle. For oracle high-level we use [PDDL](https://planning.wiki/guide/whatis/pddl), and for oracle low-level we use instantaneous transitions, with the environment set to the final desired state. Additionally, for navigation, we provide an oracle navigation skill that uses A-star and the map of the environment to move the robot to its goal.

To run the following examples, you need the [ReplicaCAD dataset](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replicacad).

To train a high-level policy, while using pre-learned low-level skills (SRL baseline from [Habitat2.0](https://arxiv.org/abs/2106.14405)), you can run:

```bash
python -u -m habitat_baselines.run \
  --config-name=rearrange/rl_hierarchical.yaml
```
To run a rearrangement episode with oracle low-level skills and a fixed task planner, run:

```bash
python -u -m habitat_baselines.run \
  --config-name=rearrange/rl_hierarchical.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines/rl/policy=hl_fixed \
  habitat_baselines/rl/policy/hierarchical_policy/defined_skills=oracle_skills
```

To change the task (like set table) that you train your skills on, you can change the line `/habitat/task/rearrange: rearrange_easy` to `/habitat/task/rearrange: set_table` in the defaults of your config.

# Habitat-3.0 Multi-Agent Training
First download the necessary data with `python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab3_bench_assets`.

## Social Navigation

In the social navigation task, a robot is tasked with finding and following a human. The goal is to train a neural network policy that takes the input of (1) Spot's arm depth image, (2) the humanoid detector sensor, and (3) Spot's depth stereo cameras, and outputs the linear and angular velocities.

### Observation
The observation of the social nav policy is defined under `habitat.gym.obs_keys` with the prefix of `agent_0` in `habitat-lab/habitat/config/benchmark/multi_agent/hssd_spot_human_social_nav.yaml`. In this yaml, `agent_0_articulated_agent_arm_depth` is the robot's arm depth camera, and `agent_0_humanoid_detector_sensor` is a humanoid detector that returns either a human's segmentation or bounding box given an arm RGB camera. For `humanoid_detector_sensor`, please see `HumanoidDetectorSensorConfig` in `habitat-lab/habitat/config/default_structured_configs.py` to learn more about how to configure the sensor (e.g., do you want the return to be bounding box or segmentation). Finally, `agent_0_spot_head_stereo_depth_sensor` is a Spot's body stereo depth image.

Note that if you want to add more or use other observation sensors, you can do that by adding sensors into `habitat.gym.obs_keys`. For example, you can provide a humanoid GPS to a policy's input by adding `agent_0_goal_to_agent_gps_compass` into `habitat.gym.obs_keys` in `hssd_spot_human_social_nav.yaml`. Notice that the observation key in `habitat.gym.obs_keys` must be a subset of sensors in `/habitat/task/lab_sensors`. Finally, another example would be adding an arm RGB sensor. You can do that by adding `agent_0_articulated_agent_arm_rgb` into `habitat.gym.obs_keys` in `hssd_spot_human_social_nav.yaml`.

For more advanced users, if you want to create a new sensor for social nav agents, there are three steps.
- Step 1. Define a new sensor config class in `habitat.config.default_structured_configs.py`.
- Step 2. Based on `type` string you define in `habitat.config.default_structured_configs.py`, create the same sensor name in sensor file using `@registry.register_sensor` method. See examples in `habitat.tasks.rearrange.social_nav.social_nav_sensors.py`.
- Step 3. Register the new sensor in `hssd_spot_human_social_nav.yaml` for using it. It should be defined in `/habitat/task/lab_sensors` in config yaml, and in `habitat.gym.obs_keys` using `agent_0_{your_sensor_name}`.

### Action
The action space of the social nav policy is defined under `/habitat/task/actions@habitat.task.actions.agent_0_base_velocity: base_velocity_non_cylinder` in `habitat-lab/habitat/config/benchmark/multi_agent/hssd_spot_human_social_nav.yaml`. The action consists of linear and angular velocities. You can learn more about the hyperparameters for this action under `BaseVelocityNonCylinderActionConfig` in `habitat-lab/habitat/config/default_structured_configs.py`.

### Reward
The reward function of the social nav policy is defined in `social_nav_reward`. It encourages the robot to find the human as soon as possible while maintaining a safe distance from the human after finding the human. You can learn more about the hyperparameters for this reward function under `SocialNavReward` in `habitat-lab/habitat/config/default_structured_configs.py`.

### Command
We have released a [checkpoint](https://huggingface.co/datasets/ai-habitat/hab3_episodes/tree/main/checkpoint) based on the below command. To reproduce this, run multi-agent training with a Spot robot's policy being a low-level navigation policy and a humanoid's policy being a fixed (non-trainable) policy that navigates a sequence of navigation targets (please make sure the `tensorboard_dir`, `video_dir`, `checkpoint_folder`, `eval_ckpt_path_dir` are the paths you want):

```bash
python -u -m habitat_baselines.run \
    --config-name=social_nav/social_nav.yaml \
    benchmark/multi_agent=hssd_spot_human_social_nav \
    habitat_baselines.evaluate=False \
    habitat_baselines.num_checkpoints=5000 \
    habitat_baselines.total_num_steps=1.0e9 \
    habitat_baselines.num_environments=24 \
    habitat_baselines.tensorboard_dir=tb_social_nav \
    habitat_baselines.video_dir=video_social_nav \
    habitat_baselines.checkpoint_folder=checkpoints_social_nav \
    habitat_baselines.eval_ckpt_path_dir=checkpoints_social_nav \
    habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.ang_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True \
    habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.ang_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.ang_speed=10.0 \
    habitat.task.measurements.social_nav_reward.facing_human_reward=3.0 \
    habitat.task.measurements.social_nav_reward.count_coll_pen=0.01 \
    habitat.task.measurements.social_nav_reward.max_count_colls=-1 \
    habitat.task.measurements.social_nav_reward.count_coll_end_pen=5 \
    habitat.task.measurements.social_nav_reward.use_geo_distance=True \
    habitat.task.measurements.social_nav_reward.facing_human_dis=3.0 \
    habitat.task.measurements.social_nav_seek_success.following_step_succ_threshold=400 \
    habitat.task.measurements.social_nav_seek_success.need_to_face_human=True \
    habitat.task.measurements.social_nav_seek_success.use_geo_distance=True \
    habitat.task.measurements.social_nav_seek_success.facing_threshold=0.5 \
    habitat.task.lab_sensors.humanoid_detector_sensor.return_image=True \
    habitat.task.lab_sensors.humanoid_detector_sensor.is_return_image_bbox=True \
    habitat.task.success_reward=10.0 \
    habitat.task.end_on_success=True \
    habitat.task.slack_reward=-0.1 \
    habitat.environment.max_episode_steps=1500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat.simulator.agents.agent_0.joint_start_noise=0.0
```

It is expected to observe the following reward training (learning) curve:
![Social Nav Reward Training Curve](/res/img/habitat3_social_nav_training_reward.png) In addition, under the following slurm job batch script setting:
```bash
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 10
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu=6GB
```
we have the following training wall clock time versus reward:
![Social Nav Reward Training Curve versus Time](/res/img/habitat3_social_nav_training_reward_time.png)

We have the following training FPS:
![Social Nav Training FPS](/res/img/habitat3_social_nav_training_fps.png)
Note the training FPS depends on multiple factors such as the number of GPUs and the number of environments.

For evaluating the trained Spot robot's policy based on 500 episodes, run (please make sure `video_dir` and `eval_ckpt_path_dir` are the paths you want and the checkpoint is there):

```bash
python -u -m habitat_baselines.run \
    --config-name=social_nav/social_nav.yaml \
    benchmark/multi_agent=hssd_spot_human_social_nav \
    habitat_baselines.evaluate=True \
    habitat_baselines.num_checkpoints=5000 \
    habitat_baselines.total_num_steps=1.0e9 \
    habitat_baselines.num_environments=12 \
    habitat_baselines.video_dir=video_social_nav \
    habitat_baselines.checkpoint_folder=checkpoints_social_nav \
    habitat_baselines.eval_ckpt_path_dir=checkpoints_social_nav/social_nav_latest.pth \
    habitat.task.actions.agent_0_base_velocity.longitudinal_lin_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.ang_speed=10.0 \
    habitat.task.actions.agent_0_base_velocity.allow_dyn_slide=True \
    habitat.task.actions.agent_0_base_velocity.enable_rotation_check_for_dyn_slide=False \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.human_stop_and_walk_to_robot_distance_threshold=-1.0 \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_randcoord_action.ang_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.lin_speed=10.0 \
    habitat.task.actions.agent_1_oracle_nav_action.ang_speed=10.0 \
    habitat.task.measurements.social_nav_reward.facing_human_reward=3.0 \
    habitat.task.measurements.social_nav_reward.count_coll_pen=0.01 \
    habitat.task.measurements.social_nav_reward.max_count_colls=-1 \
    habitat.task.measurements.social_nav_reward.count_coll_end_pen=5 \
    habitat.task.measurements.social_nav_reward.use_geo_distance=True \
    habitat.task.measurements.social_nav_reward.facing_human_dis=3.0 \
    habitat.task.measurements.social_nav_seek_success.following_step_succ_threshold=400 \
    habitat.task.measurements.social_nav_seek_success.need_to_face_human=True \
    habitat.task.measurements.social_nav_seek_success.use_geo_distance=True \
    habitat.task.measurements.social_nav_seek_success.facing_threshold=0.5 \
    habitat.task.lab_sensors.humanoid_detector_sensor.return_image=True \
    habitat.task.lab_sensors.humanoid_detector_sensor.is_return_image_bbox=True \
    habitat.task.success_reward=10.0 \
    habitat.task.end_on_success=False \
    habitat.task.slack_reward=-0.1 \
    habitat.environment.max_episode_steps=1500 \
    habitat.simulator.kinematic_mode=True \
    habitat.simulator.ac_freq_ratio=4 \
    habitat.simulator.ctrl_freq=120 \
    habitat.simulator.agents.agent_0.joint_start_noise=0.0 \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.test_episode_count=500 \
    habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.height=1080 \
    habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor.width=1920
```

The evaluation is expected to produce values similar to those below:

```bash
Average episode social_nav_reward: 1.8821
Average episode social_nav_stats.has_found_human: 0.9020
Average episode social_nav_stats.found_human_rate_after_encounter_over_epi: 0.6423
Average episode social_nav_stats.found_human_rate_over_epi: 0.4275
Average episode social_nav_stats.first_encounter_steps: 376.0420
Average episode social_nav_stats.follow_human_steps_after_first_encounter: 398.6340
Average episode social_nav_stats.avg_robot_to_human_after_encounter_dis_over_epi: 1.4969
Average episode social_nav_stats.avg_robot_to_human_dis_over_epi: 3.6885
Average episode social_nav_stats.backup_ratio: 0.1889
Average episode social_nav_stats.yield_ratio: 0.0192
Average episode num_agents_collide: 0.7020
```

Note that in Habitat-3.0 paper, we report our numbers in the full evaluation dataset (1200 episodes). As a result, the number could be a bit different than the ones in the paper.

## Social Rearrangement

To run multi-agent training with a Spot robot and humanoid on the social rearrangement task.
- Learn-Single: `python habitat_baselines/run.py --config-name=social_rearrange/pop_play.yaml`
- Learn-Pop with 8 humanoid policies during training: `python habitat_baselines/run.py --config-name=social_rearrange/pop_play.yaml habitat_baselines.rl.agent.num_pool_agents_per_type=[1,8]`
- Plan-Pop-4: `python habitat_baselines/run.py --config-name=social_rearrange/plan_pop.yaml habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=4`. To run Plan-Pop-p for other `p` values, set `habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx`.

For zero-shot evaluate against the unseen agent population:
- With planner-based collaborators: `python habitat_baselines/run.py --config-name=social_rearrange/pop_play.yaml habitat_baselines.evaluate=True habitat_baselines.eval_ckpt_path_dir=PATH_TO_CKPT.pth +habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.select_random_goal=False +habitat_baselines.rl.policy.agent_1.hierarchical_policy.high_level_policy.plan_idx=1` change `plan_idx` to be 1, 2, 3, or 4 to evaluate against the other 4 planner agents.
