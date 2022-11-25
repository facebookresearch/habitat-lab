Habitat-Lab's configuration system has been changed from [YACS](https://github.com/rbgirshick/yacs) to [Hydra](https://hydra.cc).

<details>
<summary>For more details see Habitat-Lab's input PointNav benchmark config example.</summary>

```yaml
# @package _global_

defaults:
  - /habitat: habitat_config_base
  - /habitat/task: pointnav
  - /agent@habitat.simulator.agents.agent_0: agent_base
  - /habitat/simulator/sim_sensors:
    - rgb_sensor
  - /habitat/dataset/pointnav: gibson
  - _self_

habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agent_0:
      sim_sensors:
        rgb_sensor:
          width: 256
          height: 256
        depth_sensor:
          width: 256
          height: 256
```
</details>

<details>
<summary>Corresponding output config.</summary>

```yaml
habitat:
  seed: 100
  env_task: GymHabitatEnv
  env_task_gym_dependencies: []
  env_task_gym_id: ''
  environment:
    max_episode_steps: 500
    max_episode_seconds: 10000000
    iterator_options:
      cycle: true
      shuffle: true
      group_by_scene: true
      num_episode_sample: -1
      max_scene_repeat_episodes: -1
      max_scene_repeat_steps: 10000
      step_repetition_range: 0.2
  simulator:
    type: Sim-v0
    action_space_config: v0
    action_space_config_arguments: {}
    forward_step_size: 0.25
    create_renderer: false
    requires_textures: true
    lag_observations: 0
    auto_sleep: false
    step_physics: true
    concur_render: false
    needs_markers: true
    update_robot: true
    scene: data/scene_datasets/habitat-test-scenes/van-gogh-room.glb
    scene_dataset: default
    additional_object_paths: []
    seed: ${habitat.seed}
    turn_angle: 10
    tilt_angle: 15
    default_agent_id: 0
    debug_render: false
    debug_render_robot: false
    kinematic_mode: false
    debug_render_goal: true
    robot_joint_start_noise: 0.0
    ctrl_freq: 120.0
    ac_freq_ratio: 4
    load_objs: false
    hold_thresh: 0.09
    grasp_impulse: 1000.0
    agents:
    - agent_0
    agent_0:
      height: 1.5
      radius: 0.1
      sim_sensors:
        rgb_sensor:
          type: HabitatSimRGBSensor
          height: 256
          width: 256
          position:
          - 0.0
          - 1.25
          - 0.0
          orientation:
          - 0.0
          - 0.0
          - 0.0
          hfov: 90
          sensor_subtype: PINHOLE
          noise_model: None
          noise_model_kwargs: {}
      is_set_start_state: false
      start_position:
      - 0.0
      - 0.0
      - 0.0
      start_rotation:
      - 0.0
      - 0.0
      - 0.0
      - 1.0
      joint_start_noise: 0.0
      robot_urdf: data/robots/hab_fetch/robots/hab_fetch.urdf
      robot_type: FetchRobot
      ik_arm_urdf: data/robots/hab_fetch/robots/fetch_onlyarm.urdf
    habitat_sim_v0:
      gpu_device_id: 0
      gpu_gpu: false
      allow_sliding: true
      frustum_culling: true
      enable_physics: false
      physics_config_file: ./data/default.physics_config.json
      leave_context_with_background_renderer: false
      enable_gfx_replay_save: false
    ep_info: null
  task:
    reward_measure: distance_to_goal_reward
    success_measure: spl
    success_reward: 2.5
    slack_reward: -0.01
    end_on_success: true
    type: Nav-v0
    lab_sensors:
      pointgoal_with_gps_compass_sensor:
        type: PointGoalWithGPSCompassSensor
        goal_format: POLAR
        dimensionality: 2
    measurements:
      distance_to_goal:
        type: DistanceToGoal
        distance_to: POINT
      success:
        type: Success
        success_distance: 0.2
      spl:
        type: SPL
      distance_to_goal_reward:
        type: DistanceToGoalReward
    goal_sensor_uuid: pointgoal_with_gps_compass
    count_obj_collisions: true
    settle_steps: 5
    constraint_violation_ends_episode: true
    constraint_violation_drops_object: false
    force_regenerate: false
    should_save_to_cache: true
    must_look_at_targ: true
    object_in_hand_sample_prob: 0.167
    render_target: true
    ee_sample_factor: 0.2
    ee_exclude_region: 0.0
    base_angle_noise: 0.15
    base_noise: 0.05
    spawn_region_scale: 0.2
    joint_max_impulse: -1.0
    desired_resting_position:
    - 0.5
    - 0.0
    - 1.0
    use_marker_t: true
    cache_robot_init: false
    success_state: 0.0
    easy_init: false
    should_enforce_target_within_reach: false
    task_spec_base_path: habitat/task/rearrange/pddl/
    task_spec: ''
    pddl_domain_def: replica_cad
    obj_succ_thresh: 0.3
    art_succ_thresh: 0.15
    robot_at_thresh: 2.0
    filter_nav_to_tasks: []
    actions:
      stop:
        type: StopAction
      move_forward:
        type: MoveForwardAction
      turn_left:
        type: TurnLeftAction
      turn_right:
        type: TurnRightAction
  dataset:
    type: PointNav-v1
    split: train
    scenes_dir: data/scene_datasets
    content_scenes:
    - '*'
    data_path: data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz
  gym:
    auto_name: ''
    obs_keys: null
    action_keys: null
    achieved_goal_keys: []
    desired_goal_keys: []
```
</details>


## Default configuration
```python
# habitat/config/default_structured_configs.py

from dataclasses import dataclass, field
from typing import List, Dict

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class GymConfig:
    ...


@dataclass
class SensorConfig:
    ...


@dataclass
class MeasurementConfig:
    ...


@dataclass
class TaskConfig:
    lab_sensors: Dict[str, SensorConfig] = field(default_factory=dict)
    measurements: Dict[str, MeasurementConfig] = field(default_factory=dict)
    ...


@dataclass
class EnvironmentConfig:
    ...


@dataclass
class SimulatorSensorConfig:
    ...


@dataclass
class HabitatSimDepthSensorConfig:
    ...


@dataclass
class HabitatSimRGBSensorConfig:
    ...

@dataclass
class AgentConfig:
    sim_sensors: Dict[str, SimulatorSensorConfig] = field(default_factory=dict)
    ...


@dataclass
class SimulatorConfig:
    agent_0: AgentConfig = AgentConfig()
    ...


@dataclass
class DatasetConfig:
    ...


@dataclass
class HabitatConfig:
    seed: int = 100
    env_task: str = "GymHabitatEnv"
    env_task_gym_dependencies: List = field(default_factory=list)
    env_task_gym_id: str = ""
    environment: EnvironmentConfig = EnvironmentConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    task: TaskConfig = MISSING
    dataset: DatasetConfig = MISSING
    gym: GymConfig = GymConfig()


cs = ConfigStore.instance()
cs.store(
    group="habitat",
    name="habitat_config_base",
    node=HabitatConfig
)
cs.store(
    package="habitat.simulator.agent_0.sim_sensors.depth_sensor",
    group="habitat/simulator/sim_sensors",
    name="depth_sensor",
    node=HabitatSimDepthSensorConfig,
)
cs.store(
    package="habitat.simulator.agent_0.sim_sensors.rgb_sensor",
    group="habitat/simulator/sim_sensors",
    name="rgb_sensor",
    node=HabitatSimRGBSensorConfig,
)
cs.store(
    package="habitat.task",
    group="habitat/task",
    name="task_config_base",
    node=TaskConfig,
)
...
```
## Load a configuration from a yaml file
```yaml
# @package _global_

defaults:
  - /habitat: habitat_config_base
  - /habitat/task: pointnav
  - /habitat/simulator/sim_sensors:
    - depth_sensor
    - rgb_sensor
  - _self_

habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agent_0:
      sim_sensors:
        rgb_sensor:
          width: 256
          height: 256
        depth_sensor:
          width: 256
          height: 256
```

```yaml
# @package _global_

defaults:
  - pointnav_base
  - /habitat/dataset/pointnav: gibson
```


```python
import habitat

config = habitat.get_config("benchmark/nav/pointnav/pointnav_gibson.yaml")
```

## Make custom configuration defaults

## “Frozen”/read-only configs (config.freeze(True)) see here for details

## Overwrite configuration
### Via code
### Via a yaml file

## Delete node from the loaded yaml file

## Must be compatible with other hydra libraries (no `@main`)
## Support registries and their configs (Oleksandr Maksymets)
## Handle instances of the same class with different configs (example : multiple depth sensors). To avoid copy paste of config nodes have a sensor other nodes can inherit a config and override (RGB sensor with different uuids)


## Results No large default configuration with all the fields

## No sensor list, all sensors in the config are used?
## Composite of several configs: use same task config with different datasets
