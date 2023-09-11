Habitat-Lab Configuration System
================================
![Habitat with Hydra](/res/img/habitat_with_hydra.png)

For a description of some of the most important configuration keys of the habitat benchmarks, refer to [this file](CONFIG_KEYS.md).

Habitat-Lab's configuration system has been changed from [YACS](https://github.com/rbgirshick/yacs)
to [Hydra](https://hydra.cc).

## Hydra Concepts used in Habitat-Lab

With Hydra, the [Output Config](https://hydra.cc/docs/advanced/terminology/#output-config)
is composed dynamically at run time from
the [Input Configs](https://hydra.cc/docs/advanced/terminology/#input-configs) specified in
the [Defaults List](https://hydra.cc/docs/advanced/terminology/#defaults-list) and
[Overrides](https://hydra.cc/docs/advanced/terminology/#overrides) from the primary yaml config file or command line.

Default config values and their types are defined via
[Structured Configs](https://hydra.cc/docs/advanced/terminology/#structured-config). Structured Configs are used as a
config validation schemas to ensure that all the required fields are set and match the required type, and also as a
configs, in place of configuration yaml files. All Structured Configs are registered in
the [ConfigStore](https://hydra.cc/docs/tutorials/structured_config/config_store/) - in-memory
Structured Configs registry. Habitat-Lab's Structured Configs are defined and registered to the ConfigStore in
[habitat-lab/habitat/config/default_structured_configs.py](default_structured_configs.py).

Similar configs are grouped in [Config Groups](https://hydra.cc/docs/advanced/terminology/#config-group)
(placed in the same directories in the [Config Search Path](https://hydra.cc/docs/advanced/terminology/#config-search-path)).
These configs are also called [Config Group Options](https://hydra.cc/docs/advanced/terminology/#config-group-option).
For example, Embodied AI task specifications supported by Habitat-Lab are placed in the `habitat/task` Config Group
and to use the PointNav task just add `habitat/task: pointnav` line to your config's Defaults List.

Each Input Config's position in the Output Config is determined via
[Package](https://hydra.cc/docs/advanced/terminology/#package). In other words, a Package is the path to node in a config.
By default, the Package of a Config Group Option is derived from the Config Group. e.g: task configs in `habitat/task`
will have the package `habitat.task` by default (i.e. will be placed under `habitat.task` in the final Output Config).

To make all habitat configs visible to Hydra, the `habitat/config` path is added to Config Search Path by
extending [SearchPathPlugin](https://hydra.cc/docs/advanced/plugins/overview/#searchpathplugin)
(see `HabitatConfigPlugin` class in the [habitat-lab/habitat/config/default_structured_configs.py](default_structured_configs.py)).
`HabitatConfigPlugin` is registered by `register_hydra_plugin(HabitatConfigPlugin)` function call in Habitat-Lab's `get_config`
(see `get_config` definition in [habitat-lab/habitat/config/default.py](default.py)).

Note, due to the fact that Habitat-Lab doesn't have a single entry point, but creates config objects in multiple places,
for example, Jupyter Notebooks, example scripts in the `/examples` folder, tests, etc. Hydra's
[Compose API](https://hydra.cc/docs/advanced/compose_api/) is used to create the config in the Habitat-Lab's
`get_config` function instead of `@hydra.main` decorator (used in most Hydra examples).

For more detailed explanation of Hydra concepts visit Hydra's [website](https://hydra.cc/docs/intro/).

## Config directory structure
```
habitat-lab/habitat/config/
|
|_benchmark  # benchmark configs (primary configs to be used in habitat.get_config)
| |_nav
| |_rearrange
|
|_habitat    # habitat configs (habitat config groups options)
| |_dataset
| |_simulator
| |_task
|
|_test       # test configs
```


## What's changed?
<details>
<summary>PointNav benchmark: Input Config.</summary>

```yaml
# @package _global_

# defaults list:
defaults:
  - /habitat: habitat_config_base
  - /habitat/task: pointnav
  - /habitat/simulator/agents:
    - rgbd_agent
  - /habitat/dataset/pointnav: gibson
  - _self_

# config values overrides:
habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agents:
      rgbd_agent:
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
<summary>PointNav benchmark: Output Config.</summary>

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
    forward_step_size: 0.25
    turn_angle: 10
    create_renderer: false
    requires_textures: true
    lag_observations: 0
    auto_sleep: false
    step_physics: true
    concur_render: false
    needs_markers: true
    update_articulated_agent: true
    scene: data/scene_datasets/habitat-test-scenes/van-gogh-room.glb
    scene_dataset: default
    additional_object_paths: []
    seed: ${habitat.seed}
    default_agent_id: 0
    debug_render: false
    debug_render_robot: false
    kinematic_mode: false
    debug_render_goal: true
    robot_joint_start_noise: 0.0
    ctrl_freq: 120.0
    ac_freq_ratio: 4
    load_objs: true
    hold_thresh: 0.09
    grasp_impulse: 1000.0
    agents:
      rgbd_agent:
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
          depth_sensor:
            type: HabitatSimDepthSensor
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
            min_depth: 0.0
            max_depth: 10.0
            normalize_depth: true
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
        articulated_agent_urdf: data/robots/hab_fetch/robots/hab_fetch.urdf
        articulated_agent_type: FetchRobot
        ik_arm_urdf: data/robots/hab_fetch/robots/fetch_onlyarm.urdf
    agents_order:
    - rgbd_agent
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
    actions:
      stop:
        type: StopAction
      move_forward:
        type: MoveForwardAction
      turn_left:
        type: TurnLeftAction
        turn_angle: 10
      turn_right:
        type: TurnRightAction
        turn_angle: 10
  dataset:
    type: PointNav-v1
    split: train
    scenes_dir: data/scene_datasets
    content_scenes:
    - '*'
    data_path: data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz
  gym:
    obs_keys: null
    action_keys: null
    achieved_goal_keys: []
    desired_goal_keys: []
```
</details>

- The configuration keys are lowercase and live under the `habitat` namespace.
- Embodied AI task specifications are separated from the benchmark configs (more general configs that include
  task config as one of their sub-configs in the Defaults List). Task configs are grouped in the `habitat/task`
  Config Group (located in [habitat-lab/config/habitat/task](habitat/task) directory) and benchmark configs
  are placed in the [habitat-lab/config/benchmark](benchmark) directory.
- `agent.sensors` is renamed to `agent.sim_sensors`,  `task.sensors` is renamed to `task.lab_sensors`.
- Actions, agents, measurements and sensors configs are not directly attached to the simulator, task or agent
  config nodes but are added to `task.actions`, `simulator.agents`, `task.measurements`, `task.lab_sensors`,
  `simulator.<agent_name>.sim_sensors` sub-nodes of type `Dict[str, CorrespondingConfigNodeType]`
  that can be updated dynamically. Consequently, Output Config contains only those config nodes that are listed in
  `task.actions`, `simulator.agents`, `task.measurements`, `task.lab_sensors`, `simulator.agents.agent_name.sim_sensors`
  (not all possible actions, agents, measurements and sensors config nodes).
- `task.possible_actions` is removed.

### New functionality enabled
After the configuration system migration to Hydra, Habitat-Lab automatically supports all Hydra's
and [OmegaConf's](https://omegaconf.readthedocs.io/) features (note that Hydra is build on top of OmegaConf).
Some of new features are listed below:
- Config node reuse by redefining the config package. For example, it is possible to use the same agent config to
  configure two agents of the same type:
```yaml
defaults:
  - /habitat/simulator/sensor_setups@habitat.simulator.agents.agent_0: depth_head_agent
  - /habitat/simulator/sensor_setups@habitat.simulator.agents.agent_1: depth_head_agent
```
- [Variable interpolation](https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#variable-interpolation).
  For example, use the same seed value for `SimulatorConfig.seed` as in `HabitatConfig.seed` (see `SimulatorConfig`
  in the [habitat-lab/habitat/config/default.py](default.py)):
```python
from omegaconf import II

@dataclass
class SimulatorConfig(HabitatBaseConfig):
    # Other SimulatorConfig keys are omitted in this code snippet
    seed: int = II("habitat.seed")
```
- [Parameter sweeping and multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). For example, launching 3 experiments with three different learning rates:
```bash
python -u -m habitat_baselines.run --config-name=config.yaml  \
–-multirun habitat_baselines.rl.ppo.lr 2.5e-4,2.5e-5,2.5e-6
```
- Seamless [SLURM](https://slurm.schedmd.com/documentation.html) integration through
  [Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher/).
  To enable the feature Submitit plugin should be installed: `pip install hydra-submitit-launcher --upgrade`
  and `submitit_slurm` launcher specified in the command line `hydra/launcher=submitit_slurm`:
```bash
python -u -m habitat_baselines.run --config-name=config.yaml  \
hydra/launcher=submitit_slurm --multirun
```
- Making the config key required by setting its value to `MISSING`. For example, we require the user to explicitly
  set the `task` and the `dataset` in every Habitat-Lab benchmark config (see `HabitatConfig` Structured Config
 in the [habitat-lab/habitat/config/default.py](default.py)):
```python
from omegaconf import MISSING

@dataclass
class HabitatConfig(HabitatBaseConfig):
    # Other HabitatConfig keys are omitted in this code snippet
    task: TaskConfig = MISSING
    dataset: DatasetConfig = MISSING
```


But Hydra allows a lot more! Visit Hydra's [website](https://hydra.cc/docs/intro/) for mode details about Hydra's features.

## How to ______?
### read the config
```python
import habitat

config = habitat.get_config("benchmark/nav/pointnav/pointnav_gibson.yaml")
```
### override the config
#### via command line
Override config values:
```bash
python -u -m habitat_baselines.run --config-name=pointnav/ddppo_pointnav.yaml \
habitat.environment.max_episode_steps=250 \
habitat_baselines.total_num_steps=100
```

Override the Config Group Option value:
```bash
python -u -m habitat_baselines.run --config-name=pointnav/ddppo_pointnav.yaml \
benchmark/nav/pointnav=pointnav_hm3d  # overriding benchmark config to be pointnav_hm3d
```

#### via yaml
Yaml file definitions or overrides are defined after the Defaults List:
```yaml
# @package _global_

defaults:
  - /habitat: habitat_config_base
  - /habitat/task: pointnav
  - /habitat/simulator/agents:
    - rgbd_agent
  - /habitat/dataset/pointnav: gibson
  - _self_

# config values overrides:
habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agents:
      rgbd_agent:
        sim_sensors:
          rgb_sensor:
            width: 256
            height: 256
          depth_sensor:
            width: 256
            height: 256
```
Note, that its also possible to override the Config Group Options in the Defaults list:
```yaml
# @package _global_

defaults:
  - any_base_config
  - override /habitat/dataset/pointnav: gibson  # override the any_base_config dataset to gibson
```

#### via code
The config object returned from `habitat.get_config` is readonly. To be able to update the config object via code, wrap
it in the `read_write(config)` context manager:

```python
from habitat.config.read_write import read_write

with read_write(config):
    config.habitat.simulator.concur_render = False
    agent_config = get_agent_config(config.habitat.simulator)
    agent_config.sim_sensors.update(
        {"third_rgb_sensor": ThirdRGBSensorConfig(height=512, width=512)}
    )
```

### extend the config
pointnav_base.yaml
```yaml
# @package _global_

defaults:
  - /habitat: habitat_config_base
  - /habitat/task: pointnav
  - /habitat/simulator/agents:
    - rgbd_agent
  - _self_

habitat:
  environment:
    max_episode_steps: 500
  simulator:
    agents:
      rgbd_agent:
        sim_sensors:
          rgb_sensor:
            width: 256
            height: 256
          depth_sensor:
            width: 256
            height: 256
```

pointnav_habitat_test.yaml
```yaml
# @package _global_

defaults:
  - pointnav_base
  - /habitat/dataset/pointnav: habitat_test
```

pointnav_gibson.yaml
```yaml
# @package _global_

defaults:
  - pointnav_base
  - /habitat/dataset/pointnav: gibson
```

### define new structured config
In case, you directly work with code in the habitat-lab repository (for example, clone and extend), then to add new
custom Structured Config just define and add the class to the ConfigStore in the
[habitat-lab/habitat/config/default_structured_configs.py](default_structured_configs.py).

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class CustomStructuredConfig:
    custom_config_key: KeyType = DefaultValue

cs = ConfigStore.instance()
cs.store(
    group="habitat/custom_structured_config",  # config group
    name="custom_structured_config",           # config name
    node=CustomStructuredConfig,
    # Note, it is also possible to override the package (that's derived from the Config Group by default)
    # package="habitat.new.custom_structured_config",
)
```
