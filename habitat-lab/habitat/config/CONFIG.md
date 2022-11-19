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
