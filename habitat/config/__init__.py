#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from habitat.config.default import Config, extend_default_config, get_config

r"""Habitat-API Configuration
==============================

Habitat-API uses [Yacs configuration system](https://github.com/rbgirshick/yacs)
with the paradigm of `your code + a YACS config for experiment E (+
external dependencies + hardware + other nuisance terms ...) =
reproducible experiment E`. Yacs advantages:
- Checks for type consistency.
- All parameters and default values are searchable in the code.
- A parameter doesn't need to be set always as each parameter can have a
    default value.
- Ability to freeze config to prevent unintended changes.

## Config usage
An example of how to merge default config with 2 others configs and overwrite
one parameter that could come from the command line:
```
    merged_config = get_config(
        config_paths=["configs/tasks/pointnav.yaml",
            "configs/dataset/val.yaml"],
        opts=["habitat.environment.max_episode_steps", steps_limit]
    )

```

## Config structure
Below is the structure of config used for Habitat:
- habitat
    - Environment
    - Task
        - Sensors
        - Measurements
    - Simulator
        - Agent
            - Sensors
    - Dataset

We use node names (e.g. `sensors: ['rgb_sensor', 'depth_sensor']`) instead of list
of config nodes (e.g. `sensors: [{TYPE = "HabitatSimDepthSensor",
min_depth = 0}, ...]`) to declare the Sensors attached to an Agent or Measures
enabled for the Task . With this approach, it's still easy to overwrite a
particular sensor parameter in yaml file without redefining the whole sensor
config.

## Extending the config without defaults
Create a YAML file and add new fields and values. Load the custom config using
`habitat.get_config()` and defined fields will be merged in default Habitat config:
```
import habitat
import argparse
from typing import List, Optional, Union

config = habitat.get_config("{path to user define yaml config}")
env = habitat.Env(config)
```

## Extending the config with default values
Example of how to extend a config outside of `habtiat-api` repository.
```
import habitat
import argparse
from typing import List, Optional, Union

episode_info_example = habitat.Config()
# The type field is used to look-up the measure in the registry.
# By default, the things are registered with the class name
episode_info_example.type = "EpisodeInfoExample"
episode_info_example.value = 5
# Extend the default config to include this
habitat.config.extend_default_config(
    "habitat.task.episode_info_example", episode_info_example
)

# It will then show up in the default config
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        type=str,
        default="configs/tasks/pointnav.yaml,"
                "configs/datasets/pointnav/habitat_test.yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    config = get_config(config_paths=args.task_config, opts=args.opts)
    env = habitat.Env(config)

```"""

__all__ = ["Config", "get_config", "extend_default_config"]
