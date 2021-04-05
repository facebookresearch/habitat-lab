#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Habitat Lab Configuration
==============================

Habitat Lab uses [Yacs configuration system](https://github.com/rbgirshick/yacs)
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
        opts=["ENVIRONMENT.MAX_EPISODE_STEPS", steps_limit]
    )

```

## Config structure
Below is the structure of config used for Habitat:
- Environment
- Task
    - Sensors
    - Measurements
- Simulator
    - Agent
        - Sensors
- Dataset

We use node names (e.g. `SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR']`) instead of list
of config nodes (e.g. `SENSORS: [{TYPE = "HabitatSimDepthSensor",
MIN_DEPTH = 0}, ...]`) to declare the Sensors attached to an Agent or Measures
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
Example of how to extend a config outside of `habtiat-lab` repository.
First, we create a config extending the default config in the code and re-use
`habitat.get_config()`:
```
import habitat
import argparse
from typing import List, Optional, Union

_C = habitat.get_config()
_C.defrost()
# Add new parameters to the config
_C.TASK.EPISODE_INFO = habitat.Config()
_C.TASK.EPISODE_INFO.TYPE = "EpisodeInfo"
_C.TASK.EPISODE_INFO.VALUE = 5
_C.TASK.MEASUREMENTS.append("EPISODE_INFO")

# New function returning extended Habitat config that should be used instead
# of habitat.get_config()
def my_get_config(
        config_paths: Optional[Union[List[str], str]] = None,
        opts: Optional[list] = None,
) -> habitat.Config:
    CONFIG_FILE_SEPARATOR = ","
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

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
    config = my_get_config(config_paths=args.task_config, opts=args.opts)
    env = habitat.Env(config)

```"""

from habitat.config.default import Config, get_config

__all__ = ["Config", "get_config"]
