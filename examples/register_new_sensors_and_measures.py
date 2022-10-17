#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
from gym import spaces

import habitat


# Define the measure and register it with habitat
# By default, the things are registered with the class name
@habitat.registry.register_measure
class EpisodeInfoExample(habitat.Measure):
    def __init__(self, sim, config, **kwargs: Any):
        # This measure only needs the config
        self._config = config

        super().__init__()

    # Defines the name of the measure in the measurements dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "episode_info"

    # This is called whenver the environment is reset
    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        # Our measure always contains all the attributes of the episode
        self._metric = vars(episode).copy()
        # But only on reset, it has an additional field of my_value
        self._metric["my_value"] = self._config.VALUE

    # This is called whenver an action is taken in the environment
    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        # Now the measure will just have all the attributes of the episode
        self._metric = vars(episode).copy()


# Define the sensor and register it with habitat
# For the sensor, we will register it with a custom name
@habitat.registry.register_sensor(name="my_supercool_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim
        # Prints out the answer to life on init
        print("The answer to life is", self.config.answer_to_life)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position


def main():
    # Get the default config node
    config = habitat.get_config(config_paths="tasks/pointnav.yaml")
    with habitat.config.read_write(config):
        # Add things to the config to for the measure
        config.habitat.task.episode_info_example = habitat.Config()
        # The type field is used to look-up the measure in the registry.
        # By default, the things are registered with the class name
        config.habitat.task.episode_info_example.type = "EpisodeInfoExample"
        config.habitat.task.episode_info_example.VALUE = 5
        # Add the measure to the list of measures in use
        config.habitat.task.measurements.append("episode_info_example")

        # Now define the config for the sensor
        config.habitat.task.agent_position_sensor = habitat.Config()
        # Use the custom name
        config.habitat.task.agent_position_sensor.type = "my_supercool_sensor"
        config.habitat.task.agent_position_sensor.answer_to_life = 42
        # Add the sensor to the list of sensors in use
        config.habitat.task.sensors.append("agent_position_sensor")

    with habitat.Env(config=config) as env:
        print(env.reset()["agent_position"])
        print(env.get_metrics()["episode_info"])
        print(env.step("move_forward")["agent_position"])
        print(env.get_metrics()["episode_info"])


if __name__ == "__main__":
    main()
