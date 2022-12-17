#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import spaces
from omegaconf import MISSING

import habitat
from habitat.config.default_structured_configs import (
    LabSensorConfig,
    MeasurementConfig,
)


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

    # This is called whenever the environment is reset
    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        # Our measure always contains all the attributes of the episode
        self._metric = vars(episode).copy()
        # But only on reset, it has an additional field of my_value
        self._metric["my_value"] = self._config.VALUE

    # This is called whenever an action is taken in the environment
    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        # Now the measure will just have all the attributes of the episode
        self._metric = vars(episode).copy()


# define a configuration for this new measure
@dataclass
class EpisodeInfoExampleConfig(MeasurementConfig):
    # Note that typing is required on all fields
    type: str = "EpisodeInfoExample"
    VALUE: int = -1


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

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position


# define a configuration for this new sensor
@dataclass
class AgentPositionSensorConfig(LabSensorConfig):
    # Note that typing is required on all fields
    type: str = "my_supercool_sensor"
    # MISSING makes this field have no defaults
    answer_to_life: int = MISSING


def main():
    # Get the default config node
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    with habitat.config.read_write(config):
        my_value = 5
        # Add things to the config to for the measure
        config.habitat.task.measurements[
            "episode_info_example"
        ] = EpisodeInfoExampleConfig(VALUE=my_value)

        # Now define the config for the sensor
        config.habitat.task.lab_sensors[
            "agent_position_sensor"
        ] = AgentPositionSensorConfig(answer_to_life=42)

    with habitat.Env(config=config) as env:
        print(env.reset()["agent_position"])
        print(env.get_metrics()["episode_info"])
        # After reset my_value should be set
        assert env.get_metrics()["episode_info"]["my_value"] == my_value
        print(env.step("move_forward")["agent_position"])
        print(env.get_metrics()["episode_info"])
        # my_value should only be present at reset, not after step
        assert "my_value" not in env.get_metrics()["episode_info"]


if __name__ == "__main__":
    main()
