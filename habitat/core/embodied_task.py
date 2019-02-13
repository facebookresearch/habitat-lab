#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Type, Optional

from habitat.config import Config
from habitat.core.dataset import Episode, Dataset
from habitat.core.simulator import SensorSuite, Simulator


class EmbodiedTask:
    """Base class for embodied tasks. When subclassing the user has
    to define the attributes listed below.

    Args:
        config: config for the task.
        sim: reference to the simulator for calculating task observations.
        dataset: reference to dataset for task instance level information.
        sensor_suite: task specific custom sensors which are added
            on top of simulator's sensors. These sensors can be used to
            connect observation from simulator and task instances,
            eg PointGoalSensor defined in habitat.tasks.nav.nav_task.

    Attributes:
        sensor_suite: suits of task sensors.
    """

    _config: Any
    _sim: Optional[Simulator]
    _dataset: Optional[Dataset]
    _sensor_suite: SensorSuite

    def __init__(
        self,
        config: Config,
        sim: Simulator,
        dataset: Optional[Dataset] = None,
        sensor_suite: SensorSuite = SensorSuite([]),
    ) -> None:
        self._config = config
        self._sim = sim
        self._dataset = dataset
        self._sensor_suite = sensor_suite

    def overwrite_sim_config(
        self, sim_config: Config, episode: Type[Episode]
    ) -> Config:
        """
        Args:
            sim_config: config for simulator.
            episode: current episode.

        Returns:
            update config merging information from sim_config and episode.
        """
        raise NotImplementedError

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite
