#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements tasks and measurements needed for training and benchmarking of
``habitat.Agent`` inside ``habitat.Env``.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from gym import Space, spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.core.simulator import SensorSuite, Simulator


class Measure:
    r"""Represents a measure that provides measurement on top of environment
    and task. This can be used for tracking statistics when running
    experiments. The user of this class needs to implement the reset_metric
    and update_metric method and the user is also required to set the below
    attributes:

    Attributes:
        uuid: universally unique id.
        _metric: metric for the ``Measure``, this has to be updated with each
            ``step`` call on ``habitat.Env``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.uuid = self._get_uuid(*args, **kwargs)
        self._metric = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Reset ``_metric``, this method is called from ``Env`` on each reset.
        """
        raise NotImplementedError

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Update ``_metric``, this method is called from ``Env`` on each 
        ``step``.
        """
        raise NotImplementedError

    def get_metric(self):
        r"""
        Returns:
             the current metric for ``Measure``.
        """
        return self._metric


class Metrics(dict):
    r"""Dictionary containing measurements.

    Args:
        measures: list of ``Measure`` whose metrics are fetched and packaged.
    """

    def __init__(self, measures: Dict[str, Measure]) -> None:
        data = [
            (uuid, measure.get_metric()) for uuid, measure in measures.items()
        ]
        super().__init__(data)


class Measurements:
    r"""Represents a set of Measures, with each ``Measure`` being identified
    through a unique id.

    Args:
        measures: list containing ``Measure``, uuid of each
            ``Measure`` must be unique.
    """

    measures: Dict[str, Measure]

    def __init__(self, measures: List[Measure]) -> None:
        self.measures = OrderedDict()
        for measure in measures:
            assert (
                measure.uuid not in self.measures
            ), "'{}' is duplicated measure uuid".format(measure.uuid)
            self.measures[measure.uuid] = measure

    def reset_measures(self, *args: Any, **kwargs: Any) -> None:
        for measure in self.measures.values():
            measure.reset_metric(*args, **kwargs)

    def update_measures(self, *args: Any, **kwargs: Any) -> None:
        for measure in self.measures.values():
            measure.update_metric(*args, **kwargs)

    def get_metrics(self) -> Metrics:
        r"""
        Returns:
            collect measurement from all Measures and return it packaged inside
            Metrics.
        """
        return Metrics(self.measures)


class EmptySpace(Space):
    def sample(self):
        return None

    def contains(self, x):
        return False


class EmbodiedTask:
    r"""Base class for embodied tasks. When subclassing the user has to
    define the attributes ``measurements`` and ``sensor_suite``.

    Args:
        config: config for the task.
        sim: reference to the simulator for calculating task observations.
        dataset: reference to dataset for task instance level information.

    Attributes:
        measurements: set of task measures.
        sensor_suite: suite of task sensors.
    """

    _config: Any
    _sim: Optional[Simulator]
    _dataset: Optional[Dataset]
    measurements: Measurements
    sensor_suite: SensorSuite

    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        self._config = config
        self._sim = sim
        self._dataset = dataset
        self._possible_actions = OrderedDict()
        for action in config.POSSIBLE_ACTIONS:
            assert action in registry.mapping["task_action"]
            self._possible_actions[action] = registry.mapping["task_action"][action]

    def step(self, action: int, episode, action_args=None):
        if action_args is None:
            action_args = {}
        if isinstance(action, int):
            action = list(self._possible_actions.keys())[action]
        assert action in self._possible_actions.keys()
        action_method = self._possible_actions[action]
        observations = action_method(self, **action_args)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode
            )
        )

        return observations

    def step(
        self,
        action: Union[str, int],
        episode: Type[Episode],
        action_args: Dict[str, Any] = None,
    ):
        if action_args is None:
            action_args = {}
        if isinstance(action, (int, np.integer)):
            if action >= len(self._possible_actions):
                raise ValueError(f"Action index '{action}' is out of range.")
            action = list(self._possible_actions.keys())[action]
        assert (
            action in self._possible_actions
        ), f"Can't find '{action}' action in {self._possible_actions.keys()}."

        action_method = self._possible_actions[action]
        observations = action_method(self, **action_args)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode
            )
        )

        return observations

    def get_action_by_index(self, action_index: int):
        if action_index >= len(self._possible_actions):
            raise ValueError(f"Action index '{action}' is out of range.")
        return list(self._possible_actions.keys())[action_index]

    def action_space(self, action: Union[int, str] = None) -> Space:
        if action is None:
            return spaces.Discrete(len(self._possible_actions))
        else:
            if isinstance(action, (int, np.integer)):
                action = self.get_action_by_index(action)
            return registry.get_task_action_spec(name=action)

    def overwrite_sim_config(
        self, sim_config: Config, episode: Type[Episode]
    ) -> Config:
        r"""
        Args:
            sim_config: config for simulator.
            episode: current episode.

        Returns:
            update config merging information from ``sim_config`` and 
                ``episode``.
        """
        raise NotImplementedError
