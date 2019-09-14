#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements tasks and measurements needed for training and benchmarking of
``habitat.Agent`` inside ``habitat.Env``.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Type, Union

import gym
import numpy as np
from gym import Space, spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.core.simulator import SensorSuite, Simulator


class Action:
    r"""
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.uuid = self._get_uuid(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def reset(self, *args: Any, **kwargs: Any) -> None:
        r"""Reset method is called from ``Env`` on each reset.
        """
        raise NotImplementedError

    def step(self, *args: Any, **kwargs: Any) -> None:
        r"""Step method is called from ``Env`` on each ``step``.
        """
        raise NotImplementedError

    def get_action_space(self):
        r"""
        Returns:
             the current action space.
        """
        return None


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


class ActionSpace(spaces.Dict):
    """
    A dictionary of action and their argument spaces

    Example usage:
    self.observation_space = spaces.ActionSpace(
        "move": spaces.Dict({
            "position": spaces.Discrete(2),
            "velocity": spaces.Discrete(3)
            },
        "move_forward": EmptySpace,
        )
    )
    """

    def __init__(self, spaces):
        if isinstance(spaces, dict):
            self.spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            self.spaces = OrderedDict(spaces)
        self.actions_select = gym.spaces.Discrete(len(self.spaces))

    @property
    def n(self):
        return len(self.spaces)

    def sample(self):
        action_index = self.actions_select.sample()
        return {
            "action": list(self.spaces.keys())[action_index],
            "action_args": list(self.spaces.values())[action_index].sample(),
        }

    def contains(self, x):
        if not isinstance(x, dict) and {"action", "action_args"} not in x:
            return False
        if not self.spaces[x["action"]].contains(x["action_args"]):
            return False
        return True

    def __repr__(self):
        return (
            "ActionSpace("
            + ", ".join([k + ":" + str(s) for k, s in self.spaces.items()])
            + ")"
        )


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
    _is_episode_active: bool
    measurements: Measurements
    sensor_suite: SensorSuite

    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        self._config = config
        self._sim = sim
        self._dataset = dataset

        task_measurements = []
        for measurement_name in config.MEASUREMENTS:
            measurement_cfg = getattr(config, measurement_name)
            measure_type = registry.get_measure(measurement_cfg.TYPE)
            assert (
                measure_type is not None
            ), "invalid measurement type {}".format(measurement_cfg.TYPE)
            print("measurement type {}".format(measurement_cfg.TYPE))
            task_measurements.append(
                measure_type(
                    sim=sim, config=measurement_cfg, dataset=dataset, task=self
                )
            )
        self.measurements = Measurements(task_measurements)

        task_sensors = []
        for sensor_name in config.SENSORS:
            sensor_cfg = getattr(config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)
            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            print("sensor type {}".format(sensor_cfg.TYPE))
            task_sensors.append(
                sensor_type(sim=sim, config=sensor_cfg, dataset=dataset)
            )

        self.sensor_suite = SensorSuite(task_sensors)

        # TODO: move sensors and measures to `_init_entities`
        self.actions = self._init_entities(
            entity_names=config.POSSIBLE_ACTIONS,
            register_func=registry.get_task_action,
            entities_config=self._config.ACTIONS,
        )

    def _init_entities(
        self, entity_names, register_func, entities_config=None
    ):
        if entities_config is None:
            entities_config = self._config

        entities = OrderedDict()
        for entity_name in entity_names:
            entity_cfg = getattr(entities_config, entity_name)
            entity_type = register_func(entity_cfg.TYPE)
            assert (
                entity_type is not None
            ), f"invalid {entity_name} type {entity_cfg.TYPE}"
            entities[entity_name] = entity_type(
                sim=self._sim,
                config=entity_cfg,
                dataset=self._dataset,
                task=self,
            )
        return entities

    def reset(self, episode):
        observations = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode)

        return observations

    def step(self, action: Union[int, Dict[str, Any]], episode: Type[Episode]):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]
        if isinstance(action_name, (int, np.integer)):
            if action_name >= len(self.actions):
                raise ValueError(
                    f"Action index '{action_name}' is out of range."
                )
            action_name = list(self.actions.keys())[action_name]
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]
        observations = task_action.step(self, **action["action_args"])
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, action=action
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations

    def get_action_name(self, action_index: int):
        if action_index >= len(self.actions):
            raise ValueError(f"Action index '{action}' is out of range.")
        return list(self.actions.keys())[action_index]

    def action_space(self) -> Space:
        return ActionSpace(
            {
                action_name: action_instance.get_action_space()
                for action_name, action_instance in self.actions.items()
            }
        )

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

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Type[Episode],
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError

    @property
    def is_episode_active(self):
        return self._is_episode_active

    def seed(self, seed: int) -> None:
        return
