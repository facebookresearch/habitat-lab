#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements tasks and measurements needed for training and benchmarking of
``habitat.Agent`` inside ``habitat.Env``.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Type

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import SensorSuite, Simulator


class Measure:
    r"""Represents a measure that provides measurement on top of environment
    and task.

    :data uuid: universally unique id.
    :data _metric: metric for the `Measure`, this has to be updated with
        each `step() <env.Env.step()>` call on `env.Env`.

    This can be used for tracking statistics when running experiments. The
    user of this class needs to implement the `reset_metric()` and
    `update_metric()` method and the user is also required to set the
    `uuid <Measure.uuid>` and `_metric` attributes.

    .. (uuid is a builtin Python module, so just `uuid` would link there)
    """

    _metric: Any
    uuid: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.uuid = self._get_uuid(*args, **kwargs)
        self._metric = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Reset `_metric`, this method is called from `env.Env` on each
        reset.
        """
        raise NotImplementedError

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        r"""Update `_metric`, this method is called from `env.Env` on each
        `step() <env.Env.step()>`
        """
        raise NotImplementedError

    def get_metric(self):
        r"""..

        :return: the current metric for `Measure`.
        """
        return self._metric


class Metrics(dict):
    r"""Dictionary containing measurements.
    """

    def __init__(self, measures: Dict[str, Measure]) -> None:
        """Constructor

        :param measures: list of `Measure` whose metrics are fetched and
            packaged.
        """
        data = [
            (uuid, measure.get_metric()) for uuid, measure in measures.items()
        ]
        super().__init__(data)


class Measurements:
    r"""Represents a set of Measures, with each `Measure` being identified
    through a unique id.
    """

    measures: Dict[str, Measure]

    def __init__(self, measures: List[Measure]) -> None:
        """Constructor

        :param measures: list containing `Measure`, uuid of each
            `Measure` must be unique.
        """
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
        r"""Collects measurement from all `Measure`\ s and returns it
        packaged inside `Metrics`.
        """
        return Metrics(self.measures)


class EmbodiedTask:
    r"""Base class for embodied tasks.

    Args:
        config: config for the task.
        sim: reference to the simulator for calculating task observations.
        dataset: reference to dataset for task instance level information.

    When subclassing the user has to define the attributes `measurements` and
    `sensor_suite`.

    :data measurements: set of task measures.
    :data sensor_suite: suite of task sensors.
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

    def overwrite_sim_config(
        self, sim_config: Config, episode: Type[Episode]
    ) -> Config:
        r"""Update config merging information from :p:`sim_config` and
        :p:`episode`.

        :param sim_config: config for simulator.
        :param episode: current episode.
        """
        raise NotImplementedError
