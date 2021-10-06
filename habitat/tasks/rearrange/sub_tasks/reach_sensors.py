#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from gym import spaces

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sensors import EndEffectorToRestDistance
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import CollisionDetails
from habitat.tasks.utils import get_angle


@registry.register_measure
class RearrangeReachReward(Measure):
    cls_uuid: str = "rearrange_reach_reward"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangeReachReward.cls_uuid

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev = None
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToRestDistance.cls_uuid,
                RearrangeReachSuccess.cls_uuid,
            ],
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        cur_dist = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()
        if self._config.SPARSE_REWARD:
            is_succ = task.measurements.measures[
                RearrangeReachSuccess.cls_uuid
            ].get_metric()
            self._metric = self._config.SCALE * float(is_succ)
        else:
            if self._config.DIFF_REWARD:
                if self._prev is None:
                    self._metric = 0.0
                else:
                    self._metric = self._prev - cur_dist
            else:
                self._metric = -1.0 * self._config.SCALE * cur_dist

        self._prev = cur_dist


@registry.register_measure
class RearrangeReachSuccess(Measure):
    cls_uuid: str = "rearrange_reach_success"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RearrangeReachSuccess.cls_uuid

    def __init__(self, *args, sim, config, task, **kwargs):
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)
        self._config = config

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EndEffectorToRestDistance.cls_uuid,
            ],
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = (
            task.measurements.measures[
                EndEffectorToRestDistance.cls_uuid
            ].get_metric()
            < self._config.SUCC_THRESH
        )
