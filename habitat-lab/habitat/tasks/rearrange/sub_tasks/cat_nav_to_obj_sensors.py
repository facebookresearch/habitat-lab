#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.sub_tasks.cat_nav_to_obj_task import CatDynNavRLEnv

@registry.register_sensor
class ReceptacleSegmentationSensor(Sensor):
    cls_uuid: str = "receptacle_segmentation"

    def __init__(
        self,
        sim,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._config = config
        self._dimensionality = self._config.dimensionality
        self._sim = sim
        self._instance_ids_start = self._sim.habitat_config.instance_ids_start
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(
                self._dimensionality,
                self._dimensionality,
                1,
            ),
            low=0,
            high=1,
            dtype=np.uint8,
        )

    def get_observation(self, observations, *args, episode, task: CatDynNavRLEnv, **kwargs):
        obs = np.copy(observations["robot_head_panoptic"])
        obj_id_map = np.zeros(np.max(obs) + 1, dtype=np.int32)
        for obj_id, semantic_id in task.receptacle_semantic_ids.items():
            instance_id = obj_id + self._instance_ids_start
            # Skip if receptacle is not in the agent's viewport
            if instance_id >= obj_id_map.shape[0]:
                continue
            obj_id_map[instance_id] = semantic_id
        obs = obj_id_map[obs]
        # from habitat_sim.utils.viz_utils import observation_to_image
        # sem = observation_to_image(obs, "semantic")
        # sem.save("sem.png")
        # import pdb; pdb.set_trace()
        return obs
