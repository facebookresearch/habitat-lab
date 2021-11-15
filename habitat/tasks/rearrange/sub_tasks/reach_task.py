#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import magnum as mn
import numpy as np

from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


@registry.register_task(name="RearrangeReachTask-v0")
class RearrangeReachTaskV1(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)

    def step(self, action, episode):
        obs = super().step(action=action, episode=episode)

        return obs

    def reset(self, episode):
        super().reset(episode)

        # Pick a random goal in the robot's workspace

        ee_region = self._sim.robot.params.ee_constraint
        full_range = mn.Range3D.from_size(
            mn.Vector3(ee_region[:, 0]),
            mn.Vector3(ee_region[:, 1] - ee_region[:, 0]),
        )

        allowed_space = mn.Range3D.from_center(
            full_range.center(),
            0.5 * full_range.size() * self._config.EE_SAMPLE_FACTOR,
        )
        if self._config.EE_EXCLUDE_REGION != 0.0:
            not_allowed_space = mn.Range3D.from_center(
                full_range.center(),
                0.5 * full_range.size() * self._config.EE_EXCLUDE_REGION,
            )
            while True:
                self._desired_resting = np.random.uniform(
                    low=allowed_space.min, high=allowed_space.max
                )
                if not not_allowed_space.contains(self._desired_resting):
                    break
        else:
            self._desired_resting = np.random.uniform(
                low=allowed_space.min, high=allowed_space.max
            )

        if self._config.RENDER_TARGET:
            global_pos = self._sim.robot.base_transformation.transform_point(
                self._desired_resting
            )
            self._sim.viz_ids["reach_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["reach_target"]
            )

        return super(RearrangeReachTaskV1, self).reset(episode)
