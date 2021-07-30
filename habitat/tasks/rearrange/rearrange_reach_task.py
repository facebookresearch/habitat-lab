#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


@registry.register_task(name="RearrangeReachTask-v0")
class RearrangeReachTaskV1(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.targ_idx = 0
        self.abs_targ_idx = 0
        self.cur_dist = 0

    def step(self, action, episode):
        obs = super().step(action=action, episode=episode)

        return obs

    def reset(self, episode):
        super().reset(episode)

        # Pick a random goal in the robot's workspace
        allowed_space = (
            self._config.EE_SAMPLE_FACTOR
            * self._sim.robot.params.ee_constraint
        )

        self._desired_resting = np.random.uniform(
            low=allowed_space[:, 0], high=allowed_space[:, 1]
        )

        if self._config.RENDER_TARGET:
            global_pos = self._sim.robot.base_transformation.transform_point(
                self._desired_resting
            )
            self._sim.viz_ids["reach_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["reach_target"]
            )

        return super(RearrangeReachTaskV1, self).reset(episode)
