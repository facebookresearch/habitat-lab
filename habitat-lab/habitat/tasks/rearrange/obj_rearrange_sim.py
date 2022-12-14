#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from typing import Tuple
import numpy as np


@registry.register_simulator(name="ObjectRearrangeSim-v0")
class ObjectRearrangeSim(RearrangeSim):
    # def get_targets(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Get a mapping of object ids to goal positions for rearrange targets.

    #     :return: ([idx: int], [goal_pos: list]) The index of the target object
    #       in self.scene_obj_ids and the 3D goal POSITION, rotation is IGNORED.
    #       Note that goal_pos is the desired position of the object, not the
    #       starting position.
    #     """
    #     if self.ep_info['candidate_objects'] is not None:
    #         target_ids = []
    #         target_trans = []
    #         for goal in self.ep_info['candidate_objects']:
    #             target_ids.append(goal.object_id)
    #             target_trans.append(goal.position)
    #         return target_ids, target_trans
    #     else:
    #         return super().get_targets()

    def _setup_targets(self):
        super()._setup_targets()
        self.target_categories = {}
        if "goal_recep_category" in self.ep_info:
            self.target_categories["goal_recep"] = self.ep_info[
                "goal_recep_category"
            ]
            self.target_categories["start_recep"] = self.ep_info[
                "start_recep_category"
            ]
