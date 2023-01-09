#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


@registry.register_simulator(name="ObjectRearrangeSim-v0")
class ObjectRearrangeSim(RearrangeSim):
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
        self.valid_goal_rec_obj_ids = {
            int(g.object_id) for g in self.ep_info["candidate_goal_receps"]
        }

        self.valid_goal_rec_names = [
            g.object_name for g in self.ep_info["candidate_goal_receps"]
        ]
