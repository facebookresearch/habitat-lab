#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim

@registry.register_simulator(name="OVMMSim-v0")
class OVMMSim(RearrangeSim):
    def _setup_targets(self, ep_info):
        super()._setup_targets(ep_info)
        self.valid_goal_rec_obj_ids = {
            int(g.object_id) for g in ep_info.candidate_goal_receps
        }

        self.valid_goal_rec_names = [
            g.object_name for g in ep_info.candidate_goal_receps
        ]
