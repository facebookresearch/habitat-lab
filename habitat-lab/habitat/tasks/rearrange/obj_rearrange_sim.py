#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim


@registry.register_simulator(name="ObjectRearrangeSim-v0")
class ObjectRearrangeSim(RearrangeSim):
    def _setup_targets(self, ep_info):
        super()._setup_targets(ep_info)
        self.target_categories = {}
        self.target_categories["goal_recep"] = ep_info.goal_recep_category
        self.target_categories["start_recep"] = ep_info.start_recep_category
