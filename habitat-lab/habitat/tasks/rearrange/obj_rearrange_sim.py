#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING, Dict)
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
if TYPE_CHECKING:
    from omegaconf import DictConfig

@registry.register_simulator(name="ObjectRearrangeSim-v0")
class ObjectRearrangeSim(RearrangeSim):
    def __init__(self, config: "DictConfig"):
        super().__init__(config)
        self._receptacle_semantic_ids: Dict[int, int] = {}

    def _setup_targets(self, ep_info):
        super()._setup_targets(ep_info)
        self.target_categories = {}
        self.target_categories["goal_recep"] = ep_info.goal_recep_category
        self.target_categories["start_recep"] = ep_info.start_recep_category
        self.valid_goal_rec_obj_ids = {
            int(g.object_id) for g in ep_info.candidate_goal_receps
        }

        self.valid_goal_rec_names = [
            g.object_name for g in ep_info.candidate_goal_receps
        ]

    @property
    def receptacle_semantic_ids(self):
        return self._receptacle_semantic_ids

    def reconfigure(self, config: "DictConfig", ep_info: RearrangeEpisode):
        super().reconfigure(config, ep_info)
        self._cache_receptacles()

    def _cache_receptacles(self):
        # TODO: potentially this is slow, get receptacle list from episode instead
        rom = self.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)
            user_attr_keys = obj.user_attributes.get_subconfig_keys()
            if any(key.startswith("receptacle_") for key in user_attr_keys):
                self._receptacle_semantic_ids[
                    obj.object_id
                ] = obj.creation_attributes.semantic_id