#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as osp
from typing import Dict

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv


@registry.register_task(name="OVMMNavToObjTask-v0")
class OVMMDynNavRLEnv(DynNavRLEnv):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self._receptacle_semantic_ids: Dict[int, int] = {}
        self._receptacle_categories: Dict[str, str] = {}
        self._recep_category_to_recep_category_id = (
            dataset.recep_category_to_recep_category_id
        )
        self._loaded_receptacle_categories = False
        if config.receptacle_categories_file is not None and osp.exists(
            config.receptacle_categories_file
        ):
            with open(config.receptacle_categories_file) as f:
                for line in f.readlines():
                    name, category = line.strip().split(",")
                    self._receptacle_categories[name] = category
            self._loaded_receptacle_categories = True

    @property
    def receptacle_semantic_ids(self):
        return self._receptacle_semantic_ids

    @property
    def loaded_receptacle_categories(self):
        return self._loaded_receptacle_categories

    def reset(self, episode: Episode):
        self._receptacle_semantic_ids = {}
        self._cache_receptacles()
        obs = super().reset(episode)
        self._nav_to_obj_goal = np.stack(
            [
                view_point.agent_state.position
                for goal in episode.candidate_objects
                for view_point in goal.view_points
            ],
            axis=0,
        )
        return obs

    def _cache_receptacles(self):
        # TODO: potentially this is slow, get receptacle list from episode instead
        rom = self._sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)
            user_attr_keys = obj.user_attributes.get_subconfig_keys()
            if any(key.startswith("receptacle_") for key in user_attr_keys):
                obj_name = osp.basename(obj.creation_attributes.handle).split(
                    ".", 1
                )[0]
                category = self._receptacle_categories.get(obj_name)
                if (
                    category is None
                    or category
                    not in self._recep_category_to_recep_category_id
                ):
                    continue
                category_id = self._recep_category_to_recep_category_id[
                    category
                ]
                self._receptacle_semantic_ids[obj.object_id] = category_id + 1

    def _generate_nav_to_pos(
        self, episode, start_hold_obj_idx=None, force_idx=None
    ):
        # learn nav to pick skill if not holding object currently
        if start_hold_obj_idx is None:
            # starting positions of candidate objects
            all_pos = np.stack(
                [
                    view_point.agent_state.position
                    for goal in episode.candidate_objects
                    for view_point in goal.view_points
                ],
                axis=0,
            )
            if force_idx is not None:
                raise NotImplementedError
        else:
            # positions of candidate goal receptacles
            all_pos = np.stack(
                [
                    view_point.agent_state.position
                    for goal in episode.candidate_goal_receps
                    for view_point in goal.view_points
                ],
                axis=0,
            )

        return all_pos
