#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger


@dataclass
class NavToInfo:
    """
    :property nav_goal_pos: Where the articulated_agent should navigate to. This is likely
    on a receptacle and not a navigable position.
    """

    nav_goal_pos: np.ndarray
    articulated_agent_start_pos: np.ndarray
    articulated_agent_start_angle: float
    start_hold_obj_idx: Optional[int]


@registry.register_task(name="NavToObjTask-v0")
class DynNavRLEnv(RearrangeTask):
    """
    :property _nav_to_info: Information about the next skill we are navigating to.
    """

    _nav_to_info: Optional[NavToInfo]

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=False,
            **kwargs,
        )
        self.force_obj_to_idx = None
        self.force_recep_to_name = None

        # Set config options
        self._object_in_hand_sample_prob = (
            self._config.object_in_hand_sample_prob
        )
        self._min_start_distance = self._config.min_start_distance

        self._nav_to_info = None

    @property
    def nav_goal_pos(self):
        return self._nav_to_info.nav_goal_pos

    def set_args(self, obj, **kwargs):
        self.force_obj_to_idx = obj
        self.force_kwargs = kwargs
        if "marker" in kwargs:
            self.force_recep_to_name = kwargs["marker"]

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the articulated_agent hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_start_goal(self, episode, force_idx=None) -> NavToInfo:
        """
        Returns the starting information for a navigate to object task.
        """

        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        if (
            force_idx is None
            and random.random() < self._object_in_hand_sample_prob
        ):
            start_hold_obj_idx = self._generate_snap_to_obj()

        if start_hold_obj_idx is None:
            # Select an object at random and navigate to that object.
            all_pos = self._sim.get_target_objs_start()
            if force_idx is None:
                nav_to_pos = all_pos[np.random.randint(0, len(all_pos))]
            else:
                nav_to_pos = all_pos[force_idx]
        else:
            # Select a goal at random and navigate to that goal.
            _, all_pos = self._sim.get_targets()
            nav_to_pos = all_pos[np.random.randint(0, len(all_pos))]

        def filter_func(start_pos, _):
            return (
                np.linalg.norm(start_pos - nav_to_pos)
                > self._min_start_distance
            )

        (
            articulated_agent_pos,
            articulated_agent_angle,
        ) = self._sim.set_articulated_agent_base_to_random_point(
            filter_func=filter_func
        )

        return NavToInfo(
            nav_goal_pos=nav_to_pos,
            articulated_agent_start_pos=articulated_agent_pos,
            articulated_agent_start_angle=articulated_agent_angle,
            start_hold_obj_idx=start_hold_obj_idx,
        )

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)

        self._nav_to_info = self._generate_nav_start_goal(
            episode, force_idx=self.force_obj_to_idx
        )

        self._sim.articulated_agent.base_pos = (
            self._nav_to_info.articulated_agent_start_pos
        )
        self._sim.articulated_agent.base_rot = (
            self._nav_to_info.articulated_agent_start_angle
        )

        if self._nav_to_info.start_hold_obj_idx is not None:
            if self._sim.grasp_mgr.is_grasped:
                raise ValueError(
                    f"Attempting to grasp {self._nav_to_info.start_hold_obj_idx} even though object is already grasped"
                )
            rearrange_logger.debug(
                f"Forcing to grasp object {self._nav_to_info.start_hold_obj_idx}"
            )
            self._sim.grasp_mgr.snap_to_obj(
                self._nav_to_info.start_hold_obj_idx, force=True
            )

        if self._sim.habitat_config.debug_render:
            # Visualize the position the agent is navigating to.
            self._sim.viz_ids["nav_targ_pos"] = self._sim.visualize_position(
                self._nav_to_info.nav_goal_pos,
                self._sim.viz_ids["nav_targ_pos"],
                r=0.2,
            )
        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)
