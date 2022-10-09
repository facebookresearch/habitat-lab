#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_logger


@dataclass
class NavToInfo:
    """
    :property nav_target_pos: Where the robot should navigate to.
    :property nav_target_angle: What angle the robot should be at when at the goal.
    :property nav_to_task_name: The name of the sub-task we are navigating to.
    """

    nav_target_pos: mn.Vector3
    nav_target_angle: float
    nav_to_task_name: str
    nav_to_entity_name: str
    start_hold_obj_idx: Optional[int] = None
    start_base_pos: Optional[mn.Vector3] = None
    start_base_rot: Optional[float] = None


@registry.register_task(name="LangNavToObjTask-v0")
class LangDynNavRLEnv(RearrangeTask):
    """
    :property _nav_to_info: Information about the next skill we are navigating to.
    """

    _nav_to_info: Optional[NavToInfo]

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.force_obj_to_idx = None
        self.force_recep_to_name = None
        self._prev_measure = 1.0

        data_path = dataset.config.DATA_PATH.format(split=dataset.config.SPLIT)
        fname = data_path.split("/")[-1].split(".")[0]
        save_dir = osp.dirname(data_path)
        self.cache = CacheHelper(
            osp.join(save_dir, f"{fname}_{config.TYPE}_start.pickle"),
            def_val={},
            verbose=False,
        )
        self.start_states = self.cache.load()

        self._nav_to_info = None

    @property
    def nav_target_pos(self):
        return [goal.position for goal in self.goals]

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the robot hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_start_goal(self, episode) -> NavToInfo:
        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        if random.random() < self._config.OBJECT_IN_HAND_SAMPLE_PROB:
            start_hold_obj_idx = self._generate_snap_to_obj()

        target_pos = np.random.choice(self.goals).position
        start_pos, start_rot = get_robo_start_pos(self._sim, target_pos)

        # TODO: update fields for ObjectRearrange - support multiple instances
        return NavToInfo(
            nav_target_pos=None,
            nav_target_angle=None,
            nav_to_task_name=None,
            nav_to_entity_name=None,
            start_hold_obj_idx=start_hold_obj_idx,
            start_base_pos=start_pos,
            start_base_rot=start_rot,
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode, fetch_observations=False)
        rearrange_logger.debug("Resetting navigation task")

        episode_id = sim.ep_info["episode_id"]
        self.goals = episode.candidate_start_receps

        # Rest the nav to information for this episode.
        self._nav_to_info = None
        if self.force_obj_to_idx is not None:
            # TODO
            raise NotImplementedError
        else:
            if (
                episode_id in self.start_states
                and not self._config.FORCE_REGENERATE
            ):
                self._nav_to_info = self.start_states[episode_id]

                if (
                    not isinstance(self._nav_to_info, NavToInfo)
                    or self._nav_to_info.start_base_pos is None
                    or self._nav_to_info.start_base_rot is None
                ):
                    rearrange_logger.warning(
                        f"Incorrect cache saved to file {self._nav_to_info}. Regenerating now."
                    )
                    self._nav_to_info = None
                else:
                    rearrange_logger.debug(
                        f"Loaded episode from cache {self.cache.cache_id}."
                    )

            if (
                self._nav_to_info is not None
                and self._nav_to_info.start_hold_obj_idx is not None
            ):
                # The object to hold was generated from stale object IDs.
                # Reselect a new object to hold.
                self._nav_to_info.start_hold_obj_idx = (
                    self._generate_snap_to_obj()
                )

            if self._nav_to_info is None:
                self._nav_to_info = self._generate_nav_start_goal(episode)
                self.start_states[episode_id] = self._nav_to_info
                if self._config.SHOULD_SAVE_TO_CACHE:
                    self.cache.save(self.start_states)
                    rearrange_logger.debug(
                        f"Saved episode to cache {self.cache.cache_id}."
                    )
            sim.robot.base_pos = self._nav_to_info.start_base_pos
            sim.robot.base_rot = self._nav_to_info.start_base_rot
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

        rearrange_logger.debug(f"Got nav to info {self._nav_to_info}")

        # TODO: add navigability check for goal and debug visualization
        return self._get_observations(episode)


def get_robo_start_pos(
    sim, nav_targ_pos: mn.Vector3
) -> Tuple[np.ndarray, float]:
    orig_state = sim.capture_state()

    start_pos, start_rot = sim.set_robot_base_to_random_point(
        max_attempts=1000
    )

    # Reset everything except for the robot state.
    orig_state["robot_T"] = None
    sim.set_state(orig_state)
    return start_pos, start_rot
