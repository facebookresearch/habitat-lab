#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random
import magnum as mn
import numpy as np
import quaternion

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import (
    place_agent_at_dist_from_pos,
    rearrange_logger,
    set_agent_base_via_obj_trans,
)


@registry.register_task(name="RearrangePickTask-v0")
class RearrangePickTaskV1(RearrangeTask):
    DISTANCE_TO_RECEPTACLE = 1.0
    """
    Rearrange Pick Task with Fetch robot interacting with objects and environment.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            should_place_articulated_agent=False,
            **kwargs,
        )

        self.prev_colls = None
        self.force_set_idx = None
        self._base_angle_noise = self._config.base_angle_noise
        self._spawn_max_dist_to_obj = self._config.spawn_max_dist_to_obj
        self._num_spawn_attempts = self._config.num_spawn_attempts
        self._filter_colliding_states = self._config.filter_colliding_states
        self._spawn_max_dist_to_obj_delta = (
            self._config.spawn_max_dist_to_obj_delta
        )

    def set_args(self, obj, **kwargs):
        self.force_set_idx = obj

    def _get_targ_pos(self, sim):
        scene_pos = sim.get_scene_pos()
        targ_idxs = sim.get_targets()[0]
        return scene_pos[targ_idxs]

    def _sample_idx(self, sim):
        if self.force_set_idx is not None:
            idxs = self._sim.get_targets()[0]
            sel_idx = self.force_set_idx
            sel_idx = list(idxs).index(sel_idx)
        else:
            sel_idx = np.random.randint(0, len(self._get_targ_pos(sim)))
        return sel_idx

    def _gen_start_pos(self, sim, episode, sel_idx):
        target_positions = self._get_targ_pos(sim)
        targ_pos = target_positions[sel_idx]

        was_fail = True
        spawn_attempt_count = 0

        while was_fail and spawn_attempt_count < self._num_spawn_attempts:
            start_pos, angle_to_obj, was_fail = place_agent_at_dist_from_pos(
                targ_pos,
                self._base_angle_noise,
                self._spawn_max_dist_to_obj
                + spawn_attempt_count * self._spawn_max_dist_to_obj_delta,
                sim,
                self._num_spawn_attempts,
                self._filter_colliding_states,
            )
            spawn_attempt_count += 1

        if was_fail:
            rearrange_logger.error(
                f"Episode {episode.episode_id} failed to place robot"
            )

        return start_pos, angle_to_obj

    def _should_prevent_grip(self, action_args):
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
        )

    def step(self, action, episode):
        action_args = action["action_args"]

        if self._should_prevent_grip(action_args):
            # No releasing the object once it is held.
            action_args["grip_action"] = None
        obs = super().step(action=action, episode=episode)

        return obs

    def _set_arm_to_target_pose(self):
        """Set the arm to the target pose."""
        # Set the arm based on the random selection
        robot_init_arm_angle = np.copy(
            self._sim.get_agent_data(None).articulated_agent.arm_joint_pos
        )
        # Set the arm to a pose
        self._sim.get_agent_data(
            None
        ).articulated_agent.arm_joint_pos = np.array(
            random.choice(self._config.semantic_pick_target_arm_pose)
        )
        # Get the initial EE orientation at the time of begining of placing
        _, ee_orientation = self._sim.get_agent_data(
            None
        ).articulated_agent.get_ee_local_pose()  # type: ignore
        self.target_obj_orientation = quaternion.quaternion(ee_orientation)
        # Revert the robot arm to the initial pose
        self._sim.get_agent_data(
            None
        ).articulated_agent.arm_joint_pos = robot_init_arm_angle

    def reset(self, episode: Episode, fetch_observations: bool = True):
        sim = self._sim

        assert isinstance(
            episode, RearrangeEpisode
        ), "Provided episode needs to be of type RearrangeEpisode for RearrangePickTaskV1"

        super().reset(episode, fetch_observations=False)

        self.prev_colls = 0

        sel_idx = self._sample_idx(sim)
        sel_idx = 1
        # 1 -> white box
        # 0 -> spam can
        start_pos, start_rot = self._gen_start_pos(sim, episode, sel_idx)
        # Fix location
        start_pos = mn.Vector3(-2.6196303, 0.1766345, 3.9796443)
        start_rot = 1.0381504609239443

        print(f"sel_idx: {sel_idx}")
        print(f"start_pos: {start_pos}")
        print(f"start_rot: {start_rot}")

        set_agent_base_via_obj_trans(
            start_pos, start_rot, sim.articulated_agent
        )

        self._targ_idx = sel_idx

        # Set the arm to the target pose, and the revert it
        if self._config.semantic_pick_training:
            self._set_arm_to_target_pose()
        if self._config.topdown_side_training:
            self.grasping_type = np.random.choice(["topdown", "side"])

        if fetch_observations:
            self._sim.maybe_update_articulated_agent()
            return self._get_observations(episode)
        return None
