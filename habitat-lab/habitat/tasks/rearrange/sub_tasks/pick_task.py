#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.robots.stretch_robot import StretchRobot
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import get_robot_spawns, rearrange_logger
import habitat.robots.stretch_robot as robot_class


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
            should_place_robot=False,
            **kwargs,
        )

        self.prev_colls = None
        self.force_set_idx = None

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

    def _get_spawn_recs(self, sim, episode):
        return [
            sim.receptacles[
                episode.name_to_receptacle[
                    list(sim.instance_handle_to_ref_handle.keys())[0]
                ]
            ]
        ]

    def _gen_start_pos(self, sim, episode, sel_idx):
        if self._config.biased_init:
            target_positions = self._get_targ_pos(sim)
            snap_pos = np.expand_dims(target_positions[sel_idx], axis=0)
        else:
            snap_pos = np.array(
                [
                    r.get_surface_center(sim)
                    for r in self._get_spawn_recs(sim, episode)
                ]
            )

        start_pos, angle_to_obj, was_unsucc = get_robot_spawns(
            snap_pos,
            self._config.base_angle_noise,
            self._config.spawn_max_dists_to_obj,
            sim,
            self._config.num_spawn_attempts,
            self._config.physics_stability_steps,
        )

        if was_unsucc:
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

    def reset(self, episode: Episode, fetch_observations: bool = True):
        sim = self._sim

        assert isinstance(
            episode, RearrangeEpisode
        ), "Provided episode needs to be of type RearrangeEpisode for RearrangePickTaskV1"

        super().reset(episode, fetch_observations=False)

        self.prev_colls = 0

        sel_idx = self._sample_idx(sim)
        # in the case of Stretch, force the agent to look down and retract arm with the gripper pointing downwards
        if isinstance(sim.robot, StretchRobot):
            sim.robot.arm_motor_pos = np.array(
                [0.0] * 4 + [0.775, 0.0, -1.57000005, 0.0, -1.7375, -0.7125]
            )
            sim.robot.arm_joint_pos = np.array(
                [0.0] * 4 + [0.775, 0.0, -1.57000005, 0.0, -1.7375, -0.7125]
            )
        start_pos, start_rot = self._gen_start_pos(sim, episode, sel_idx)

        sim.robot.base_pos = start_pos
        # in the case of Stretch, rotate base so that the arm faces the target location
        if isinstance(self._sim.robot, StretchRobot):
            sim.robot.base_rot = start_rot + np.pi / 2
        else:
            sim.robot.base_rot = start_rot

        self._targ_idx = sel_idx

        if fetch_observations:
            self._sim.maybe_update_robot()
            return self._get_observations(episode)
        return None
