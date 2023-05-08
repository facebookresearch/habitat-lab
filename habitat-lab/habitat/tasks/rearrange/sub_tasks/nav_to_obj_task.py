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
from habitat.robots.stretch_robot import StretchRobot
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger, get_robot_spawns


@dataclass
class NavToInfo:
    """
    :property nav_goal_pos: Where the robot should navigate to. This is likely
    on a receptacle and not a navigable position.
    """

    nav_goal_pos: np.ndarray
    robot_start_pos: np.ndarray
    robot_start_angle: float
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
            should_place_robot=False,
            **kwargs,
        )
        self.force_obj_to_idx = None
        self.force_recep_to_name = None

        self._nav_to_info = None
        self.robot_start_position = None
        self.robot_start_rotation = None

        self._min_start_distance = self._config.min_start_distance
        self._goal_type = config.goal_type
        self._pick_init = config.pick_init
        self._place_init = config.place_init

        assert not (self._pick_init and self._place_init), "Can init near either pick or place."


    @property
    def nav_goal_pos(self):
        return self._nav_to_info.nav_goal_pos

    @property
    def is_nav_to_obj(self):
        return self._config.object_in_hand_sample_prob == 0

    @property
    def should_end(self) -> bool:
        return self._should_end or self.actions["rearrange_stop"].does_want_terminate

    def set_args(self, obj, **kwargs):
        self.force_obj_to_idx = obj
        self.force_kwargs = kwargs
        if "marker" in kwargs:
            self.force_recep_to_name = kwargs["marker"]

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the robot hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_to_pos(
        self, episode, start_hold_obj_idx=None, force_idx=None
    ):

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
        return nav_to_pos

    def _generate_nav_start_goal(self, episode, nav_to_pos, start_hold_obj_idx=None) -> NavToInfo:
        """
        Returns the starting information for a navigate to object task.
        """


        def filter_func(start_pos, _):
            if len(nav_to_pos.shape) == 1:
                goals = np.expand_dims(nav_to_pos, axis=0)
            else:
                goals = nav_to_pos
            distance = self._sim.geodesic_distance(start_pos, goals, episode)
            return (
                distance != np.inf
                and distance > self._min_start_distance
            )

        robot_pos, robot_angle = self._sim.set_robot_base_to_random_point(
            filter_func=filter_func
        )

        return NavToInfo(
            nav_goal_pos=nav_to_pos,
            robot_start_pos=robot_pos,
            robot_start_angle=robot_angle,
            start_hold_obj_idx=start_hold_obj_idx,
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode, fetch_observations=False)

        # in the case of Stretch, force the agent to look down and retract arm with the gripper pointing downwards
        if isinstance(sim.robot, StretchRobot):
            camera_rot = -1.7375 if self._pick_init or self._place_init else 0.0
            sim.robot.arm_motor_pos = np.array(
                [0.0] * 4 + [0.775, 0.0, -1.57000005, 0.0, camera_rot, -0.7125]
            )
            sim.robot.arm_joint_pos = np.array(
                [0.0] * 4 + [0.775, 0.0, -1.57000005, 0.0, camera_rot, -0.7125]
            )

        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        if (
            self.force_obj_to_idx is None
            and random.random() < self._config.object_in_hand_sample_prob
            and self._goal_type != 'ovmm'
        ):
            start_hold_obj_idx = self._generate_snap_to_obj()

        nav_to_pos = self._generate_nav_to_pos(
            episode, start_hold_obj_idx=start_hold_obj_idx, force_idx=self.force_obj_to_idx
        )

        self._nav_to_info = self._generate_nav_start_goal(
            episode, nav_to_pos, start_hold_obj_idx=start_hold_obj_idx
        )
        if self._pick_init:
            spawn_recs = [
                sim.receptacles[
                    episode.name_to_receptacle[
                        list(sim.instance_handle_to_ref_handle.keys())[0]
                    ]
                ]
            ]
            snap_pos = np.array(
                [
                    r.get_surface_center(sim)
                    for r in spawn_recs
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
            sim.robot.base_pos = start_pos
            # in the case of Stretch, rotate base so that the arm faces the target location
            if isinstance(self._sim.robot, StretchRobot):
                sim.robot.base_rot = angle_to_obj + np.pi / 2
            else:
                sim.robot.base_rot = angle_to_obj
        elif self._place_init:

            if isinstance(sim.robot, StretchRobot):
                sim.robot.arm_motor_pos = np.array(
                    [0.0] * 4 + [0.775, 0.0, 0.0, 0.0, 0.0, -0.7125] # gripper straight out
                )
                sim.robot.arm_joint_pos = np.array(
                    [0.0] * 4 + [0.775, 0.0, 0.0, 0.0, 0.0, -0.7125]
                )

            spawn_recs = [
                sim.receptacles[r]
                for r in sim.receptacles.keys()
                if r in sim.valid_goal_rec_names]
            
            snap_pos = np.array(
                [
                    r.get_surface_center(sim)
                    for r in spawn_recs
                ]
            )
            start_pos, angle_to_obj, was_unsucc = get_robot_spawns(
                target_positions=snap_pos,
                rotation_perturbation_noise=self._config.base_angle_noise,
                distance_threshold=self._config.spawn_max_dists_to_obj,
                sim=sim,
                num_spawn_attempts=self._config.num_spawn_attempts,
                physics_stability_steps=self._config.physics_stability_steps,
            )

            if was_unsucc:
                rearrange_logger.error(
                    f"Episode {episode.episode_id} failed to place robot"
                )

            sim.robot.base_pos = start_pos
            sim.robot.base_rot = angle_to_obj
        else:
            sim.robot.base_pos = self._nav_to_info.robot_start_pos
            sim.robot.base_rot = self._nav_to_info.robot_start_angle
        self.robot_start_position = sim.robot.sim_obj.translation
        start_quat = sim.robot.sim_obj.rotation
        self.robot_start_rotation = np.array(
            [
                start_quat.vector.x,
                start_quat.vector.y,
                start_quat.vector.z,
                start_quat.scalar,
            ]
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
        self._sim.maybe_update_robot()
        return self._get_observations(episode)
