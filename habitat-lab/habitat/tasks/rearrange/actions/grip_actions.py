#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import magnum as mn
import numpy as np
from gym import spaces

from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.articulated_agent_action import (
    ArticulatedAgentAction,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    coll_link_name_matches,
    coll_name_matches,
    get_camera_object_angle,
    get_camera_transform,
)


class GripSimulatorTaskAction(ArticulatedAgentAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    @property
    def requires_action(self):
        return self.action_space is not None


@registry.register_task_action
class MagicGraspAction(GripSimulatorTaskAction):
    @property
    def action_space(self):
        return spaces.Box(shape=(1,), high=1.0, low=-1.0)

    def _grasp(self):
        scene_obj_pos = self._sim.get_scene_pos()
        ee_pos = self.cur_articulated_agent.ee_transform().translation
        # Get objects we are close to.
        if len(scene_obj_pos) != 0:
            # Get the target the EE is closest to.
            closest_obj_idx = np.argmin(
                np.linalg.norm(scene_obj_pos - ee_pos, ord=2, axis=-1)
            )

            to_target = np.linalg.norm(
                ee_pos - scene_obj_pos[closest_obj_idx], ord=2
            )

            keep_T = mn.Matrix4.translation(mn.Vector3(0.1, 0.0, 0.0))

            if to_target < self._config.grasp_thresh_dist:
                self.cur_grasp_mgr.snap_to_obj(
                    self._sim.scene_obj_ids[closest_obj_idx],
                    force=False,
                    rel_pos=mn.Vector3(0.1, 0.0, 0.0),
                    keep_T=keep_T,
                )
                return

        # Get markers we are close to.
        markers = self._sim.get_all_markers()
        if len(markers) > 0:
            names = list(markers.keys())
            pos = np.array([markers[k].get_current_position() for k in names])

            closest_idx = np.argmin(
                np.linalg.norm(pos - ee_pos, ord=2, axis=-1)
            )

            to_target = np.linalg.norm(ee_pos - pos[closest_idx], ord=2)

            if to_target < self._config.grasp_thresh_dist:
                self.cur_articulated_agent.open_gripper()
                self.cur_grasp_mgr.snap_to_marker(names[closest_idx])

    def _ungrasp(self):
        self.cur_grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):
        if grip_action is None:
            return

        if grip_action >= 0 and not self.cur_grasp_mgr.is_grasped:
            self._grasp()
        elif grip_action < 0 and self.cur_grasp_mgr.is_grasped:
            self._ungrasp()


@registry.register_task_action
class SuctionGraspAction(MagicGraspAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim

    def _grasp(self):
        attempt_snap_entity: Optional[Union[str, int]] = None
        match_coll = None
        contacts = self._sim.get_physics_contact_points()

        # TODO: the two arguments below should be part of args
        ee_index = 0
        index_grasp_manager = 0

        robot_id = self._sim.articulated_agent.sim_obj.object_id
        all_gripper_links = list(
            self._sim.articulated_agent.params.gripper_joints
        )
        robot_contacts = [
            c
            for c in contacts
            if coll_name_matches(c, robot_id)
            and any(coll_link_name_matches(c, l) for l in all_gripper_links)
        ]

        if len(robot_contacts) == 0:
            return

        # Contacted any objects?
        for scene_obj_id in self._sim.scene_obj_ids:
            for c in robot_contacts:
                if coll_name_matches(c, scene_obj_id):
                    match_coll = c
                    break
            if match_coll is not None:
                attempt_snap_entity = scene_obj_id
                break

        if attempt_snap_entity is not None:
            rom = self._sim.get_rigid_object_manager()
            ro = rom.get_object_by_id(attempt_snap_entity)

            ee_T = self.cur_articulated_agent.ee_transform()
            obj_in_ee_T = ee_T.inverted() @ ro.transformation

            # here we need the link T, not the EE T for the constraint frame
            ee_link_T = self.cur_articulated_agent.sim_obj.get_link_scene_node(
                self.cur_articulated_agent.params.ee_links[ee_index]
            ).absolute_transformation()

            self._sim.grasp_mgr.snap_to_obj(
                int(attempt_snap_entity),
                force=False,
                # rel_pos is the relative position of the object COM in link space
                rel_pos=ee_link_T.inverted().transform_point(ro.translation),
                keep_T=obj_in_ee_T,
                should_open_gripper=False,
            )
            return

        # Contacted any markers?
        markers = self._sim.get_all_markers()
        for marker_name, marker in markers.items():
            has_match = any(
                c
                for c in robot_contacts
                if coll_name_matches(c, marker.ao_parent.object_id)
                and coll_link_name_matches(c, marker.link_id)
            )
            if has_match:
                attempt_snap_entity = marker_name

        if attempt_snap_entity is not None:
            self._sim.grasp_mgrs[index_grasp_manager].snap_to_marker(
                str(attempt_snap_entity)
            )


@registry.register_task_action
class GazeGraspAction(MagicGraspAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.min_dist, self.max_dist = config.gaze_distance_range
        self.center_cone_angle_threshold = np.deg2rad(
            config.center_cone_angle_threshold
        )
        self.center_cone_vector = mn.Vector3(
            config.center_cone_vector
        ).normalized()
        self.auto_grasp = config.auto_grasp

    @property
    def action_space(self):
        if self.auto_grasp:
            return None
        else:
            return spaces.Box(shape=(1,), high=1.0, low=-1.0)

    def _get_camera_object_angle(self, obj_pos):
        """Calculates angle between gripper line-of-sight and given global position."""
        # Get the camera transformation
        cam_T = get_camera_transform(self.cur_articulated_agent)

        # Get angle between (normalized) location and the vector that the camera should
        # look at
        obj_angle = get_camera_object_angle(
            cam_T, obj_pos, self.center_cone_vector
        )

        return obj_angle

    def _determine_center_object(self):
        """Determine if an object is at the center of the frame and in range"""
        if isinstance(self.cur_articulated_agent, SpotRobot):
            cam_pos = (
                self._sim.agents[0]
                .get_state()
                .sensor_states["articulated_agent_arm_rgb"]
                .position
            )
        elif isinstance(self.cur_articulated_agent, StretchRobot):
            cam_pos = (
                self._sim.agents[0]
                .get_state()
                .sensor_states["head_rgb"]
                .position
            )
        else:
            raise NotImplementedError(
                "This robot does not have GazeGraspAction."
            )

        if isinstance(self.cur_articulated_agent, SpotRobot):
            panoptic_img = self._sim._sensor_suite.get_observations(
                self._sim.get_sensor_observations()
            )["articulated_agent_arm_panoptic"]
        elif isinstance(self.cur_articulated_agent, StretchRobot):
            panoptic_img = self._sim._sensor_suite.get_observations(
                self._sim.get_sensor_observations()
            )["head_panoptic"]
        else:
            raise NotImplementedError(
                "This robot does not have GazeGraspAction."
            )

        height, width = panoptic_img.shape[:2]
        # Note that panoptic_img is a 3D array
        center_obj_id = (
            panoptic_img[height // 2, width // 2, 0]
            - self._sim.habitat_config.object_ids_start
        )

        rom = self._sim.get_rigid_object_manager()
        if center_obj_id in self._sim.scene_obj_ids:
            obj_pos = rom.get_object_by_id(center_obj_id).translation

            # Skip if not in distance range
            dist = np.linalg.norm(obj_pos - cam_pos)
            if dist < self.min_dist or dist > self.max_dist:
                return None, None

            # Skip if not in the central cone
            obj_angle = self._get_camera_object_angle(obj_pos)
            if abs(obj_angle) > self.center_cone_angle_threshold:
                return None, None

            return center_obj_id, obj_pos

        return None, None

    def _grasp(self):
        # Check if the object is in the center of the camera
        center_obj_idx, center_obj_pos = self._determine_center_object()

        # If there is nothing to grasp, then we return
        if center_obj_idx is None:
            return

        keep_T = mn.Matrix4.translation(mn.Vector3(0.1, 0.0, 0.0))

        self.cur_grasp_mgr.snap_to_obj(
            center_obj_idx,
            force=True,
            rel_pos=mn.Vector3(0.1, 0.0, 0.0),
            keep_T=keep_T,
        )
        return

    def _ungrasp(self):
        self.cur_grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):
        if self.auto_grasp and not self.cur_grasp_mgr.is_grasped:
            self._grasp()
            return

        if grip_action is None:
            return

        if grip_action >= 0 and not self.cur_grasp_mgr.is_grasped:
            self._grasp()
        elif grip_action < 0 and self.cur_grasp_mgr.is_grasped:
            self._ungrasp()
