#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import cv2
import magnum as mn
import numpy as np
from gym import spaces

import habitat
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.robot_action import RobotAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    coll_link_name_matches,
    coll_name_matches,
)


class GripSimulatorTaskAction(RobotAction):
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
        ee_pos = self.cur_robot.ee_transform.translation
        # print("scene_obj_pos:", scene_obj_pos, ee_pos)
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
            if to_target < self._config.GRASP_THRESH_DIST:
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

            if to_target < self._config.GRASP_THRESH_DIST:
                self.cur_robot.open_gripper()
                self.cur_grasp_mgr.snap_to_marker(names[closest_idx])

    def _ungrasp(self):
        self.cur_grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):

        # Force to grip each time
        # grip_action = 1
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

        robot_id = self._sim.robot.sim_obj.object_id
        all_gripper_links = list(self._sim.robot.params.gripper_joints)
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

            ee_T = self.cur_robot.ee_transform
            obj_in_ee_T = ee_T.inverted() @ ro.transformation

            # here we need the link T, not the EE T for the constraint frame
            ee_link_T = self.cur_robot.sim_obj.get_link_scene_node(
                self.cur_robot.params.ee_link
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
            self._sim.grasp_mgr.snap_to_marker(str(attempt_snap_entity))


@registry.register_task_action
class GazeGraspAction(GripSimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.min_dist, self.max_dist = config.GAZE_DISTANCE_RANGE
        self.central_cone = np.deg2rad(config.CENTER_CONE_ANGLE)
        self.snap_markers = config.get("GRASP_MARKERS", False)

        # Auto grasp logic
        self.auto_grasp = config.get("AUTO_GAZE_GRASP", False)
        self.in_center_needed = config.get("AUTO_GAZE_GRASP_NEEDED", 5)
        self.center_tolerance = config.get(
            "AUTO_GAZE_GRASP_TOL", 0.1
        )  # Not important
        self.in_center_count = 0

    @property
    def action_space(self):
        # Change to continous space
        return spaces.Box(shape=(1,), high=1.0, low=-1.0)

    def angle_between(self, v1, v2):
        # stack overflow question ID: 2827393
        cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
        object_angle = np.arccos(cosine)

        return object_angle

    def get_grasp_object_angle(self, obj_translation):
        """Calculates angle between gripper line-of-sight and given global position"""
        # breakpoint()
        camera_T_matrix = self.get_gripper_transform()

        # Get object location in camera frame
        camera_obj_trans = (
            camera_T_matrix.inverted()
            .transform_point(obj_translation)
            .normalized()
        )

        # Get angle between (normalized) location and unit vector
        object_angle = self.angle_between(
            camera_obj_trans, mn.Vector3(0, 0, -1)
        )

        return object_angle

    def get_gripper_transform(self):
        # link_rigid_state = self._sim.get_articulated_link_rigid_state(
        #     self.robot_id, self.ee_link
        # )

        # ee_trans = mn.Matrix4.from_(
        #     link_rigid_state.rotation.to_matrix(), link_rigid_state.translation
        # )

        ee_trans = self._sim.robot.ee_transform

        if isinstance(
            self._sim.robot, habitat.robots.spot_robot.SpotRobot
        ):  # self.robot_name == "hab_spot_arm":
            # Moves the camera in front of the gripper and up a bit
            offset_trans = mn.Matrix4.translation(
                mn.Vector3(0.15, 0.0, 0.025)
            )  # TODO: offset?
        else:
            # Moves the camera above the gripper
            offset_trans = mn.Matrix4.translation(mn.Vector3(0, 0.0, 0.1))
        arm_T = (
            ee_trans
            @ offset_trans
            @ mn.Matrix4.rotation(mn.Deg(-90), mn.Vector3(0.0, 1.0, 0.0))
            @ mn.Matrix4.rotation(mn.Deg(-90), mn.Vector3(0.0, 0.0, 1.0))
        )

        return arm_T

    def determine_center_object(self):
        arm_depth_state = self._sim.get_agent_state().sensor_states[
            "robot_arm_depth"
        ]
        arm_depth_cam_pos = arm_depth_state.position

        """
        Goals:
        - Determine if an object is at the center of the frame and in range
        - Get the center of the bbox of the target object (to save time? do later.)
        """
        trans = self._sim._get_raw_target_trans()
        targ_abs_obj_idx = trans[0][0]  # Get first target index
        rom = (
            self._sim.get_rigid_object_manager()
        )  # .get_object_by_id(79).translation

        for obj_idx, abs_obj_idx in enumerate(self._sim.scene_obj_ids):
            if targ_abs_obj_idx != abs_obj_idx:
                continue
            object_pos = rom.get_object_by_id(
                abs_obj_idx
            ).translation  # self._sim.get_translation(abs_obj_idx)
            # Skip if not in distance range
            dist = np.linalg.norm(object_pos - arm_depth_cam_pos)
            if dist >= 0.4:
                continue

            # Skip if not in the central cone
            # breakpoint()
            object_angle = self.get_grasp_object_angle(object_pos)

            # print('Dist: {}, Angle {}'.format(dist, object_angle))
            print(abs(object_angle), self.central_cone)
            if abs(object_angle) > self.central_cone:
                continue

            # breakpoint()
            # Now we can check if the object is blocking the center pixel
            abs_diff_denoised = self._sim.get_grasp_object_mask(abs_obj_idx)
            x, y, w, h = cv2.boundingRect(abs_diff_denoised)
            height, width = abs_diff_denoised.shape
            if (
                x <= width // 2
                and width // 2 <= x + w
                and y <= height // 2
                and height // 2 <= y + h
            ):
                # At this point, there should be an object at the center pixel
                cx, cy = [
                    (start + side_length / 2) / max_length
                    for start, side_length, max_length in [
                        (x, w, width),
                        (y, h, height),
                    ]
                ]
                return obj_idx, object_pos, cx, cy

        return None, None, None, None

    def _grasp(self, object_idx_pos=None):
        if object_idx_pos is None:
            obj_idx, object_pos, _, _ = self.determine_center_object()
        else:
            obj_idx, object_pos = object_idx_pos

        if obj_idx is None:
            # Nothing to grasp.
            return

        # Get transform from global to robot frame
        # link_state = self._sim.get_articulated_object_root_state(self._sim.robot_id)
        # link_T = mn.Matrix4.from_(link_state.rotation(), link_state.translation)
        link_T = (
            self._sim.robot.sim_obj.transformation
        )  # TODO: should be inverted?

        # breakpoint()
        if isinstance(
            self._sim.robot, habitat.robots.spot_robot.SpotRobot
        ):  # self._sim.robot_name == "hab_spot_arm":
            # Just retract the arm
            # for idx, angle in enumerate(self._sim.robot.spot_arm_init_params):
            #     joint_idx = self._sim.robot.arm_joints[idx] #self._sim.arm_start + idx
            #     self._sim.set_mtr_pos(joint_idx, angle)
            #     self._sim.set_joint_pos(joint_idx, angle)
            # self._sim.robot.reset() # TODO: only reset arm joints
            self._sim.robot.sim_obj.clear_joint_states()

        else:
            # Snap the gripper to the object if using Fetch
            joint_pos = self._sim.get_arm_pos()
            joint_vel = self._sim.get_arm_vel()
            self._sim._ik.set_arm_state(joint_pos, joint_vel)

            # Set EE target to a spot above the target object location
            local_object_pos = link_T.inverted().transform_point(
                object_pos + np.array([0.0, 0.05, 0.0])
            )
            des_joint_pos = list(self._sim._ik.calc_ik(local_object_pos))
            self._sim.set_arm_joint_pos(
                des_joint_pos,
                check_contact=False,
            )
            self._sim.null_step_world()

        # Grab the object
        # self._sim.set_snapped_obj(obj_idx)
        self._sim.grasp_mgr.snap_to_obj(self._sim.scene_obj_ids[obj_idx])

    # def step(self, grip_action, should_step=True, **kwargs):
    #     object_idx_pos = None
    #     if self.auto_grasp:
    #         # This block will override the state arg
    #         obj_idx, object_pos, cx, cy = self.determine_center_object()
    #         object_idx_pos = (obj_idx, object_pos)
    #         if obj_idx is None:
    #             # No object at gripper image center
    #             self.in_center_count = 0
    #         else:
    #             # See if object at center has its bbox centered within the overall image
    #             object_is_centered = all(
    #                 [abs(c - 0.5) < self.center_tolerance for c in [cx, cy]]
    #             )
    #             if object_is_centered:
    #                 self.in_center_count += 1
    #             else:
    #                 self.in_center_count = 0

    #         grip_action = 1 if self.in_center_count == self.in_center_needed else 0

    #     if grip_action == 1:
    #         self._grasp(object_idx_pos=object_idx_pos)
    #         #self._sim.set_gripper_state(0.04)
    #     else:
    #         #self._sim.grasp_mgr.desnap() # desnap_object()
    #         #self._sim.set_gripper_state(0.0)
    #         self._ungrasp()
    #     # if should_step:
    #     #     return self._sim.step(HabitatSimActions.GAZE_GRASP)

    def _ungrasp(self):
        self._sim.grasp_mgr.desnap()

    def step(self, grip_action, should_step=True, *args, **kwargs):
        if grip_action is None:
            return

        # Since we disable the grasp action, we want to see if
        # there is an object on the scene each time
        if not self._sim.grasp_mgr.is_grasped:
            self._grasp()

        # if grip_action >= 0 and not self._sim.grasp_mgr.is_grasped:
        #     self._grasp()
        # elif grip_action < 0 and self._sim.grasp_mgr.is_grasped:
        #     self._ungrasp()

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.in_center_count = 0
