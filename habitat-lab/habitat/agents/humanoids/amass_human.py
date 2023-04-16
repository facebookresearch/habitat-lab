# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import magnum as mn
import numpy as np

import habitat_sim.physics as phy
from habitat.agents.humanoids.human_base import Humanoid
from habitat.agents.robots.mobile_manipulator import (
    MobileManipulator,
    MobileManipulatorParams,
    RobotCameraParams,
)
from habitat_sim.logging import LoggingContext, logger


@dataclass
class HumanParams:
    arm_init_params_left: Optional[np.ndarray]
    arm_init_params_right: Optional[np.ndarray]
    arm_init_params: Optional[np.ndarray]
    cameras: Dict
    gripper_init_params: Optional[np.ndarray]
    ee_link_left: int
    ee_link_right: int
    ee_offset: mn.Vector3
    gripper_joints: List[int]
    arm_joints_right: List[int]
    arm_joints_left: List[int]
    base_link_names: Set[str]
    base_offset: mn.Vector3


class AmassHuman(Humanoid):
    def __init__(self, urdf_path, sim):
        self.params = self._get_human_params()

        super().__init__(self.params, urdf_path, sim)

        self.urdf_path = urdf_path
        self.sim = sim
        self.ROOT = 0
        self.arm_joint_pos_left = self.params.arm_init_params_left
        self.arm_joint_pos_right = self.params.arm_init_params_right
        self.all_joints = None

    def _get_human_params(self):
        return HumanParams(
            gripper_joints=[0, 0],
            base_offset=mn.Vector3([0, 1.1, 0]),
            gripper_init_params=np.array([0.00, 0.00], dtype=np.float32),
            arm_init_params_left=np.array(
                [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                dtype=np.float32,
            ),
            arm_init_params_right=np.array(
                [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                dtype=np.float32,
            ),
            arm_init_params=np.array(
                [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005],
                dtype=np.float32,
            ),
            ee_offset=mn.Vector3(),
            ee_link_right=15,
            ee_link_left=9,
            arm_joints_right=[11, 12, 13],
            arm_joints_left=[15, 16, 17],
            cameras={
                "robot_arm": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                    attached_link_id=22,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(90)),
                ),
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.5, 0.25),
                    cam_look_at_pos=mn.Vector3(0, 0.5, 0.75),
                    attached_link_id=-1,
                ),
                "robot_third": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(-1.2, 1.5, -1.2),
                    cam_look_at_pos=mn.Vector3(1, 0.3, 0.75),
                    attached_link_id=-1,
                ),
            },
            base_link_names={"base_link"},
        )

    def reset_path_info(self):
        self.path_ind = 0
        self.path_distance_walked = 0
        self.path_distance_covered_next_wp = 0

    def close_gripper(self):
        pass

    def reconfigure(self) -> None:
        """Instantiates the human in the scene. Loads the URDF (TODO: complete descr)"""
        # Manipulator.reconfigure(self)
        # RobotBase.reconfigure(self)
        super().reconfigure()
        self.sim_obj.motion_type = phy.MotionType.KINEMATIC
        # self.translation_offset = self.sim_obj.translation + mn.Vector3([0,0.90,0])

    @property
    def arm_joint_pos(self):
        """Get the current arm joint positions."""

        # deref self vars to cut access in half
        joint_pos_indices = self.joint_pos_indices
        arm_joints = self.params.arm_joints_right
        sim_obj_joint_pos = self.sim_obj.joint_positions

        arm_pos_indices = (joint_pos_indices[x] for x in arm_joints)
        return np.array(
            [sim_obj_joint_pos[i] for i in arm_pos_indices], dtype=np.float32
        )

    @arm_joint_pos.setter
    def arm_joint_pos(self, ctrl: List[float]):
        """Kinematically sets the arm joints and sets the motors to target."""
        # self._validate_arm_ctrl_input(ctrl)
        # breakpoint()
        joint_positions = self.sim_obj.joint_positions

        # breakpoint
        for i, jidx in enumerate(self.params.arm_joints_right):
            # self._set_motor_pos(jidx, ctrl[i])
            joint_positions[self.joint_pos_indices[jidx]] = ctrl[i]
        # self.sim_obj.joint_positions = joint_positions
        # breakpoint()
        self.sim_obj.joint_positions = np.array(ctrl)
        for joint_id in range(len(self.joint_motors)):
            ctrl_quat_id = joint_id * 4
            quat = ctrl[ctrl_quat_id : (ctrl_quat_id + 4)]
            # breakpoint()
            if joint_id in self.joint_motors:
                self.joint_motors[joint_id][
                    1
                ].spherical_position_target = mn.Quaternion(
                    mn.Vector3(quat[:3]), quat[-1]
                )
                self.sim_obj.update_joint_motor(
                    self.joint_motors[joint_id][0],
                    self.joint_motors[joint_id][1],
                )
            # breakpoint()

    # Probably move somewhere else, maybe make manipualtor an attribute instead?
    def ee_link_id(self, hand=0) -> int:
        """Gets the Habitat Sim link id of the end-effector."""
        if hand == 0:
            return self.params.ee_link_right
        else:
            return self.params.ee_link_left

    @property
    def ee_local_offset(self) -> mn.Vector3:
        """Gets the relative offset of the end-effector center from the
        end-effector link.
        """
        return self.params.ee_offset

    @property
    def ee_transform(self) -> mn.Matrix4:
        return self.ee_transform_hand(0)

    def ee_transform_hand(self, hand=0) -> mn.Matrix4:
        """Gets the transformation of the end-effector location. This is offset
        from the end-effector link location.
        """
        # breakpoint()
        ee_link = self.params.ee_link_left
        if hand == 1:
            ee_link = self.params.ee_link_right
        ef_link_transform = self.sim_obj.get_link_scene_node(
            ee_link
        ).transformation

        ef_link_transform.translation = ef_link_transform.transform_point(
            self.ee_local_offset
        )

        return ef_link_transform

    def set_joint_transform(self, pos: List, transform: mn.Matrix4):
        self.joint_rotation = pos
        # TODO: sim_obj.joint_positions isnt in reality joint_Rotation?
        # breakpoint()
        other = [0,0,0,1] * 37
        self.sim_obj.joint_positions = pos + other
        # breakpoint()
        self.sim_obj.transformation = transform
        # breakpoint()

    def open_gripper(self):
        pass
