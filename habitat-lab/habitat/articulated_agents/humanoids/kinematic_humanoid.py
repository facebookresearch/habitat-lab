# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import magnum as mn
import numpy as np

import habitat_sim
from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)


class KinematicHumanoid(MobileManipulator):
    def _get_humanoid_params(self):
        return MobileManipulatorParams(
            arm_joints=[],  # For now we do not add arm_joints
            gripper_joints=[],
            wheel_joints=None,
            arm_init_params=None,
            gripper_init_params=None,
            gripper_closed_state=np.array([]),
            gripper_open_state=np.array([]),
            gripper_state_eps=None,
            wheel_mtr_pos_gain=None,
            wheel_mtr_vel_gain=None,
            wheel_mtr_max_impulse=None,
            ee_offset=[mn.Vector3(), mn.Vector3()],
            ee_links=[14, 18],
            ee_constraint=np.zeros((2, 2, 3)),
            cameras={
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 0.5, 0.25),
                    cam_look_at_pos=mn.Vector3(0.0, 0.5, 0.75),
                    attached_link_id=-1,
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-1.2, 2.0, -1.2),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            },
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            base_offset=mn.Vector3(0, -0.9, 0),
            base_link_names={
                "base_link",
            },
            ee_count=2,
        )

    def __init__(
        self, urdf_path, sim, limit_robo_joints=False, fixed_base=False
    ):
        super().__init__(
            self._get_humanoid_params(),
            urdf_path,
            sim,
            limit_robo_joints,
            fixed_base,
            maintain_link_order=True,
        )

        # The offset and base transform are used so that the
        # character can have different rotations and shifts but the
        # when walking but the base is not affected.
        # Base transform will move linearly when the character follows
        # a path, whereas the offset transform will be changing the position
        # to simulate different gaits
        self.offset_transform = mn.Matrix4()

    @property
    def inverse_offset_transform(self):
        rot = self.offset_transform.rotation().transposed()
        translation = -rot * self.offset_transform.translation
        return mn.Matrix4.from_(rot, translation)

    @property
    def base_transformation(self):
        return self.sim_obj.transformation @ self.inverse_offset_transform

    @property
    def base_pos(self):
        """Get the humanoid base ground position"""
        # via configured local offset from origin
        base_transform = self.base_transformation
        return base_transform.translation + base_transform.transform_vector(
            self.params.base_offset
        )

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        """Set the robot base to a desired ground position (e.g. NavMesh point)"""
        # via configured local offset from origin.
        # TODO: maybe this can be simplified

        if len(position) != 3:
            raise ValueError("Base position needs to be three dimensions")
        base_transform = self.base_transformation
        base_pos = position - base_transform.transform_vector(
            self.params.base_offset
        )
        base_transform.translation = base_pos
        final_transform = base_transform @ self.offset_transform

        self.sim_obj.transformation = final_transform

    @property
    def base_rot(self) -> float:
        return self.base_transformation.rotation.angle()

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        if self._base_type == "mobile" or self._base_type == "leg":
            self.sim_obj.rotation = mn.Quaternion.rotation(
                mn.Rad(rotation_y_rad), mn.Vector3(0, 1, 0)
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    def reconfigure(self) -> None:
        """Instantiates the human in the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        super().reconfigure()
        self.sim_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        # remove any remaining joint motors
        for motor_id in self.sim_obj.existing_joint_motor_ids:
            self.sim_obj.remove_joint_motor(motor_id)

    def set_joint_transform(
        self,
        joint_list: List[float],
        offset_transform: mn.Matrix4,
        base_transform: mn.Matrix4,
    ) -> None:
        """Sets the joints, base and offset transform of the humanoid"""
        # TODO: should this go into articulated agent?
        self.sim_obj.joint_positions = joint_list
        self.offset_transform = offset_transform
        final_transform = base_transform @ offset_transform

        self.sim_obj.transformation = final_transform

    def get_joint_transform(self):
        """Returns the joints and base transform of the humanoid"""
        # TODO: should this go into articulated agent?
        return self.sim_obj.joint_positions, self.sim_obj.transformation
