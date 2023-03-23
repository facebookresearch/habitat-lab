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

        # TODO: make this part of reset skill
        self.rest_joints = [
            0.01580232,
            0.00312952,
            0.00987611,
            0.99982146,
            0.04868594,
            -0.06787038,
            -0.00789235,
            0.99647429,
            -0.11562324,
            0.09068378,
            0.02232343,
            0.98889301,
            -0.03059093,
            0.0022178,
            0.03425224,
            0.99894247,
            0.11757437,
            -0.0052593,
            0.00770549,
            0.99302026,
            -0.10860021,
            -0.06826786,
            -0.00304792,
            0.99173394,
            0.06766052,
            -0.01406872,
            0.01672267,
            0.99746904,
            -0.05634359,
            -0.01199805,
            0.0221017,
            0.99809467,
            0.01740856,
            -0.03378484,
            0.00277393,
            0.99927365,
            -0.0625657,
            -0.08138424,
            0.02090534,
            0.99449741,
            0.09520805,
            -0.13450076,
            -0.01502134,
            0.98621465,
            0.02049966,
            0.10848507,
            -0.24204067,
            0.96396425,
            0.05280473,
            -0.07628868,
            -0.49520045,
            0.86381029,
            0.08596348,
            -0.22918482,
            0.10918048,
            0.9634128,
            -0.02809719,
            -0.06218687,
            0.24364708,
            0.9674603,
            0.10582447,
            0.06064448,
            0.47002208,
            0.87418686,
            -0.02126049,
            0.17058167,
            -0.09252438,
            0.98075946,
        ]
        self.rest_matrix = mn.Matrix4(
            np.array(
                [
                    [-0.9993708, -0.03505326, 0.00530393, 0],
                    [-0.03499541, 0.9993309, 0.01063382, 0],
                    [-0.00567313, 0.01044152, -0.99992883, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

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

    def set_rest_position(self) -> None:
        """Sets the agents in a resting position"""
        joint_list = self.rest_joints
        offset_transform = self.rest_matrix
        self.set_joint_transform(
            joint_list, offset_transform, self.base_transformation
        )

    def reconfigure(self) -> None:
        """Instantiates the human in the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        super().reconfigure()
        self.sim_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self.update()
        self.set_rest_position()

    def reset(self) -> None:
        super().reset()
        self.update()
        self.set_rest_position()

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
