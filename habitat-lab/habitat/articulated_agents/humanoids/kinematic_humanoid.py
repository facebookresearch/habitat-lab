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
            ee_links=[15, 9],
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

    def reconfigure(self) -> None:
        """Instantiates the human in the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        super().reconfigure()
        self.sim_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

    def set_joint_transform(
        self, joint_list: List[float], transform: mn.Matrix4
    ) -> None:
        """Sets the joints and base transform of the humanoid"""
        # TODO: should this go into articulated agent?
        self.sim_obj.joint_positions = joint_list
        self.sim_obj.transformation = transform

    def get_joint_transform(self):
        """Returns the joints and base transform of the humanoid"""
        # TODO: should this go into articulated agent?
        return self.sim_obj.joint_positions, self.sim_obj.transformation
