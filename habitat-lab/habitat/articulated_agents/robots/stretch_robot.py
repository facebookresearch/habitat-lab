# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)


class StretchRobot(MobileManipulator):
    def _get_fetch_params(self):
        return MobileManipulatorParams(
            arm_joints=[28, 27, 26, 25, 23, 31, 33, 34, 7, 8],
            gripper_joints=[36, 38],
            wheel_joints=[4, 42],
            arm_init_params=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            gripper_init_params=np.array([0.0, 0.0], dtype=np.float32),
            ee_offset=[mn.Vector3(0.08, 0, 0)],
            ee_links=[36],
            ee_constraint=np.array(
                [[[-0.08, 0.29], [-0.84, -0.27], [0.01, 1.12]]]
            ),
            cameras={
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.1),
                    attached_link_id=14,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            },
            gripper_closed_state=np.array([0.0, 0.0], dtype=np.float32),
            gripper_open_state=np.array([0.6, 0.6], dtype=np.float32),
            gripper_state_eps=0.1,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            wheel_mtr_pos_gain=0.0,
            wheel_mtr_vel_gain=1.3,
            wheel_mtr_max_impulse=10.0,
            base_offset=mn.Vector3(0.0, -0.5, 0.0),
            base_link_names={
                "link_right_wheel",
                "link_left_wheel",
                "caster_link",
                "link_mast",
                "base_link",
                "laser",
            },
        )

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def __init__(
        self, agent_cfg, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_fetch_params(),
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
        )


class StretchRobotNoWheels(StretchRobot):
    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(urdf_path, sim, limit_robo_joints, fixed_base)
