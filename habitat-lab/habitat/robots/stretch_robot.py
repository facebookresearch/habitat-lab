# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat.robots.mobile_manipulator import (
    MobileManipulator,
    MobileManipulatorParams,
    RobotCameraParams,
)
from habitat_sim.utils.common import orthonormalize_rotation_shear


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
            ee_offset=mn.Vector3(0.08, 0, 0),
            ee_link=36,
            ee_constraint=np.array(
                [[-0.15, 0.32], [-0.90, -0.38], [0.02, 1.12]]
            ),
            cameras={
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.1),
                    attached_link_id=14,
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "robot_third": RobotCameraParams(
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
            base_offset=mn.Vector3(0, -0.5, 0),
            base_link_names={
                "link_right_wheel",
                "link_left_wheel",
                "caster_link",
                "link_mast",
                "base_link",
                "laser",
            },
        )

    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(
            self._get_fetch_params(),
            urdf_path,
            sim,
            limit_robo_joints,
            fixed_base,
        )

    def reconfigure(self) -> None:
        super().reconfigure()
        self.update()

    def reset(self) -> None:
        super().reset()
        self.update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def update(self):
        agent_node = self._sim._default_agent.scene_node
        inv_T = agent_node.transformation.inverted()

        for cam_prefix, sensor_names in self._cameras.items():
            for sensor_name in sensor_names:
                sens_obj = self._sim._sensors[sensor_name]._sensor_object
                cam_info = self.params.cameras[cam_prefix]

                if cam_info.attached_link_id == -1:
                    link_trans = self.sim_obj.transformation
                else:
                    link_trans = self.sim_obj.get_link_scene_node(
                        cam_info.attached_link_id
                    ).transformation

                if cam_info.cam_look_at_pos == mn.Vector3(0, 0, 0):
                    pos = cam_info.cam_offset_pos
                    ori = cam_info.cam_orientation
                    Mt = mn.Matrix4.translation(pos)
                    Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
                    My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
                    Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
                    cam_transform = Mt @ Mz @ My @ Mx
                else:
                    cam_transform = mn.Matrix4.look_at(
                        cam_info.cam_offset_pos,
                        cam_info.cam_look_at_pos,
                        mn.Vector3(0, 1, 0),
                    )
                cam_transform = (
                    link_trans @ cam_transform @ cam_info.relative_transform
                )
                cam_transform = inv_T @ cam_transform

                sens_obj.node.transformation = orthonormalize_rotation_shear(
                    cam_transform
                )


class StretchRobotNoWheels(StretchRobot):
    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(urdf_path, sim, limit_robo_joints, fixed_base)
