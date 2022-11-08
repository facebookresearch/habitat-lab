# Copyright (c) Facebook, Inc. and its affiliates.
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
            # 28: joint_arm_l0
            # 27: joint_arm_l1
            # 26: joint_arm_l2
            # 25: joint_arm_l3
            # 23: joint_lift
            # 31: joint_wrist_yaw
            # 39: joint_wrist_pitch
            # 40: joint_wrist_roll
            # 7: joint_head_pan
            # 8: joint_head_tilt
            #arm_joints=[28, 27, 26, 25, 23, 31, 39, 40, 7, 8],
            arm_joints=[28, 27, 26, 25, 23, 31, 33, 34, 7, 8],
            # 34: joint_gripper_finger_left
            # 36: joint_gripper_finger_right
            #gripper_joints=[34, 36],
            gripper_joints=[36, 38],
            # 4: joint_left_wheel
            # 44: joint_right_wheel
            #wheel_joints=[4, 44],
            wheel_joints=[4, 42],
            arm_init_params=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            gripper_init_params=np.array([0.0, 0.0], dtype=np.float32),
            ee_offset=mn.Vector3(0.08, 0, 0),
            ee_link=36,
            # ee_constraint=np.array([[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]),
            # inner-forward, left-right, height-high and down
            # ee_constraint=np.array(
            #     [[0.00, 0.23], [-0.74, -0.34], [-0.06, 1.03]]
            # ),
            ee_constraint=np.array(
                #[[-0.15, 0.32], [-0.90, -0.38], [0.02, 1.12]]
                [[-0.08, 0.29], [-0.84, -0.27], [0.01, 1.12]]
            ),
            # Camera color optical frame...
            # correct angles
            cameras={
                # "robot_arm": RobotCameraParams(
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                #     cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                #     attached_link_id=14,
                #     relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                #     @ mn.Matrix4.rotation_z(mn.Deg(90)),
                # ),
                "robot_head": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                    cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.1),
                    attached_link_id=14,
                    # cam_orientation=mn.Vector3(-3.1125141, -0.940569, 2.751605),
                    relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                    @ mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                # "robot_head_stereo_right": RobotCameraParams(
                #     # cam_offset_pos=mn.Vector3(
                #     #     0.4164822634134684, 0.0, 0.03614789234067159
                #     # ),
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                #     cam_orientation=mn.Vector3(
                #         0.0290787, -0.940569, -0.38998877
                #     ),
                #     cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                #     relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                #     @ mn.Matrix4.rotation_z(mn.Deg(90)),
                #     attached_link_id=14,
                # ),
                # "robot_head_stereo_left": RobotCameraParams(
                #     # cam_offset_pos=mn.Vector3(
                #     #     0.4164822634134684, 0.0, -0.03740343144695029
                #     # ),
                #     cam_offset_pos=mn.Vector3(0, 0.0, 0.1),
                #     cam_orientation=mn.Vector3(
                #         -3.1125141, -0.940569, 2.751605
                #     ),
                #     cam_look_at_pos=mn.Vector3(0.1, 0.0, 0.0),
                #     relative_transform=mn.Matrix4.rotation_y(mn.Deg(-90))
                #     @ mn.Matrix4.rotation_z(mn.Deg(90)),
                #     attached_link_id=14,
                # ),
                "robot_third": RobotCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
            },
            gripper_closed_state=np.array([0.0, 0.0], dtype=np.float32),
            #gripper_open_state=np.array([0.6, 0.6], dtype=np.float32),
            # Samller number to make it more relaisit
            gripper_open_state=np.array([0.02, 0.02], dtype=np.float32),
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

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    def reset(self) -> None:
        super().reset()

        # NOTE: this is necessary to set locked head and back positions
        self.update()

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    # def update(self):
    #     # super().update()

    #     agent_node = self._sim._default_agent.scene_node
    #     inv_T = agent_node.transformation.inverted()

    #     for cam_prefix, sensor_names in self._cameras.items():
    #         for sensor_name in sensor_names:
    #             sens_obj = self._sim._sensors[sensor_name]._sensor_object
    #             cam_info = self.params.cameras[cam_prefix]

    #             if cam_info.attached_link_id == -1:
    #                 link_trans = self.sim_obj.transformation
    #             else:
    #                 link_trans = self.sim_obj.get_link_scene_node(
    #                     cam_info.attached_link_id  # self.params.ee_link
    #                 ).transformation

    #             if cam_info.cam_look_at_pos == mn.Vector3(0, 0, 0):
    #                 pos = cam_info.cam_offset_pos
    #                 ori = cam_info.cam_orientation
    #                 Mt = mn.Matrix4.translation(pos)
    #                 Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
    #                 My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
    #                 Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
    #                 cam_transform = Mt @ Mz @ My @ Mx
    #             else:
    #                 cam_transform = mn.Matrix4.look_at(
    #                     cam_info.cam_offset_pos,
    #                     cam_info.cam_look_at_pos,
    #                     mn.Vector3(0, 1, 0),
    #                 )
    #             cam_transform = (
    #                 link_trans @ cam_transform @ cam_info.relative_transform
    #             )
    #             cam_transform = inv_T @ cam_transform

    #             sens_obj.node.transformation = orthonormalize_rotation_shear(
    #                 cam_transform
    #             )


class StretchRobotNoWheels(StretchRobot):
    def __init__(
        self, urdf_path, sim, limit_robo_joints=True, fixed_base=True
    ):
        super().__init__(urdf_path, sim, limit_robo_joints, fixed_base)
