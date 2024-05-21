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


class SpotRobot(MobileManipulator):
    def _get_spot_params(self):
        return MobileManipulatorParams(
            arm_joints=list(range(0, 7)),
            gripper_joints=[7],
            leg_joints=list(range(8, 20)),
            arm_init_params=np.array([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0]),
            gripper_init_params=np.array([-1.56]),
            leg_init_params=[
                0.0,
                0.7,
                -1.5,
                0.0,
                0.7,
                -1.5,
                0.0,
                0.7,
                -1.5,
                0.0,
                0.7,
                -1.5,
            ],
            ee_offset=[mn.Vector3(0.08, 0, 0)],
            ee_links=[7],
            ee_constraint=np.array([[[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]]),
            cameras={
                "articulated_agent_arm_depth": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "articulated_agent_arm_rgb": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.023, 0.03),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "articulated_agent_arm_panoptic": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.0, 0.018),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "head_stereo_right": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(
                        0.4164822634134684, 0.0, 0.03614789234067159
                    ),
                    cam_orientation=mn.Vector3(
                        0.0290787, -0.940569, -0.38998877
                    ),
                    attached_link_id=-1,
                ),
                "head_stereo_left": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(
                        0.4164822634134684, 0.0, -0.03740343144695029
                    ),
                    cam_orientation=mn.Vector3(
                        -3.1125141, -0.940569, 2.751605
                    ),
                    attached_link_id=-1,
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-0.5, 1.7, -0.5),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-1,
                ),
                "articulated_agent_jaw_depth": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "articulated_agent_jaw_rgb": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
                "articulated_agent_jaw_panoptic": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.166, 0.0, -0.107),
                    cam_orientation=mn.Vector3(0, -1.571, 0.0),
                    attached_link_id=6,
                    relative_transform=mn.Matrix4.rotation_z(mn.Deg(-90)),
                ),
            },
            gripper_closed_state=np.array([0.0], dtype=np.float32),
            gripper_open_state=np.array([-1.56], dtype=np.float32),
            gripper_state_eps=0.01,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            leg_mtr_pos_gain=2.0,
            leg_mtr_vel_gain=1.3,
            leg_mtr_max_impulse=100.0,
            base_offset=mn.Vector3(0.0, -0.48, 0.0),
            base_link_names={
                "base",
            },
            navmesh_offsets=[[0.0, 0.0], [0.25, 0.0], [-0.25, 0.0]],
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
            self._get_spot_params(),
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
            base_type="leg",
        )
