# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, Tuple

import attr
import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
)
from habitat.utils.geometry_utils import quat_to_euler


@attr.s(auto_attribs=True, slots=True)
class SpotParams:
    """Data to configure a mobile manipulator.
    :property arm_joints: The joint ids of the arm joints.
    :property gripper_joints: The habitat sim joint ids of any grippers.
    :property arm_init_params: The starting joint angles of the arm. If None,
        resets to 0.
    :property gripper_init_params: The starting joint positions of the gripper. If None,
        resets to 0.
    :property ee_offset: The 3D offset from the end-effector link to the true
        end-effector position.
    :property ee_link: The Habitat Sim link ID of the end-effector.
    :property ee_constraint: A (2, 3) shaped array specifying the upper and
        lower limits for the 3D end-effector position.
    :property cameras: The cameras and where they should go. The key is the
        prefix to match in the sensor names. For example, a key of `"head"`
        will match sensors `"head_rgb"` and `"head_depth"`
    :property gripper_closed_state: All gripper joints must achieve this
        state for the gripper to be considered closed.
    :property gripper_open_state: All gripper joints must achieve this
        state for the gripper to be considered open.
    :property gripper_state_eps: Error margin for detecting whether gripper is closed.
    :property arm_mtr_pos_gain: The position gain of the arm motor.
    :property arm_mtr_vel_gain: The velocity gain of the arm motor.
    :property arm_mtr_max_impulse: The maximum impulse of the arm motor.
    :property base_offset: The offset of the root transform from the center ground point for navmesh kinematic control.
    :property base_link_names: The name of the links
    :property leg_joints: The joint ids of the legs if applicable. If the legs are not controlled, then this should be None
    :property leg_init_params: The starting joint positions of the leg joints. If None,
        resets to 0.
    :property leg_mtr_pos_gain: The position gain of the leg motor (if
        there are legs).
    :property leg_mtr_vel_gain: The velocity gain of the leg motor (if
        there are legs).
    :property leg_mtr_max_impulse: The maximum impulse of the leg motor (if
        there are legs).
    :property ee_count: how many end effectors
    """

    arm_joints: List[int]
    gripper_joints: List[int]

    arm_init_params: Optional[List[float]]
    gripper_init_params: Optional[List[float]]

    ee_offset: List[mn.Vector3]
    ee_links: List[int]
    ee_constraint: np.ndarray

    cameras: Dict[str, ArticulatedAgentCameraParams]

    gripper_closed_state: List[float]
    gripper_open_state: List[float]
    gripper_state_eps: float

    arm_mtr_pos_gain: float
    arm_mtr_vel_gain: float
    arm_mtr_max_impulse: float

    base_offset: mn.Vector3
    base_link_names: Set[str]

    leg_joints: Optional[List[int]] = None
    leg_init_params: Optional[List[float]] = None
    leg_mtr_pos_gain: Optional[float] = None
    leg_mtr_vel_gain: Optional[float] = None
    leg_mtr_max_impulse: Optional[float] = None

    ee_count: Optional[int] = 1


class SpotRobot(MobileManipulator):
    def _get_spot_params(self):
        return SpotParams(
            arm_joints=list(range(0, 7)),
            gripper_joints=[7],
            leg_joints=list(range(8, 20)),
            arm_init_params=[0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0],
            gripper_init_params=[-1.56],
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
            ee_offset=[mn.Vector3(0.0, 0, -0.1)],
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
                    cam_offset_pos=mn.Vector3(0.166, 0.023, -0.095),
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
            gripper_closed_state=[0.0],
            gripper_open_state=[-1.56],
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
        )

    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        return self.sim_obj.transformation @ add_rot

    def get_ee_local_pose(
        self, ee_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the local pose of the end-effector.

        :param ee_index: the end effector index for which we want the link transform
        """
        if ee_index >= len(self.params.ee_links):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )

        ef_link_transform = self.ee_transform()
        base_transform = self.base_transformation
        # Get the local ee location (x,y,z)
        local_ee_location = base_transform.inverted().transform_point(
            ef_link_transform.translation
        )
        # Get the local ee rotation (r,p,y)
        local_ee_transform = base_transform.inverted() @ ef_link_transform
        local_ee_quat = mn.Quaternion.from_matrix(
            local_ee_transform.rotation()
        )
        local_ee_euler = quat_to_euler(
            (
                local_ee_quat.scalar,
                local_ee_quat.vector[0],
                local_ee_quat.vector[2],
                local_ee_quat.vector[1],
            )
        )
        return np.array(local_ee_location), np.array(local_ee_euler)

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
