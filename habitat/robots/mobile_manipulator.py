# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Set

import attr
import magnum as mn
import numpy as np

from habitat.robots.robot_manipulator import RobotManipulator
from habitat_sim.physics import JointMotorSettings
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import orthonormalize_rotation_shear


@attr.s(auto_attribs=True, slots=True)
class RobotCameraParams:
    """Data to configure a camera placement on the robot.
    :property attached_link_id: Which link ID this camera is attached to, -1
        for the base link.
    :property cam_offset_pos: The 3D position of the camera relative to the
        transformation of the attached link.
    :property cam_look_at_pos: The 3D of where the camera should face relative
        to the transformation of the attached link.
    :property relative_transform: An added local transform for the camera.
    """

    attached_link_id: int
    cam_offset_pos: mn.Vector3
    cam_look_at_pos: mn.Vector3
    relative_transform: mn.Matrix4 = mn.Matrix4.identity_init()


# TODO: refactor this class to support spherical joints: multiple dofs per link and #dofs != #positions
@attr.s(auto_attribs=True, slots=True)
class MobileManipulatorParams:
    """Data to configure a mobile manipulator.
    :property arm_joints: The joint ids of the arm joints.
    :property gripper_joints: The habitat sim joint ids of any grippers.
    :property wheel_joints: The joint ids of the wheels. If the wheels are not controlled, then this should be None
    :property arm_init_params: The starting joint angles of the arm. If None,
        resets to 0.
    :property gripper_init_params: The starting joint positions of the gripper. If None,
        resets to 0.
    :property ee_offset: The 3D offset from the end-effector link to the true
        end-effector position.
    :property ee_link: The Habitat Sim link ID of the end-effector.
    :property ee_constraint: A (2, N) shaped array specifying the upper and
        lower limits for each end-effector joint where N is the arm DOF.
    :property cameras: The cameras and where they should go. The key is the
        prefix to match in the sensor names. For example, a key of `"robot_head"`
        will match sensors `"robot_head_rgb"` and `"robot_head_depth"`
    :property gripper_closed_state: All gripper joints must achieve this
        state for the gripper to be considered closed.
    :property gripper_open_state: All gripper joints must achieve this
        state for the gripper to be considered open.
    :property gripper_state_eps: Error margin for detecting whether gripper is closed.
    :property arm_mtr_pos_gain: The position gain of the arm motor.
    :property arm_mtr_vel_gain: The velocity gain of the arm motor.
    :property arm_mtr_max_impulse: The maximum impulse of the arm motor.
    :property wheel_mtr_pos_gain: The position gain of the wheeled motor (if
        there are wheels).
    :property wheel_mtr_vel_gain: The velocity gain of the wheel motor (if
        there are wheels).
    :property wheel_mtr_max_impulse: The maximum impulse of the wheel motor (if
        there are wheels).
    :property base_offset: The offset of the root transform from the center ground point for navmesh kinematic control.
    """

    arm_joints: List[int]
    gripper_joints: List[int]
    wheel_joints: Optional[List[int]]

    arm_init_params: Optional[np.ndarray]
    gripper_init_params: Optional[np.ndarray]

    ee_offset: mn.Vector3
    ee_link: int
    ee_constraint: np.ndarray

    cameras: Dict[str, RobotCameraParams]

    gripper_closed_state: np.ndarray
    gripper_open_state: np.ndarray
    gripper_state_eps: float

    arm_mtr_pos_gain: float
    arm_mtr_vel_gain: float
    arm_mtr_max_impulse: float

    wheel_mtr_pos_gain: float
    wheel_mtr_vel_gain: float
    wheel_mtr_max_impulse: float

    base_offset: mn.Vector3
    base_link_names: Set[str]


class MobileManipulator(RobotManipulator):
    """Robot with a controllable base and arm."""

    def __init__(
        self,
        params: MobileManipulatorParams,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_base: bool = True,
    ):
        r"""Constructor
        :param limit_robo_joints: If true, joint limits of robot are always
            enforced.
        """
        super().__init__(
            urdf_path=urdf_path,
            params=params,
            sim=sim,
            limit_robo_joints=limit_robo_joints,
        )

        self._fix_joint_values: Optional[np.ndarray] = None
        self._fixed_base = fixed_base

        self._cameras = defaultdict(list)
        for camera_prefix in self.params.cameras:
            for sensor_name in self._sim._sensors:
                if sensor_name.startswith(camera_prefix):
                    self._cameras[camera_prefix].append(sensor_name)

        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}

    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        super().reconfigure()

        # set correct gains for wheels
        if self.params.wheel_joints is not None:
            jms = JointMotorSettings(
                0,  # position_target
                self.params.wheel_mtr_pos_gain,  # position_gain
                0,  # velocity_target
                self.params.wheel_mtr_vel_gain,  # velocity_gain
                self.params.wheel_mtr_max_impulse,  # max_impulse
            )
            # pylint: disable=not-an-iterable
            for i in self.params.wheel_joints:
                self.sim_obj.update_joint_motor(self.joint_motors[i][0], jms)

        self._update_motor_settings_cache()

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
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
                        self.params.ee_link
                    ).transformation

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
        if self._fix_joint_values is not None:
            self.arm_joint_pos = self._fix_joint_values

        self.sim_obj.awake = True

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only arm and gripper joint motors (not gains) are reset by default, derived class should handle any other changes."""

        # reset the initial joint positions
        self._fix_joint_values = None
        super().reset()

    #############################################
    # BASE RELATED
    #############################################
    @property
    def base_pos(self):
        """Get the robot base ground position via configured local offset from origin."""
        return (
            self.sim_obj.translation
            + self.sim_obj.transformation.transform_vector(
                self.params.base_offset
            )
        )

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        """Set the robot base to a desired ground position (e.g. NavMesh point) via configured local offset from origin."""
        if len(position) != 3:
            raise ValueError("Base position needs to be three dimensions")
        self.sim_obj.translation = (
            position
            - self.sim_obj.transformation.transform_vector(
                self.params.base_offset
            )
        )

    @property
    def base_rot(self) -> float:
        return self.sim_obj.rotation.angle()

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        self.sim_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rotation_y_rad), mn.Vector3(0, 1, 0)
        )

    def is_base_link(self, link_id: int) -> bool:
        return (
            self.sim_obj.get_link_name(link_id) in self.params.base_link_names
        )
