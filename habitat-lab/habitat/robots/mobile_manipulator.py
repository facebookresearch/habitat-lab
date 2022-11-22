# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set

import attr
import magnum as mn
import numpy as np

from habitat.robots.manipulator import Manipulator
from habitat.robots.robot_base import RobotBase
from habitat_sim.simulator import Simulator


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
    cam_offset_pos: mn.Vector3 = mn.Vector3.zero_init()
    cam_look_at_pos: mn.Vector3 = mn.Vector3.zero_init()
    cam_orientation: mn.Vector3 = mn.Vector3.zero_init()
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


class MobileManipulator(Manipulator, RobotBase):
    """Robot with a controllable base and arm."""

    def __init__(
        self,
        params: MobileManipulatorParams,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_base: bool = True,
        base_type="mobile",
    ):
        r"""Constructor
        :param params: The parameter of the manipulator robot.
        :param urdf_path: The path to the robot's URDF file.
        :param sim: The simulator.
        :param limit_robo_joints: If true, joint limits of robot are always
            enforced.
        :param fixed_base: If the robot's base is fixed or not.
        :param base_type: The base type
        """
        # instantiate a manipulator
        Manipulator.__init__(
            self,
            urdf_path=urdf_path,
            params=params,
            sim=sim,
            limit_robo_joints=limit_robo_joints,
        )
        # instantiate a robotBase
        RobotBase.__init__(
            self,
            urdf_path=urdf_path,
            params=params,
            sim=sim,
            limit_robo_joints=limit_robo_joints,
            fixed_based=fixed_base,
            sim_obj=self.sim_obj,
            base_type=base_type,
        )

    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        Manipulator.reconfigure(self)
        RobotBase.reconfigure(self)

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        Manipulator.update(self)
        RobotBase.update(self)

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only arm and gripper joint motors (not gains) are reset by default, derived class should handle any other changes."""
        Manipulator.reset(self)
        RobotBase.reset(self)
