# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import attr
import magnum as mn
import numpy as np

from habitat.articulated_agents.manipulator import Manipulator
from habitat_sim.simulator import Simulator


@attr.s(auto_attribs=True, slots=True)
class StaticManipulatorParams:
    """
    Data to configure a static manipulator.

    :property arm_joints: The joint ids of the arm joints.
    :property gripper_joints: The habitat sim joint ids of any grippers.
    :property arm_init_params: The starting joint angles of the arm. If None, resets to 0.
    :property gripper_init_params: The starting joint positions of the gripper. If None, resets to 0.
    :property ee_offset: The 3D offset from the end-effector link to the true end-effector position.
    :property ee_links: A list with the Habitat Sim link ID of the end-effector.
    :property ee_constraint: A (ee_count, 2, N) shaped array specifying the upper and lower limits for each end-effector joint where N is the arm DOF.
    :property gripper_closed_state: All gripper joints must achieve this state for the gripper to be considered closed.
    :property gripper_open_state: All gripper joints must achieve this state for the gripper to be considered open.
    :property gripper_state_eps: Error margin for detecting whether gripper is closed.
    :property arm_mtr_pos_gain: The position gain of the arm motor.
    :property arm_mtr_vel_gain: The velocity gain of the arm motor.
    :property arm_mtr_max_impulse: The maximum impulse of the arm motor.
    :property ee_count: how many end effectors
    """

    arm_joints: List[int]
    gripper_joints: List[int]

    arm_init_params: Optional[np.ndarray]
    gripper_init_params: Optional[np.ndarray]

    ee_offset: List[mn.Vector3]
    ee_links: List[int]
    ee_constraint: np.ndarray

    gripper_closed_state: np.ndarray
    gripper_open_state: np.ndarray
    gripper_state_eps: float

    arm_mtr_pos_gain: float
    arm_mtr_vel_gain: float
    arm_mtr_max_impulse: float

    ee_count: Optional[int] = 1


class StaticManipulator(Manipulator):
    """Robot with a fixed base and controllable arm."""

    def __init__(
        self,
        params: StaticManipulatorParams,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_base: bool = True,
        auto_update_sensor_transform=False,
    ):
        r"""Constructor
        :param params: The parameter of the manipulator robot.
        :param urdf_path: The path to the robot's URDF file.
        :param sim: The simulator.
        :param limit_robo_joints: If true, joint limits of robot are always
            enforced.
        :param fixed_base: If the robot's base is fixed or not.
        """
        # instantiate a manipulator
        Manipulator.__init__(
            self,
            urdf_path=urdf_path,
            params=params,
            sim=sim,
            limit_robo_joints=limit_robo_joints,
            fixed_based=fixed_base,
            auto_update_sensor_transform=auto_update_sensor_transform,
        )

    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        Manipulator.reconfigure(self)

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        Manipulator.update(self)

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only arm and gripper joint motors (not gains) are reset by default, derived class should handle any other changes.
        """
        Manipulator.reset(self)
