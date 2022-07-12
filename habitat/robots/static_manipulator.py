# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import attr
import numpy as np

from habitat.robots.robot_interface import RobotInterface
from habitat_sim.physics import JointMotorSettings
from habitat_sim.simulator import Simulator


@attr.s(auto_attribs=True, slots=True)
class StaticManipulatorParams:
    """Data to configure a static manipulator.
    :property arm_joints: The joint ids of the arm joints.
    :property gripper_joints: The habitat sim joint ids of any grippers.
    :property arm_init_params: The starting joint angles of the arm. If None,
        resets to 0.
    :property gripper_init_params: The starting joint positions of the gripper. If None,
        resets to 0.
    :property ee_offset: The 3D offset from the end-effector link to the true
        end-effector position.
    :property ee_link: The Habitat Sim link ID of the end-effector.
    :property ee_constraint: A (2, N) shaped array specifying the upper and
        lower limits for each end-effector joint where N is the arm DOF.
    :property gripper_closed_state: All gripper joints must achieve this
        state for the gripper to be considered closed.
    :property gripper_open_state: All gripper joints must achieve this
        state for the gripper to be considered open.
    :property gripper_state_eps: Error margin for detecting whether gripper is closed.
    :property arm_mtr_pos_gain: The position gain of the arm motor.
    :property arm_mtr_vel_gain: The velocity gain of the arm motor.
    :property arm_mtr_max_impulse: The maximum impulse of the arm motor.
    """

    arm_joints: List[int]
    gripper_joints: List[int]

    arm_init_params: Optional[np.ndarray]
    gripper_init_params: Optional[np.ndarray]

    ee_offset: np.ndarray
    ee_link: int
    ee_constraint: np.ndarray

    gripper_closed_state: np.ndarray
    gripper_open_state: np.ndarray
    gripper_state_eps: float

    arm_mtr_pos_gain: float
    arm_mtr_vel_gain: float
    arm_mtr_max_impulse: float


class StaticManipulator(RobotInterface):
    """Robot with a fixed base and controllable arm."""

    def __init__(
        self,
        params: StaticManipulatorParams,
        urdf_path: str,
        sim: Simulator,
    ):
        r"""Constructor"""
        super().__init__()
        self.urdf_path = urdf_path
        self.params = params
        self._sim = sim
        self.sim_obj = None

        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to motor settings for convenience
        self.joint_motors: Dict[int, Tuple[int, JointMotorSettings]] = {}
        # maps joint ids to position index
        self.joint_pos_indices: Dict[int, int] = {}
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}
        self.joint_limits: Tuple[np.ndarray, np.ndarray] = None

        # defaults for optional params
        if self.params.gripper_init_params is None:
            self.params.gripper_init_params = np.zeros(
                len(self.params.gripper_joints), dtype=np.float32
            )
        if self.params.arm_init_params is None:
            self.params.arm_init_params = np.zeros(
                len(self.params.arm_joints), dtype=np.float32
            )

    def reconfigure(self) -> None:
        pass

    def update(self) -> None:
        pass

    def reset(self) -> None:
        pass
