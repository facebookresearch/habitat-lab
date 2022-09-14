

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

<<<<<<< HEAD
from typing import Dict, List, Optional, Tuple
=======
from typing import List, Optional
>>>>>>> 5968574a (Use mutiple interherences to design modules for constructing robot parts, along with the test code)

import attr
import magnum as mn
import numpy as np

<<<<<<< HEAD
from habitat.robots.robot_interface import RobotInterface
from habitat.robots.robot_manipulator import RobotManipulator
from habitat_sim.physics import JointMotorSettings
=======
from habitat.robots.manipulator import Manipulator
from habitat.robots.robotbase import RobotBase
>>>>>>> 5968574a (Use mutiple interherences to design modules for constructing robot parts, along with the test code)
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

    ee_offset: mn.Vector3
    ee_link: int
    ee_constraint: np.ndarray

    gripper_closed_state: np.ndarray
    gripper_open_state: np.ndarray
    gripper_state_eps: float

    arm_mtr_pos_gain: float
    arm_mtr_vel_gain: float
    arm_mtr_max_impulse: float


class StaticManipulator(Manipulator, RobotBase):
    """Robot with a fixed base and controllable arm."""

    def __init__(
        self,
        params: StaticManipulatorParams,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_base: bool = True,
    ):
<<<<<<< HEAD
        r"""Constructor"""
        super().__init__(
            urdf_path=urdf_path, 
            params=params,
            sim=sim,
            limit_robo_joints=limit_robo_joints)

        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}
        self._fixed_base = True

    #############################################
    # ARM PROPERTIES GETTERS + SETTERS
    #############################################
    def clip_ee_to_workspace(self, pos: np.ndarray) -> np.ndarray:
        """Clips a 3D end-effector position within region the robot can reach."""
        return np.clip(
            pos,
            self.params.ee_constraint[:, 0],
            self.params.ee_constraint[:, 1],
        )

    @property
    def arm_motor_forces(self) -> np.ndarray:
        """Get the current torques on the arm joint motors"""
        return np.array(self.sim_obj.joint_forces)

    @arm_motor_forces.setter
    def arm_motor_forces(self, ctrl: List[float]) -> None:
        """Set the desired torques of the arm joint motors"""
        self.sim_obj.joint_forces = ctrl
=======
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
        )
        # instantiate a robot base
        RobotBase.__init__(
            self,
            urdf_path=urdf_path,
            params=params,
            sim=sim,
            limit_robo_joints=limit_robo_joints,
            fixed_based=fixed_base,
            base_type="static",
            sim_obj=self.sim_obj,
        )

    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        Manipulator.reconfigure(self)
        RobotBase.reconfigure(self)

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        Manipulator.reconfigure(self)
        RobotBase.reconfigure(self)

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only arm and gripper joint motors (not gains) are reset by default, derived class should handle any other changes."""
        Manipulator.reset(self)
        RobotBase.reset(self)
>>>>>>> 5968574a (Use mutiple interherences to design modules for constructing robot parts, along with the test code)

    #############################################
    # HIDDEN
    #############################################
    def _get_translation_from_htm(self, mat: np.ndarray) -> np.ndarray:
        assert mat.shape == (
            4,
            4,
        ), f"Invalid matrix shape. Homogenous transformation matrices should be 4x4, got {mat.shape} instead"
        return mat[:3, -1]
