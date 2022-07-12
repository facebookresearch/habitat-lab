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
        limit_robo_joints: bool = True,
    ):
        r"""Constructor"""
        super().__init__()
        self.urdf_path = urdf_path
        self.params = params
        self._sim = sim
        self._limit_robo_joints = limit_robo_joints
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
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        ao_mgr = self._sim.get_articulated_object_manager()
        self.sim_obj = ao_mgr.add_articulated_object_from_urdf(
            self.urdf_path, fixed_base=True
        )
        if self._limit_robo_joints:
            # automatic joint limit clamping after each call to sim.step_physics()
            self.sim_obj.auto_clamp_joint_limits = True
        for link_id in self.sim_obj.get_link_ids():
            self.joint_pos_indices[
                link_id
            ] = self.sim_obj.get_link_joint_pos_offset(link_id)
            self.joint_dof_indices[link_id] = self.sim_obj.get_link_dof_offset(
                link_id
            )
        self.joint_limits = self.sim_obj.joint_position_limits

        # remove any default damping motors
        for motor_id in self.sim_obj.existing_joint_motor_ids:
            self.sim_obj.remove_joint_motor(motor_id)
        # re-generate all joint motors with arm gains.

        jms = JointMotorSettings()
        self.sim_obj.create_all_motors(jms)
        self._update_motor_settings_cache()

        if self.params.arm_joints is not None:
            jms = JointMotorSettings(
                0,  # position_target
                self.params.arm_mtr_pos_gain,  # position_gain
                0,  # velocity_target
                self.params.arm_mtr_vel_gain,  # velocity_gain
                self.params.arm_mtr_max_impulse,  # max_impulse
            )
            for i in self.params.arm_joints:
                self.sim_obj.update_joint_motor(self.joint_motors[i][0], jms)
        self._update_motor_settings_cache()

        if self.params.gripper_joints is not None:
            jms = JointMotorSettings(
                0,  # position_target
                self.params.arm_mtr_pos_gain,  # position_gain
                0,  # velocity_target
                self.params.arm_mtr_vel_gain,  # velocity_gain
                self.params.arm_mtr_max_impulse,  # max_impulse
            )
            for i in self.params.gripper_joints:
                self.sim_obj.update_joint_motor(self.joint_motors[i][0], jms)
        
        # set initial states and targets
        self.arm_joint_pos = self.params.arm_init_params
        self.gripper_joint_pos = self.params.gripper_init_params

        self._update_motor_settings_cache()


    def update(self) -> None:
        """Updates sleep state"""
        self.sim_obj.awake = True

    def reset(self) -> None:
        """Reset joints"""
        self.sim_obj.clear_joint_states()

        self.arm_joint_pos = self.params.arm_init_params
        self.gripper_joint_pos = self.params.gripper_init_params

        self._update_motor_settings_cache()
        self.update()

    def _update_motor_settings_cache(self):
        """Updates the JointMotorSettings cache for cheaper future updates"""
        self.joint_motors = {}
        for (
            motor_id,
            joint_id,
        ) in self.sim_obj.existing_joint_motor_ids.items():
            self.joint_motors[joint_id] = (
                motor_id,
                self.sim_obj.get_joint_motor_settings(motor_id),
            )
