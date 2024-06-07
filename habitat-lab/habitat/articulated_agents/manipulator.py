# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat.articulated_agents.articulated_agent_interface import (
    ArticulatedAgentInterface,
)
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat_sim.simulator import Simulator
from habitat_sim.utils.common import orthonormalize_rotation_shear


class Manipulator(ArticulatedAgentInterface):
    """Generic manipulator interface defines standard API functions. Robot with a controllable arm."""

    def __init__(
        self,
        params,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_based: bool = True,
        sim_obj=None,
        maintain_link_order=False,
        auto_update_sensor_transform=True,
        **kwargs,
    ):
        r"""Constructor"""
        ArticulatedAgentInterface.__init__(self)
        # Assign the variables
        self.params = params
        self.urdf_path = urdf_path
        self._sim = sim
        self._limit_robo_joints = limit_robo_joints
        self._fixed_base = fixed_based
        self.sim_obj = sim_obj
        self._maintain_link_order = maintain_link_order
        self._auto_update_sensor_transforms = auto_update_sensor_transform

        # Adapt Manipulator params to support multiple end effector indices
        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to motor settings for convenience
        self.joint_motors: Dict[int, Tuple[int, JointMotorSettings]] = {}
        # maps joint ids to position index
        self.joint_pos_indices: Dict[int, int] = {}
        # maps joint ids to velocity index
        self.joint_limits: Tuple[np.ndarray, np.ndarray] = None
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}
        # set the fixed joint values
        self._fix_joint_values: Optional[np.ndarray] = None

        # defaults for optional params
        if self.params.gripper_init_params is None:
            self.params.gripper_init_params = np.zeros(
                len(self.params.gripper_joints), dtype=np.float32
            )
        if self.params.arm_init_params is None:
            self.params.arm_init_params = np.zeros(
                len(self.params.arm_joints), dtype=np.float32
            )

        # set the camera parameters if provided
        self._cameras = None
        if hasattr(self.params, "cameras"):
            self._cameras = defaultdict(list)
            for camera_prefix in self.params.cameras:
                for sensor_name in self._sim._sensors:
                    if sensor_name.startswith(camera_prefix):
                        self._cameras[camera_prefix].append(sensor_name)

    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        # TODO: The current implementation requires users to define all the components of the robot in a single URDF.
        # The future will support loading multiple URDF files.
        if self.sim_obj is None or not self.sim_obj.is_alive:
            ao_mgr = self._sim.get_articulated_object_manager()
            self.sim_obj = ao_mgr.add_articulated_object_from_urdf(
                self.urdf_path,
                fixed_base=self._fixed_base,
                maintain_link_order=self._maintain_link_order,
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

        # set correct gains for arm joints
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

        # set correct gains for grippers
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
        self._update_motor_settings_cache()

        # set initial states and targets
        self.arm_joint_pos = self.params.arm_init_params
        self.gripper_joint_pos = self.params.gripper_init_params
        self._update_motor_settings_cache()

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        if self._cameras is not None and self._auto_update_sensor_transforms:
            # get the transformation
            agent_node = self._sim._default_agent.scene_node
            inv_T = agent_node.transformation.inverted()
            # update the cameras
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
                        link_trans
                        @ cam_transform
                        @ cam_info.relative_transform
                    )
                    cam_transform = inv_T @ cam_transform

                    sens_obj.node.transformation = (
                        orthonormalize_rotation_shear(cam_transform)
                    )

        if self._fix_joint_values is not None:
            self.arm_joint_pos = self._fix_joint_values

        self.sim_obj.awake = True

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only arm and gripper joint motors (not gains) are reset by default, derived class should handle any other changes.
        """
        self.sim_obj.clear_joint_states()
        self.arm_joint_pos = self.params.arm_init_params
        self._fix_joint_values = None
        self.gripper_joint_pos = self.params.gripper_init_params
        self._update_motor_settings_cache()
        self.update()

    @property
    def arm_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the arm joint limits in radians"""
        # deref self vars to cut access in half
        joint_pos_indices = self.joint_pos_indices
        lower_joints_limits, upper_joint_limits = self.joint_limits
        arm_joints = self.params.arm_joints
        arm_pos_indices = [joint_pos_indices[x] for x in arm_joints]
        lower_lims = np.array(
            [lower_joints_limits[i] for i in arm_pos_indices], dtype=np.float32
        )
        upper_lims = np.array(
            [upper_joint_limits[i] for i in arm_pos_indices], dtype=np.float32
        )
        return lower_lims, upper_lims

    def ee_link_id(self, ee_index: int = 0) -> int:
        """Gets the Habitat Sim link id of the end-effector.

        :param ee_index: the end effector index for which we want the link id
        """
        if ee_index >= len(self.params.ee_links):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )
        return self.params.ee_links[ee_index]

    def ee_local_offset(self, ee_index: int = 0) -> mn.Vector3:
        """Gets the relative offset of the end-effector center from the
        end-effector link.

        :param ee_index: the end effector index for which we want the link id
        """
        if ee_index >= len(self.params.ee_offset):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )
        return self.params.ee_offset[ee_index]

    def calculate_ee_forward_kinematics(
        self, joint_state: np.ndarray, ee_index: int = 0
    ) -> np.ndarray:
        """Gets the end-effector position for the given joint state."""
        self.sim_obj.joint_positions = joint_state
        return self.ee_transform(ee_index).translation

    def calculate_ee_inverse_kinematics(
        self, ee_target_position: np.ndarray, ee_index: int = 0
    ) -> np.ndarray:
        """Gets the joint states necessary to achieve the desired end-effector
        configuration.
        """
        raise NotImplementedError(
            "Currently no implementation for generic IK."
        )

    def ee_transform(self, ee_index: int = 0) -> mn.Matrix4:
        """Gets the transformation of the end-effector location. This is offset
        from the end-effector link location.


        :param ee_index: the end effector index for which we want the link transform
        """
        if ee_index >= len(self.params.ee_links):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )

        ef_link_transform = self.sim_obj.get_link_scene_node(
            self.params.ee_links[ee_index]
        ).transformation
        ef_link_transform.translation = ef_link_transform.transform_point(
            self.ee_local_offset(ee_index)
        )
        return ef_link_transform

    def clip_ee_to_workspace(
        self, pos: np.ndarray, ee_index: int = 0
    ) -> np.ndarray:
        """Clips a 3D end-effector position within region the robot can reach."""
        return np.clip(
            pos,
            self.params.ee_constraint[ee_index, :, 0],
            self.params.ee_constraint[ee_index, :, 1],
        )

    @property
    def gripper_joint_pos(self):
        """Get the current gripper joint positions."""

        # deref self vars to cut access in half
        joint_pos_indices = self.joint_pos_indices
        gripper_joints = self.params.gripper_joints
        sim_obj_joint_pos = self.sim_obj.joint_positions

        gripper_pos_indices = (joint_pos_indices[x] for x in gripper_joints)
        return np.array(
            [sim_obj_joint_pos[i] for i in gripper_pos_indices],
            dtype=np.float32,
        )

    @gripper_joint_pos.setter
    def gripper_joint_pos(self, ctrl: List[float]):
        """Kinematically sets the gripper joints and sets the motors to target."""
        joint_positions = self.sim_obj.joint_positions
        mt = self.sim_obj.motion_type
        for i, jidx in enumerate(self.params.gripper_joints):
            if mt == MotionType.DYNAMIC:
                self._set_motor_pos(jidx, ctrl[i])
            joint_positions[self.joint_pos_indices[jidx]] = ctrl[i]
        self.sim_obj.joint_positions = joint_positions

    def set_gripper_target_state(self, gripper_state: float) -> None:
        """Set the gripper motors to a desired symmetric state of the gripper [0,1] -> [open, closed]"""
        if self.sim_obj.motion_type == MotionType.DYNAMIC:
            for i, jidx in enumerate(self.params.gripper_joints):
                delta = (
                    self.params.gripper_closed_state[i]
                    - self.params.gripper_open_state[i]
                )
                target = (
                    self.params.gripper_open_state[i] + delta * gripper_state
                )
                self._set_motor_pos(jidx, target)

    def close_gripper(self) -> None:
        """Set gripper to the close state"""
        self.set_gripper_target_state(1)

    def open_gripper(self) -> None:
        """Set gripper to the open state"""
        self.set_gripper_target_state(0)

    @property
    def is_gripper_open(self) -> bool:
        """True if all gripper joints are within eps of the open state."""
        return (
            np.amax(
                np.abs(
                    self.gripper_joint_pos
                    - np.array(self.params.gripper_open_state)
                )
            )
            < self.params.gripper_state_eps
        )

    @property
    def is_gripper_closed(self) -> bool:
        """True if all gripper joints are within eps of the closed state."""
        return (
            np.amax(
                np.abs(
                    self.gripper_joint_pos
                    - np.array(self.params.gripper_closed_state)
                )
            )
            < self.params.gripper_state_eps
        )

    @property
    def arm_joint_pos(self):
        """Get the current arm joint positions."""

        # deref self vars to cut access in half
        joint_pos_indices = self.joint_pos_indices
        arm_joints = self.params.arm_joints
        sim_obj_joint_pos = self.sim_obj.joint_positions

        arm_pos_indices = (joint_pos_indices[x] for x in arm_joints)
        return np.array(
            [sim_obj_joint_pos[i] for i in arm_pos_indices], dtype=np.float32
        )

    @arm_joint_pos.setter
    def arm_joint_pos(self, ctrl: List[float]):
        """Kinematically sets the arm joints and sets the motors to target."""
        self._validate_arm_ctrl_input(ctrl)

        joint_positions = self.sim_obj.joint_positions

        mt = self.sim_obj.motion_type
        for i, jidx in enumerate(self.params.arm_joints):
            if mt == MotionType.DYNAMIC:
                self._set_motor_pos(jidx, ctrl[i])
            joint_positions[self.joint_pos_indices[jidx]] = ctrl[i]
        self.sim_obj.joint_positions = joint_positions

    def set_fixed_arm_joint_pos(self, fix_arm_joint_pos):
        """
        Will fix the arm to a desired position at every internal timestep. Can
        be used for kinematic arm control.
        """
        self._validate_arm_ctrl_input(fix_arm_joint_pos)
        self._fix_joint_values = fix_arm_joint_pos
        self.arm_joint_pos = fix_arm_joint_pos

    @property
    def arm_velocity(self) -> np.ndarray:
        """Get the velocity of the arm joints."""

        # deref self vars to cut access in half
        joint_dof_indices = self.joint_dof_indices
        arm_joints = self.params.arm_joints
        sim_obj_joint_vel = self.sim_obj.joint_velocities

        arm_dof_indices = (joint_dof_indices[x] for x in arm_joints)
        return np.array(
            [sim_obj_joint_vel[i] for i in arm_dof_indices],
            dtype=np.float32,
        )

    @property
    def arm_motor_pos(self):
        """Get the current target of the arm joints motors."""
        motor_targets = np.zeros(len(self.params.arm_init_params))
        for i, jidx in enumerate(self.params.arm_joints):
            motor_targets[i] = self._get_motor_pos(jidx)
        return motor_targets

    @arm_motor_pos.setter
    def arm_motor_pos(self, ctrl: List[float]) -> None:
        """Set the desired target of the arm joint motors."""
        self._validate_arm_ctrl_input(ctrl)
        if self.sim_obj.motion_type == MotionType.DYNAMIC:
            for i, jidx in enumerate(self.params.arm_joints):
                self._set_motor_pos(jidx, ctrl[i])

    @property
    def arm_motor_forces(self) -> np.ndarray:
        """Get the current torques on the arm joint motors"""
        return np.array(self.sim_obj.joint_forces)

    @arm_motor_forces.setter
    def arm_motor_forces(self, ctrl: List[float]) -> None:
        """Set the desired torques of the arm joint motors"""
        self.sim_obj.joint_forces = ctrl

    def _set_joint_pos(self, joint_idx, angle):
        # NOTE: This is pretty inefficient and should not be used iteratively
        set_pos = self.sim_obj.joint_positions
        set_pos[self.joint_pos_indices[joint_idx]] = angle
        self.sim_obj.joint_positions = set_pos

    def _validate_arm_ctrl_input(self, ctrl: List[float]):
        """
        Raises an exception if the control input is NaN or does not match the
        joint dimensions.
        """
        if len(ctrl) != len(self.params.arm_joints):
            raise ValueError("Dimensions do not match")
        if np.any(np.isnan(ctrl)):
            raise ValueError("Control is NaN")

    def _interpolate_arm_control(
        self, targs, idxs, seconds, ctrl_freq, get_observations=False
    ):
        curs = np.array([self._get_motor_pos(i) for i in idxs])
        diff = targs - curs
        T = int(seconds * ctrl_freq)
        delta = diff / T

        observations = []
        for i in range(T):
            joint_positions = self.sim_obj.joint_positions
            for j, jidx in enumerate(idxs):
                self._set_motor_pos(jidx, delta[j] * (i + 1) + curs[j])
                joint_positions[self.joint_pos_indices[jidx]] = (
                    delta[j] * (i + 1) + curs[j]
                )
            self.sim_obj.joint_positions = joint_positions
            self._sim.step_world(1 / ctrl_freq)
            if get_observations:
                observations.append(self._sim.get_sensor_observations())
        return observations
