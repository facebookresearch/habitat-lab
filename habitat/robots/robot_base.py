# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Optional, Tuple

import magnum as mn
import numpy as np

from habitat.robots.robot_interface import RobotInterface
from habitat_sim.physics import JointMotorSettings
from habitat_sim.simulator import Simulator


class RobotBase(RobotInterface):
    """Generic manupulator interface defines standard API functions. Robot with a controllable base."""

    def __init__(
        self,
        params,
        urdf_path: str,
        sim: Simulator,
        limit_robo_joints: bool = True,
        fixed_based: bool = True,
        base_type="mobile",
        sim_obj=None,
        **kwargs,
    ):
        r"""Constructor"""
        assert base_type in [
            "mobile",
        ], f"'{base_type}' is invalid - valid options are [mobile]. Or you write your own class."
        RobotInterface.__init__(self)
        # Assign the variables
        self.params = params
        self.urdf_path = urdf_path
        self._sim = sim
        self._limit_robo_joints = limit_robo_joints
        self._base_type = base_type
        self.sim_obj = sim_obj

        # NOTE: the follow members cache static info for improved efficiency over querying the API
        # maps joint ids to motor settings for convenience
        self.joint_motors: Dict[int, Tuple[int, JointMotorSettings]] = {}
        # maps joint ids to position index
        self.joint_pos_indices: Dict[int, int] = {}
        # maps joint ids to velocity index
        self.joint_limits: Tuple[np.ndarray, np.ndarray] = None
        # maps joint ids to velocity index
        self.joint_dof_indices: Dict[int, int] = {}
        # set the base fixed or not
        self._fixed_base = fixed_based
        # set the fixed joint values
        self._fix_joint_values: Optional[np.ndarray] = None

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
        if self.sim_obj is None or not self.sim_obj.is_alive:
            ao_mgr = self._sim.get_articulated_object_manager()
            self.sim_obj = ao_mgr.add_articulated_object_from_urdf(
                self.urdf_path, fixed_base=self._fixed_base
            )
        # set correct gains for wheels
        if (
            hasattr(self.params, "wheel_joints")
            and self.params.wheel_joints is not None
        ):
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
        pass

    def reset(self) -> None:
        pass

    @property
    def base_pos(self):
        """Get the robot base ground position"""
        # via configured local offset from origin
        if self._base_type == "mobile":
            return (
                self.sim_obj.translation
                + self.sim_obj.transformation.transform_vector(
                    self.params.base_offset
                )
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        """Set the robot base to a desired ground position (e.g. NavMesh point)"""
        # via configured local offset from origin.
        if self._base_type == "mobile":
            if len(position) != 3:
                raise ValueError("Base position needs to be three dimensions")
            self.sim_obj.translation = (
                position
                - self.sim_obj.transformation.transform_vector(
                    self.params.base_offset
                )
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    @property
    def base_rot(self) -> float:
        return self.sim_obj.rotation.angle()

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        if self._base_type == "mobile":
            self.sim_obj.rotation = mn.Quaternion.rotation(
                mn.Rad(rotation_y_rad), mn.Vector3(0, 1, 0)
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    @property
    def base_transformation(self):
        return self.sim_obj.transformation

    def is_base_link(self, link_id: int) -> bool:
        return (
            self.sim_obj.get_link_name(link_id) in self.params.base_link_names
        )

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
