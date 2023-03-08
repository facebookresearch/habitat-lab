# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from habitat_sim.physics import ManagedBulletArticulatedObject


class ArticulatedAgentInterface(ABC):
    """Generic robot interface defines standard API functions."""

    def __init__(self):
        """Initializes this wrapper, but does not instantiate the robot."""
        # the Habitat ArticulatedObject API access wrapper
        self.sim_obj: ManagedBulletArticulatedObject = None

    def get_robot_sim_id(self) -> int:
        """Get the unique id for referencing the robot."""
        return self.sim_obj.object_id

    @abstractmethod
    def update(self):
        """Updates any properties or internal systems for the robot such as camera transformations, joint limits, and sleep states."""

    @abstractmethod
    def reset(self):
        """Reset the joint and motor states of an existing robot."""

    @abstractmethod
    def reconfigure(self):
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""

    def get_link_and_joint_names(self) -> str:
        """Get a string listing all robot link and joint names for debugging purposes."""
        link_joint_names = ""
        # print relevant joint/link info for debugging
        for link_id in self.sim_obj.get_link_ids():
            link_joint_names += f"{link_id} = {self.sim_obj.get_link_name(link_id)} | {self.sim_obj.get_link_joint_name(link_id)} :: type = {self.sim_obj.get_link_joint_type(link_id)} \n"
        return link_joint_names

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

    def _validate_joint_idx(self, joint):
        if joint not in self.joint_motors:
            raise ValueError(
                f"Requested joint {joint} not in joint motors with indices (keys {self.joint_motors.keys()}) and {self.joint_motors}"
            )

    def _get_motor_pos(self, joint):
        self._validate_joint_idx(joint)
        return self.joint_motors[joint][1].position_target

    def _set_motor_pos(self, joint, ctrl):
        self._validate_joint_idx(joint)
        self.joint_motors[joint][1].position_target = ctrl
        self.sim_obj.update_joint_motor(
            self.joint_motors[joint][0], self.joint_motors[joint][1]
        )

    def _capture_articulated_agent_state(self):
        return {
            "forces": self.sim_obj.joint_forces,
            "vel": self.sim_obj.joint_velocities,
            "pos": self.sim_obj.joint_positions,
        }
