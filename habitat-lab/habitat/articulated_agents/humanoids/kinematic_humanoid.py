# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl
from typing import List

import magnum as mn
import numpy as np

import habitat_sim
from habitat.articulated_agents.mobile_manipulator import (
    ArticulatedAgentCameraParams,
    MobileManipulator,
    MobileManipulatorParams,
)
from habitat.articulated_agents.utils import (
    get_articulated_agent_camera_transform_from_cam_info,
)
from habitat_sim.utils.common import orthonormalize_rotation_shear


class KinematicHumanoid(MobileManipulator):
    def _get_humanoid_params(self):
        return MobileManipulatorParams(
            arm_joints=[],  # For now we do not add arm_joints
            gripper_joints=[],
            wheel_joints=None,
            arm_init_params=None,
            gripper_init_params=None,
            gripper_closed_state=np.array([]),
            gripper_open_state=np.array([]),
            gripper_state_eps=None,
            wheel_mtr_pos_gain=None,
            wheel_mtr_vel_gain=None,
            wheel_mtr_max_impulse=None,
            ee_offset=[mn.Vector3(), mn.Vector3()],
            ee_links=[20, 39],
            ee_constraint=np.zeros((2, 2, 3)),
            cameras={
                "head": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(0.0, 0.5, 0.25),
                    cam_look_at_pos=mn.Vector3(0.0, 0.5, 0.75),
                    attached_link_id=-1,
                ),
                "third": ArticulatedAgentCameraParams(
                    cam_offset_pos=mn.Vector3(-1.2, 2.0, -1.2),
                    cam_look_at_pos=mn.Vector3(1, 0.0, 0.75),
                    attached_link_id=-2,
                ),
            },
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
            base_offset=mn.Vector3(0, -0.9, 0),
            base_link_names={
                "base_link",
            },
            ee_count=2,
        )

    def __init__(
        self, agent_cfg, sim, limit_robo_joints=False, fixed_base=False
    ):
        super().__init__(
            self._get_humanoid_params(),
            agent_cfg,
            sim,
            limit_robo_joints,
            fixed_base,
            maintain_link_order=True,
        )

        # The offset and base transform are used so that the
        # character can have different rotations and shifts but the
        # when walking but the base is not affected.
        # Base transform will move linearly when the character follows
        # a path, whereas the offset transform will be changing the position
        # to simulate different gaits
        self.offset_transform = mn.Matrix4()
        self.offset_rot = -np.pi / 2
        add_rot = mn.Matrix4.rotation(
            mn.Rad(self.offset_rot), mn.Vector3(0, 1.0, 0)
        )
        perm = mn.Matrix4.rotation(
            mn.Rad(self.offset_rot), mn.Vector3(0, 0, 1.0)
        )
        self.offset_transform_base = perm @ add_rot

        self.rest_joints = None
        self._set_rest_pose_path(agent_cfg.motion_data_path)

    def _set_rest_pose_path(self, rest_pose_path):
        """Sets the parameters that indicate the reset state of the agent. Note that this function overrides
        _get_X_params, which is used to set parameters of the robots, but the parameters are so large that
        it is better to put that on a file
        """
        with open(rest_pose_path, "rb") as f:
            rest_pose = pkl.load(f)
            rest_pose = rest_pose["stop_pose"]
        self.rest_joints = list(rest_pose["joints"].reshape(-1))

    @property
    def inverse_offset_transform(self):
        rot = self.offset_transform.rotation().transposed()
        translation = -rot * self.offset_transform.translation
        return mn.Matrix4.from_(rot, translation)

    @property
    def base_transformation(self):
        return (
            self.sim_obj.transformation
            @ self.inverse_offset_transform
            @ self.offset_transform_base
        )

    @property
    def base_pos(self):
        """Get the humanoid base ground position"""
        # via configured local offset from origin
        base_transform = self.base_transformation
        return base_transform.translation + self.params.base_offset

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        """Set the robot base to a desired ground position (e.g. NavMesh point)"""
        # via configured local offset from origin.
        # TODO: maybe this can be simplified

        if len(position) != 3:
            raise ValueError("Base position needs to be three dimensions")

        base_transform = self.base_transformation
        base_pos = position - self.params.base_offset
        base_transform.translation = base_pos
        add_rot = self.offset_transform_base.inverted()
        final_transform = base_transform @ add_rot @ self.offset_transform
        self.sim_obj.transformation = final_transform

    @property
    def base_rot(self) -> float:
        return float(self.sim_obj.rotation.angle() + mn.Rad(self.offset_rot))

    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        if self._base_type == "mobile" or self._base_type == "leg":
            angle_rot = -self.offset_rot
            self.sim_obj.rotation = mn.Quaternion.rotation(
                mn.Rad(rotation_y_rad + angle_rot), mn.Vector3(0, 1, 0)
            )
        else:
            raise NotImplementedError("The base type is not implemented.")

    def set_rest_position(self) -> None:
        """Sets the agents in a resting position"""
        if self.rest_joints is None:
            joint_list = self.sim_obj.joint_positions
        else:
            joint_list = self.rest_joints

        offset_transform = mn.Matrix4()  # self.rest_matrix
        self.set_joint_transform(
            joint_list, offset_transform, self.base_transformation
        )

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        if self._cameras is not None:
            # get the transformation
            agent_node = self._sim._default_agent.scene_node
            inv_T = agent_node.transformation.inverted()
            # update the cameras
            for cam_prefix, sensor_names in self._cameras.items():
                for sensor_name in sensor_names:
                    sens_obj = self._sim._sensors[sensor_name]._sensor_object
                    cam_info = self.params.cameras[cam_prefix]
                    cam_transform = (
                        get_articulated_agent_camera_transform_from_cam_info(
                            self, cam_info
                        )
                    )
                    cam_transform = inv_T @ cam_transform

                    sens_obj.node.transformation = (
                        orthonormalize_rotation_shear(cam_transform)
                    )

        if self._fix_joint_values is not None:
            self.arm_joint_pos = self._fix_joint_values

        self.sim_obj.awake = True

    def reconfigure(self) -> None:
        """Instantiates the human in the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        super().reconfigure()
        self.sim_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        # remove any remaining joint motors
        for motor_id in self.sim_obj.existing_joint_motor_ids:
            self.sim_obj.remove_joint_motor(motor_id)
        self.update()
        self.set_rest_position()

    def reset(self) -> None:
        super().reset()
        self.update()
        self.set_rest_position()

    def set_joint_transform(
        self,
        joint_list: List[float],
        offset_transform: mn.Matrix4,
        base_transform: mn.Matrix4,
    ) -> None:
        """Sets the joints, base and offset transform of the humanoid"""
        # TODO: should this go into articulated agent?
        self.sim_obj.joint_positions = joint_list
        self.offset_transform = offset_transform

        add_rot = self.offset_transform_base.inverted()
        final_transform = (base_transform @ add_rot) @ offset_transform

        self.sim_obj.transformation = final_transform

    def get_joint_transform(self):
        """Returns the joints and base transform of the humanoid"""
        # TODO: should this go into articulated agent?
        return self.sim_obj.joint_positions, self.sim_obj.transformation
