# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import magnum as mn
import numpy as np

# from habitat.articulated_agents.mobile_manipulator import (
#     ArticulatedAgentCameraParams,
#     MobileManipulatorParams,
# )
from habitat.articulated_agents.robots.murp_robot import MurpRobot
from habitat.isaac_sim._internal.murp_robot_wrapper import MurpRobotWrapper
from habitat.isaac_sim.isaac_mobile_manipulator import IsaacMobileManipulator

# import quaternion


class IsaacMurpRobot(IsaacMobileManipulator):
    """Isaac-internal wrapper for a robot.


    The goal with this wrapper is convenience but not encapsulation. See also (public) IsaacMobileManipulator, which has the goal of exposing a minimal public interface to the rest of Habitat-lab.
    """

    # todo: put most of this logic in IsaacMobileManipulator
    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(mn.Rad(np.pi / 2), mn.Vector3(1.0, 0, 0))
        base_position, base_rotation = self._robot_wrapper.get_root_pose()
        pose = mn.Matrix4.from_(base_rotation.to_matrix(), base_position)
        return pose @ add_rot

    @base_transformation.setter
    def base_transformation(self, base_transformation):
        rot = mn.Matrix4.rotation(mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0))
        base_transformation = base_transformation @ rot
        rot = mn.Quaternion.from_matrix(base_transformation.rotation())
        self._robot_wrapper.set_root_pose(base_transformation.translation, rot)
        # pose = mn.Matrix4.from_(base_rotation.to_matrix(), base_position

    def ee_transform(self):
        """ "Return the ee transformation"""
        # Get the ee_trans from ee_pose
        vec, rot = self._sim.articulated_agent._robot_wrapper.ee_pose()
        global_T = mn.Matrix4.from_(rot.to_matrix(), vec)
        return global_T

    def get_link_transform(self, link_id):
        (
            link_positions,
            link_rotations,
        ) = self._robot_wrapper.get_link_world_poses()
        position, rotation = link_positions[link_id], link_rotations[link_id]
        # breakpoint()

        pose = mn.Matrix4.from_(rotation.to_matrix(), position)
        # add_rot = mn.Matrix4.rotation(
        #     mn.Rad(-np.pi / 2), mn.Vector3(1.0, 0, 0)
        # )
        return pose

    def get_ee_local_pose(
        self, ee_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the local pose of the end-effector.

        :param ee_index: the end effector index for which we want the link transform
        """
        if ee_index >= len(self.params.ee_links):
            raise ValueError(
                "The current manipulator does not have enough end effectors"
            )

        raise NotImplementedError("Need to implement get_ee_local_pose")
        # return None

        # ee_transform = self.ee_transform()
        # base_transform = self.base_transformation
        # # Get transformation
        # base_T_ee_transform = base_transform.inverted() @ ee_transform

        # # Get the local ee location (x,y,z)
        # local_ee_location = base_T_ee_transform.translation

        # # Get the local ee orientation (roll, pitch, yaw)
        # local_ee_quat = quaternion.from_rotation_matrix(
        #     base_T_ee_transform.rotation()
        # )

        # return np.array(local_ee_location), local_ee_quat

    def __init__(self, agent_cfg, isaac_service, sim=None):
        # TODO: This should be obtained from _target_arm_joint_positions but it is not intialized here yet.
        ee_index = 19
        # arm_joints = [0, 2, 4, 6, 8, 10, 12]
        arm_joints = list(range(0, 13))

        murp_params = MurpRobot._get_murp_params()
        murp_params.arm_joints = arm_joints
        murp_params.gripper_joints = [ee_index]
        murp_params.arm_init_params = [
            2.6116285,
            1.5283098,
            1.0930868,
            -0.50559217,
            0.48147443,
            2.628784,
            -1.3962275,
        ]
        robot_wrapper = MurpRobotWrapper(
            isaac_service=isaac_service,
            instance_id=0,
            right_left_hand=agent_cfg.right_left_hand,
        )
        super().__init__(
            murp_params, agent_cfg, isaac_service, robot_wrapper, sim=sim
        )
