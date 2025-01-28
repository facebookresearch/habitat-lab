# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, Tuple

import magnum as mn
import numpy as np
import quaternion

from habitat.articulated_agents.mobile_manipulator import ArticulatedAgentCameraParams
from habitat.articulated_agents.mobile_manipulator import MobileManipulatorParams
from habitat.isaac_sim.isaac_mobile_manipulator import IsaacMobileManipulator
from habitat.articulated_agents.robots.spot_robot import SpotRobot



class IsaacSpotRobot(IsaacMobileManipulator):
    """Isaac-internal wrapper for a robot.
    
    
    The goal with this wrapper is convenience but not encapsulation. See also (public) IsaacMobileManipulator, which has the goal of exposing a minimal public interface to the rest of Habitat-lab.
    """

    # todo: put most of this logic in IsaacMobileManipulator
    @property
    def base_transformation(self):
        add_rot = mn.Matrix4.rotation(
            mn.Rad(np.pi / 2), mn.Vector3(1.0, 0, 0)
        )
        base_position, base_rotation = self._robot_wrapper.get_root_pose()
        pose = mn.Matrix4.from_(base_rotation.to_matrix(), base_position)
        return pose @ add_rot

    def get_link_transform(self, link_id):
        link_positions, link_rotations = self._robot_wrapper.get_link_world_poses()
        position, rotation = link_positions[link_id], link_rotations[link_id]
        # breakpoint()
        
        pose = mn.Matrix4.from_(rotation.to_matrix(), position)
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

        assert False  # todo
        return None

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

    def __init__(
        self,
        agent_cfg,
        isaac_service,
        sim=None
    ):
        super().__init__(
            SpotRobot._get_spot_params(),
            agent_cfg,
            isaac_service,
            sim=sim
        )
