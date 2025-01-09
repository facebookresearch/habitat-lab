# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set

import attr
import magnum as mn
import numpy as np

from habitat.articulated_agents.mobile_manipulator import MobileManipulatorParams

from habitat.isaac_sim._internal.spot_robot_wrapper import SpotRobotWrapper
from habitat.isaac_sim import isaac_prim_utils

from omni.isaac.core.utils.types import ArticulationAction




class IsaacMobileManipulator:
    """Robot with a controllable base and arm.
    
    Exposes a minimal public interface to the rest of Habitat-lab. See also SpotRobotWrapper, which has the goal of a convenience wrapper (no encapsulation).    
    """

    def __init__(
        self,
        params: MobileManipulatorParams,
        agent_cfg,
        isaac_service,
        # limit_robo_joints: bool = True,
        # fixed_base: bool = True,
        # maintain_link_order: bool = False,
        # base_type="mobile",
    ):
        self._robot_wrapper = SpotRobotWrapper(isaac_service=isaac_service, instance_id=0)


    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        # todo
        pass

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        # todo
        pass

    def reset(self) -> None:
        """Reset the joints on the existing robot.
        NOTE: only arm and gripper joint motors (not gains) are reset by default, derived class should handle any other changes.
        """
        # todo
        pass

    @property
    def arm_joint_pos(self):
        assert False  # todo
        pass

    @arm_joint_pos.setter
    def arm_joint_pos(self, ctrl: List[float]):
        """Set joint target positions.
        
        The robot's controller and joint motors will work to reach target positions over time.
        """

        rw = self._robot_wrapper

        assert len(ctrl) == len(rw._arm_joint_indices) - 1

        rw._robot_controller.apply_action(
            ArticulationAction(joint_positions=np.array(ctrl), 
                                joint_indices=rw._arm_joint_indices[:-1]))
        # todo: think about setting joint target vel to zero?

    @property
    def base_pos(self):
        rw = self._robot_wrapper
        pos_usd, _ = rw._robot.get_world_pose()
        pos_habitat = isaac_prim_utils.usd_to_habitat_position(pos_usd)
        return mn.Vector3(*pos_habitat)

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        rw = self._robot_wrapper
        _, rotation_usd = rw._robot.get_world_pose()

        pos_usd = isaac_prim_utils.habitat_to_usd_position(position)
        rw._robot.set_world_pose(pos_usd, rotation_usd)


