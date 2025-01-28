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
        sim=None
        # limit_robo_joints: bool = True,
        # fixed_base: bool = True,
        # maintain_link_order: bool = False,
        # base_type="mobile",
    ):
        self._sim = sim
        self._robot_wrapper = SpotRobotWrapper(isaac_service=isaac_service, instance_id=0)
        # Modify here the params:
       
        self.params = params
        
        # TODO: this should move later, cameras should not be attached to agents
        # @alexclegg
        
        self._cameras = None
        if hasattr(self.params, "cameras"):
            from collections import defaultdict

            self._cameras = defaultdict(list)
            for camera_prefix in self.params.cameras:
                for sensor_name in self._sim._sensors:
                    if sensor_name.startswith(camera_prefix):
                        self._cameras[camera_prefix].append(sensor_name)


    def reconfigure(self) -> None:
        """Instantiates the robot the scene. Loads the URDF, sets initial state of parameters, joints, motors, etc..."""
        # todo
        pass

    def update(self) -> None:
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        """Updates the camera transformations and performs necessary checks on
        joint limits and sleep states.
        """
        if self._cameras is not None:
            # get the transformation
            agent_node = self._sim._default_agent.scene_node
            inv_T = agent_node.transformation.inverted()
            # update the cameras
            sim = self._sim
            look_up = mn.Vector3(0,1,0)
        
            for cam_prefix, sensor_names in self._cameras.items():
                for sensor_name in sensor_names:
                    sens_obj = self._sim._sensors[sensor_name]._sensor_object
                    cam_info = self.params.cameras[cam_prefix]
                    agent = sim.agents_mgr._all_agent_data[0].articulated_agent
                    look_at = sim.agents_mgr._all_agent_data[0].articulated_agent.base_pos
                    
                    if cam_info.attached_link_id == -1:
                        link_trans = agent.base_transformation
                    else:
                        link_trans = agent.get_link_transform(cam_info.attached_link_id+1)
                    if cam_info.cam_look_at_pos == mn.Vector3(0, 0, 0):
                        pos = cam_info.cam_offset_pos
                        ori = cam_info.cam_orientation
                        Mt = mn.Matrix4.translation(pos)
                        Mz = mn.Matrix4.rotation_z(mn.Rad(ori[2]))
                        My = mn.Matrix4.rotation_y(mn.Rad(ori[1]))
                        Mx = mn.Matrix4.rotation_x(mn.Rad(ori[0]))
                        cam_transform_rel = Mt @ Mz @ My @ Mx
                    else:
                        cam_transform_rel = mn.Matrix4.look_at(
                            cam_info.cam_offset_pos,
                            cam_info.cam_look_at_pos,
                            mn.Vector3(0, 1, 0),
                        )
                    
                    cam_transform = (
                        link_trans
                        @ cam_transform_rel
                        @ cam_info.relative_transform
                    )
                    
                    
                    sens_obj.node.transformation = (
                        cam_transform
                    )
                    
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


