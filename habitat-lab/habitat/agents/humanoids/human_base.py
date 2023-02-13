# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np


from habitat_sim.simulator import Simulator

import habitat_sim.physics as phy
from habitat_sim.utils.common import orthonormalize_rotation_shear
from habitat.agents import AgentInterface

class Humanoid(AgentInterface):
    """ Generic manipulator for the human """

    def __init__(
        self,
        params,
        urdf_path: str,
        sim: Simulator,
        sim_obj=None,
        **kwargs,
    ):
        # Manipulator.__init__(
        #     self,
        #     urdf_path=urdf_path,
        #     params=params,
        #     sim=sim
        # )
        self.params = params
        self.urdf_path = urdf_path
        self._sim = sim
        self.sim_obj = sim_obj

        
        self.joint_rotation = None
        self.root_position = None
        self.root_orientation = None

        # Joint index to position
        # self.joint_position = {}
        # self.joint_orientation = {}

        self.joint_pos_indices = {}

        # set the camera parameters if provided
        self._cameras = None
        if hasattr(self.params, "cameras"):
            self._cameras = defaultdict(list)
            for camera_prefix in self.params.cameras:
                for sensor_name in self._sim._sensors:
                    if sensor_name.startswith(camera_prefix):
                        self._cameras[camera_prefix].append(sensor_name)

    def reconfigure(self) -> None:
        # Manipulator.reconfigure(self)
        """Instantiates the humanoid in the scene. Loads the URDF, sets initial state of parameters, joints, etc..."""
        if self.sim_obj is None or not self.sim_obj.is_alive:
            ao_mgr = self._sim.get_articulated_object_manager()
            # TODO: is fixed_base here neededs
            self.sim_obj = ao_mgr.add_articulated_object_from_urdf(
                self.urdf_path, fixed_base=False
            )
            # TODO: is it right to do it here
            self.sim_obj.motion_type = phy.MotionType.KINEMATIC
        # breakpoint()
        for link_id in self.sim_obj.get_link_ids():
            self.joint_pos_indices[
                link_id
            ] = self.sim_obj.get_link_joint_pos_offset(link_id)
            # self.joint_dof_indices[link_id] = self.sim_obj.get_link_dof_offset(
            #     link_id
            # )
        # self.joint_limits = self.sim_obj.joint_position_limits

        # self._update_motor_settings_cache()



    def update(self) -> None:
        if self._cameras is not None:
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
        self.sim_obj.awake = True

    def reset(self) -> None:
        self.sim_obj.clear_joint_states()
        self.update()
        # Manipulator.reset(self)
        pass

    @property
    def base_transformation(self):
        return self.sim_obj.transformation

    def is_base_link(self, link_id: int) -> bool:
        return (
            self.sim_obj.get_link_name(link_id) in self.params.base_link_names
        )

    @property
    def base_pos(self):
        return self.sim_obj.translation - self.params.base_offset
        

    @base_pos.setter
    def base_pos(self, position: mn.Vector3):
        
        self.sim_obj.translation = position + self.params.base_offset
        """Set the robot base to a desired ground position (e.g. NavMesh point)"""
        # raise NotImplementedError("The base type is not implemented.")

    @property
    def base_rot(self):
        return self.root_orientation


    @base_rot.setter
    def base_rot(self, rotation_y_rad: float):
        self.sim_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(rotation_y_rad), mn.Vector3(0, 1, 0)
        )

    @property
    def root_pos(self):
        return self.root_position

    # TODO: repetitive here iwth base_rot
    @property
    def root_rot(self):
        return self.root_orientation


    @property
    def joint_rot(self):
        return self.joint_rotation
