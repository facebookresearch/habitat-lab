#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import hydra
import magnum as mn
import os

import habitat.isaacsim.isaacsim_wrapper as isaacsim_wrapper
from habitat.isaacsim.usd_visualizer import UsdVisualizer

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.key_mapping import KeyCode
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper



import habitat_sim  # unfortunately we can't import this earlier

class SpotWrapper:

    def __init__(self, sim):

        urdf_file_path = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        fixed_base = True
        aom = sim.get_articulated_object_manager()
        ao = aom.add_articulated_object_from_urdf(
            urdf_file_path,
            fixed_base,
            1.0,
            1.0,
            True,
            maintain_link_order=False,
            intertia_from_urdf=False,
        )
        ao.translation = [-2.0, 0.0, 0.0]  # [-10.0, 2.0, -2.7]
        # check removal and auto-creation
        joint_motor_settings = habitat_sim.physics.JointMotorSettings(
            position_target=0.0,
            position_gain=0.1,
            velocity_target=0.0,
            velocity_gain=0.1,
            max_impulse=1000.0,
        )
        existing_motor_ids = ao.existing_joint_motor_ids
        for motor_id in existing_motor_ids:
            ao.remove_joint_motor(motor_id)
        ao.create_all_motors(joint_motor_settings)        
        self.sim_obj = ao

        self.joint_motors = {}
        for (
            motor_id,
            joint_id,
        ) in self.sim_obj.existing_joint_motor_ids.items():
            self.joint_motors[joint_id] = (
                motor_id,
                self.sim_obj.get_joint_motor_settings(motor_id),
            )        


    def set_all_joints(self, joint_pos):

        for key in self.joint_motors:
            joint_motor = self.joint_motors[key]
            joint_motor[1].position_target = joint_pos
            self.sim_obj.update_joint_motor(
                joint_motor[0], joint_motor[1]
            )        

class AppStateIsaacSimViewer(AppState):
    """
    """

    def __init__(self, app_service: AppService):

        self._app_service = app_service
        self._sim = app_service.sim

        self._app_cfg = omegaconf_to_object(
            app_service.config.isaacsim_viewer
        )

        # todo: probably don't need video-recording stuff for this app
        self._video_output_prefix = "video"

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported
        assert not self._app_service.hitl_config.camera.first_person_mode

        self._camera_lookat_base_pos = mn.Vector3(0, 0, 0)  # mn.Vector3(-10, 0, -2.7)
        self._camera_lookat_y_offset = 1.0
        self._camera_helper.update(self._get_camera_lookat_pos(), 0.0)

        self._app_service.reconfigure_sim("data/fpss/hssd-hab-siro.scene_dataset_config.json", "102817140.scene_instance.json")

        self._spot = SpotWrapper(self._sim)
        self.add_rigid_object()

        # Either the HITL app is headless or Isaac is headless. They can't both spawn a window.
        do_isaac_headless = not self._app_service.hitl_config.experimental.headless.do_headless

        self._isaacsim_wrapper = isaacsim_wrapper.IsaacSimWrapper(headless=do_isaac_headless)
        isaac_world = self._isaacsim_wrapper.get_world()
        self._usd_visualizer = UsdVisualizer(isaac_world.stage, self._sim)
        self._isaac_spot_wrapper = isaacsim_wrapper.SpotWrapper(isaac_world, 
            env_id="env_0", usd_visualizer=self._usd_visualizer, origin=np.array([0.0, 0.0, 0.0]))        

        asset_path = "/home/eric/projects/habitat-llm/data/usd/test_scene.usda"  # "/home/eric/projects/habitat-llm/data/usd/objects/simple.usda"  # "/home/eric/projects/habitat-llm/data/usd/objects/e675a19a52fbc87aeddac5797942bd427ff340e8.usda"
        from omni.isaac.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/test_scene")
        self._usd_visualizer.on_add_reference_to_stage(usd_path=asset_path, prim_path="/World/test_scene")

        if True:
            from pxr import UsdGeom, Gf
            test_scene_root_prim = isaac_world.stage.GetPrimAtPath("/World/test_scene")
            test_scene_root_xform = UsdGeom.Xform(test_scene_root_prim)
            translate_op = test_scene_root_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f([0.0, 0.0, 0.1]))

        isaac_world.reset()
        self._isaac_spot_wrapper.post_reset()

        self._conversion_method_id = 0

        blah = 0


    def add_rigid_object(self):
        sim = self._sim
        obj_templates_mgr = sim.get_object_template_manager()
        # get the rigid object manager, which provides direct
        # access to objects
        rigid_obj_mgr = sim.get_rigid_object_manager()

        data_path = "./data/"
        obj_templates_mgr.load_configs(str(os.path.join(data_path, "objects/example_objects")))
        chefcan_template_handle = obj_templates_mgr.get_template_handles(
            "data/objects/example_objects/chefcan"
        )[0]

        # drop some dynamic objects
        chefcan_1 = rigid_obj_mgr.add_object_by_template_handle(chefcan_template_handle)
        chefcan_1.translation = [-10.0, 1.0, -2.2]



    def draw_world_origin(self):

        gui_drawer = self._app_service.gui_drawer

        gui_drawer.draw_transformed_line(mn.Vector3(0, 0, 0), mn.Vector3(1, 0, 0), 
            from_color=mn.Color3(255, 0, 0), to_color=mn.Color3(255, 0, 0))
        gui_drawer.draw_transformed_line(mn.Vector3(0, 0, 0), mn.Vector3(0, 1, 0), 
            from_color=mn.Color3(0, 255, 0), to_color=mn.Color3(0, 255, 0))
        gui_drawer.draw_transformed_line(mn.Vector3(0, 0, 0), mn.Vector3(0, 0, 1), 
            from_color=mn.Color3(0, 0, 255), to_color=mn.Color3(0, 0, 255))
                                         

    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        return controls_str

    def _get_status_text(self):
        status_str = ""
        lookat_pos = self._get_camera_lookat_pos()
        status_str += f"({lookat_pos.x:.1f}, {lookat_pos.y:.1f}, {lookat_pos.z:.1f})\n"
        return status_str

    def _update_help_text(self):

        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-120,
            )

    def _get_camera_lookat_pos(self):
        return self._camera_lookat_base_pos + mn.Vector3(0, self._camera_lookat_y_offset, 0)

    def _update_camera_lookat_base_pos(self):

        gui_input = self._app_service.gui_input
        y_speed = 0.05
        if gui_input.get_key_down(KeyCode.Z):
            self._camera_lookat_y_offset -= y_speed
        if gui_input.get_key_down(KeyCode.X):
            self._camera_lookat_y_offset += y_speed

        xz_forward = self._camera_helper.get_xz_forward()
        xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
        speed = self._app_cfg.camera_move_speed * self._camera_helper.get_cam_zoom_dist()
        if gui_input.get_key(KeyCode.W):
            self._camera_lookat_base_pos += (
                xz_forward * speed
            )
        if gui_input.get_key(KeyCode.S):
            self._camera_lookat_base_pos -= (
                xz_forward * speed
            )
        if gui_input.get_key(KeyCode.E):
            self._camera_lookat_base_pos += (
                xz_right * speed
            )
        if gui_input.get_key(KeyCode.Q):
            self._camera_lookat_base_pos -= (
                xz_right * speed
            )

        if gui_input.get_key_down(KeyCode.Y):
            self._conversion_method_id = (self._conversion_method_id - 1) % 16
        if gui_input.get_key_down(KeyCode.U):
            self._conversion_method_id = (self._conversion_method_id + 1) % 16

    def update_isaac(self, post_sim_update_dict):
        if self._isaacsim_wrapper:
            sim_app = self._isaacsim_wrapper.get_simulation_app()
            if not sim_app.is_running():
                post_sim_update_dict["application_exit"] = True
            else:
                sim_app.update()
                self._usd_visualizer.flush_to_hab_sim(self._conversion_method_id)

    def sim_update(self, dt, post_sim_update_dict):
        
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        new_joint_pos = (self._app_service.get_anim_fraction() - 0.5) * 0.3
        self._spot.set_all_joints(new_joint_pos)

        if gui_input.get_key(KeyCode.SPACE):
            self._sim.step_physics(dt=1.0/60)

        self._update_camera_lookat_base_pos()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform


        self.update_isaac(post_sim_update_dict)

        # draw lookat ring
        lookat_ring_radius = 0.1
        lookat_ring_color = mn.Color3(1, 0.75, 0)
        self._app_service.gui_drawer.draw_circle(
            self._camera_lookat_base_pos,
            lookat_ring_radius,
            lookat_ring_color,
        )

        self.draw_world_origin()

        self._update_help_text()


@hydra.main(
    version_base=None, config_path="./", config_name="isaacsim_viewer"
)
def main(config):
    hitl_main(config, lambda app_service: AppStateIsaacSimViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
