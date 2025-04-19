#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, List

import hydra
import magnum as mn

from habitat.isaac_sim import isaac_prim_utils
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper

# path to this example app directory
dir_path = os.path.dirname(os.path.realpath(__file__))


if TYPE_CHECKING:
    from habitat.isaac_sim.isaac_rigid_object_manager import (
        IsaacRigidObjectWrapper,
    )

# unfortunately we can't import this earlier
# import habitat_sim  # isort:skip


def bind_physics_material_to_hierarchy(
    stage,
    root_prim,
    material_name,
    static_friction,
    dynamic_friction,
    restitution,
):
    from omni.isaac.core.materials.physics_material import PhysicsMaterial
    from pxr import UsdShade

    # material_path = f"/PhysicsMaterials/{material_name}"
    # material_prim = stage.DefinePrim(material_path, "PhysicsMaterial")
    # material = UsdPhysics.MaterialAPI(material_prim)
    # material.CreateStaticFrictionAttr().Set(static_friction)
    # material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    # material.CreateRestitutionAttr().Set(restitution)

    physics_material = PhysicsMaterial(
        prim_path=f"/PhysicsMaterials/{material_name}",
        name=material_name,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )

    binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
    binding_api.Bind(
        physics_material.material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def multiply_transforms(rot_a, pos_a, rot_b, pos_b):
    out_pos = rot_b.transform_vector(pos_a) + pos_b
    out_rot = rot_b * rot_a
    return (out_rot, out_pos)


class AppStateIsaacSimViewer(AppState):
    """ """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._sim = app_service.sim

        self._app_cfg = omegaconf_to_object(app_service.config.isaacsim_viewer)

        # todo: probably don't need video-recording stuff for this app
        self._video_output_prefix = "video"

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported
        assert not self._app_service.hitl_config.camera.first_person_mode

        # initial state is looking at the table
        self._cursor_pos = mn.Vector3(-3.6, 0.8, -7.22)
        self._camera_helper.update(self._cursor_pos, 0.0)

        # Either the HITL app is headless or Isaac is headless. They can't both spawn a window.
        do_isaac_headless = (
            not self._app_service.hitl_config.experimental.headless.do_headless
        )

        self._isaac_wrapper = IsaacAppWrapper(
            self._sim, headless=do_isaac_headless
        )
        isaac_world = self._isaac_wrapper.service.world
        self._usd_visualizer = self._isaac_wrapper.service.usd_visualizer

        self._isaac_physics_dt = 1.0 / 180
        # beware goofy behavior if physics_dt doesn't equal rendering_dt
        isaac_world.set_simulation_dt(
            physics_dt=self._isaac_physics_dt,
            rendering_dt=self._isaac_physics_dt,
        )

        # load the asset from config
        asset_path = os.path.join(dir_path, self._app_cfg.usd_scene_path)

        from omni.isaac.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            usd_path=asset_path, prim_path="/World/test_scene"
        )
        self._usd_visualizer.on_add_reference_to_stage(
            usd_path=asset_path, prim_path="/World/test_scene"
        )

        self._rigid_objects: List["IsaacRigidObjectWrapper"] = []
        from habitat.isaac_sim.isaac_rigid_object_manager import (
            IsaacRigidObjectManager,
        )

        self._isaac_rom = IsaacRigidObjectManager(self._isaac_wrapper.service)
        # self.add_or_reset_rigid_objects()
        self._pick_target_rigid_object_idx = None

        stage = self._isaac_wrapper.service.world.stage
        prim = stage.GetPrimAtPath("/World")
        bind_physics_material_to_hierarchy(
            stage=stage,
            root_prim=prim,
            material_name="my_material",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        isaac_world.reset()
        self._isaac_rom.post_reset()

        self._hide_gui = False
        self._is_recording = False

        self._sps_tracker = AverageRateTracker(2.0)
        self._do_pause_physics = False
        self._timer = 0.0

        self.init_mouse_raycaster()

    def add_rigid_object(
        self, handle: str, bottom_pos: mn.Vector3 = None
    ) -> None:
        """
        Adds the specified rigid object to the scene and records it in self._rigid_objects.
        If specified, aligns the bottom most point of the object with bottom_pos.
        NOTE: intended to be used with raycasting to place objects on horizontal surfaces.
        """
        ro = self._isaac_rom.add_object_by_template_handle(handle)
        self._rigid_objects.append(ro)
        ro.rotation = mn.Quaternion.rotation(-mn.Deg(90), mn.Vector3.x_axis())

        # set a translation if specified offset such that the bottom most point of the object is coincident with bottom_pos
        if bottom_pos is not None:
            bounds = ro.get_aabb()
            obj_height = ro.translation[1] - bounds.bottom
            print(f"bounds = {bounds}")
            print(f"obj_height = {obj_height}")

            ro.translation = bottom_pos + mn.Vector3(0, obj_height, 0)

    def draw_lookat(self):
        if self._hide_gui:
            return

        lookat_ring_radius = 0.01
        lookat_ring_color = mn.Color3(1, 0.75, 0)
        self._app_service.gui_drawer.draw_circle(
            self._cursor_pos,
            lookat_ring_radius,
            lookat_ring_color,
        )

    def draw_world_origin(self):
        if self._hide_gui:
            return

        line_render = self._app_service.gui_drawer

        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(1, 0, 0),
            from_color=mn.Color3(255, 0, 0),
            to_color=mn.Color3(255, 0, 0),
        )
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(0, 1, 0),
            from_color=mn.Color3(0, 255, 0),
            to_color=mn.Color3(0, 255, 0),
        )
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(0, 0, 1),
            from_color=mn.Color3(0, 0, 255),
            to_color=mn.Color3(0, 0, 255),
        )

    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        controls_str += "R + mousemove: rotate camera\n"
        controls_str += "mousewheel: cam zoom\n"
        controls_str += "WASD: move cursor\n"
        controls_str += "H: toggle GUI\n"
        controls_str += "P: pause physics\n"
        controls_str += "J: reset rigid objects\n"
        controls_str += "K: start recording\n"
        controls_str += "L: stop recording\n"
        if self._sps_tracker.get_smoothed_rate() is not None:
            controls_str += (
                f"server SPS: {self._sps_tracker.get_smoothed_rate():.1f}\n"
            )

        return controls_str

    def _get_status_text(self):
        status_str = ""
        cursor_pos = self._cursor_pos
        status_str += (
            f"({cursor_pos.x:.1f}, {cursor_pos.y:.1f}, {cursor_pos.z:.1f})\n"
        )
        if self._recent_mouse_ray_hit_info:
            status_str += self._recent_mouse_ray_hit_info["rigidBody"] + "\n"

        return status_str

    def _update_help_text(self):
        if self._hide_gui:
            return

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

    def _update_cursor_pos(self):
        gui_input = self._app_service.gui_input
        y_speed = 0.02
        if gui_input.get_key_down(KeyCode.Z):
            self._cursor_pos.y -= y_speed
        if gui_input.get_key_down(KeyCode.X):
            self._cursor_pos.y += y_speed

        xz_forward = self._camera_helper.get_xz_forward()
        xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
        speed = (
            self._app_cfg.camera_move_speed * self._camera_helper.cam_zoom_dist
        )
        if gui_input.get_key(KeyCode.W):
            self._cursor_pos += xz_forward * speed
        if gui_input.get_key(KeyCode.S):
            self._cursor_pos -= xz_forward * speed
        if gui_input.get_key(KeyCode.D):
            self._cursor_pos += xz_right * speed
        if gui_input.get_key(KeyCode.A):
            self._cursor_pos -= xz_right * speed

    def update_isaac(self, post_sim_update_dict):
        if self._isaac_wrapper:
            sim_app = self._isaac_wrapper.service.simulation_app
            if not sim_app.is_running():
                post_sim_update_dict["application_exit"] = True
            else:
                approx_app_fps = 30
                num_steps = int(
                    1.0 / (approx_app_fps * self._isaac_physics_dt)
                )
                self._isaac_wrapper.step(num_steps=num_steps)
                self._isaac_wrapper.pre_render()

    def draw_axis(self, length, transform_mat=None):
        if self._hide_gui:
            return

        line_render = self._app_service.gui_drawer
        if transform_mat:
            line_render.push_transform(transform_mat)
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(length, 0, 0),
            mn.Color4(1, 0, 0, 1),
            mn.Color4(1, 0, 0, 0),
        )
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(0, length, 0),
            mn.Color4(0, 1, 0, 1),
            mn.Color4(0, 1, 0, 0),
        )
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(0, 0, length),
            mn.Color4(0, 0, 1, 1),
            mn.Color4(0, 0, 1, 0),
        )
        if transform_mat:
            line_render.pop_transform()

    def set_physics_paused(self, do_pause_physics):
        self._do_pause_physics = do_pause_physics
        world = self._isaac_wrapper.service.world
        if do_pause_physics:
            world.pause()
        else:
            world.play()

    def get_vr_camera_pose(self):
        remote_client_state = self._app_service.remote_client_state
        if not remote_client_state:
            return None

        pos, rot_quat = remote_client_state.get_head_pose()
        if not pos:
            return None

        extra_rot = mn.Quaternion.rotation(mn.Deg(180), mn.Vector3.y_axis())

        # change from forward=z+ to forward=z-
        rot_quat = rot_quat * extra_rot

        return mn.Matrix4.from_(rot_quat.to_matrix(), pos)

    def handle_keys(self, dt, post_sim_update_dict):
        """
        Handle key presses which are not used for camera updates.
        NOTE: wasdzxr reserved for camera UI
        """
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.ESC):
            post_sim_update_dict["application_exit"] = True

        if gui_input.get_key(KeyCode.SPACE):
            self._sim.step_physics(dt=1.0 / 60)

        if gui_input.get_key_down(KeyCode.P):
            self.set_physics_paused(not self._do_pause_physics)

        if gui_input.get_key_down(KeyCode.H):
            self._hide_gui = not self._hide_gui

        if (
            gui_input.get_key_down(KeyCode.J)
            and self._recent_mouse_ray_hit_info is not None
        ):
            # place an object at the mouse raycast endpoint
            hab_hit_pos = mn.Vector3(
                *isaac_prim_utils.usd_to_habitat_position(
                    self._recent_mouse_ray_hit_info["position"]
                )
            )
            self.add_rigid_object(
                handle="data/objects/ycb/configs/024_bowl.object_config.json",
                bottom_pos=hab_hit_pos,
            )

        if gui_input.get_key_down(KeyCode.K):
            self._app_service.video_recorder.start_recording()
            self._is_recording = True
            self._hide_gui = True

        elif gui_input.get_key_down(KeyCode.L):
            self._app_service.video_recorder.stop_recording_and_save_video(
                self._video_output_prefix
            )
            self._is_recording = False
            self._hide_gui = False

    def handle_mouse_press(self) -> None:
        """
        TODO: add mouse controls
        """
        if self._app_service.gui_input.get_mouse_button_down(
            MouseButton.RIGHT
        ):
            pass

    def debug_draw_rigid_objects(self):
        for ro in self._rigid_objects:
            com_world = isaac_prim_utils.get_com_world(ro._rigid_prim)
            self.draw_axis(0.05, mn.Matrix4.translation(com_world))

    def init_mouse_raycaster(self):
        self._recent_mouse_ray_hit_info = None

    def update_mouse_raycaster(self, dt):
        self._recent_mouse_ray_hit_info = None

        mouse_ray = self._app_service.gui_input.mouse_ray

        if not mouse_ray:
            return

        origin_usd = isaac_prim_utils.habitat_to_usd_position(mouse_ray.origin)
        dir_usd = isaac_prim_utils.habitat_to_usd_position(mouse_ray.direction)

        from omni.physx import get_physx_scene_query_interface

        hit_info = get_physx_scene_query_interface().raycast_closest(
            isaac_prim_utils.to_gf_vec3(origin_usd),
            isaac_prim_utils.to_gf_vec3(dir_usd),
            1000.0,
        )

        if not hit_info["hit"]:
            return

        # dist = hit_info['distance']
        hit_pos_usd = hit_info["position"]
        hit_normal_usd = hit_info["normal"]
        hit_pos_habitat = mn.Vector3(
            *isaac_prim_utils.usd_to_habitat_position(hit_pos_usd)
        )
        hit_normal_habitat = mn.Vector3(
            *isaac_prim_utils.usd_to_habitat_position(hit_normal_usd)
        )
        # collision_name = hit_info['collision']
        body_name = hit_info["rigidBody"]

        hit_radius = 0.05
        self._app_service.gui_drawer.draw_circle(
            hit_pos_habitat,
            hit_radius,
            mn.Color3(255, 0, 255),
            16,
            hit_normal_habitat,
        )

        self._recent_mouse_ray_hit_info = hit_info

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(KeyCode.Y):
            force_mag = 200.0
            import carb

            # instead of hit_normal_usd, consider dir_usd
            force_vec = carb.Float3(
                hit_normal_usd[0] * force_mag,
                hit_normal_usd[1] * force_mag,
                hit_normal_usd[2] * force_mag,
            )
            from omni.physx import get_physx_interface

            get_physx_interface().apply_force_at_pos(
                body_name, force_vec, hit_pos_usd
            )

    def sim_update(self, dt, post_sim_update_dict):
        self._sps_tracker.increment()

        self.handle_keys(dt, post_sim_update_dict)
        self._update_cursor_pos()
        self.update_mouse_raycaster(dt)
        self.update_isaac(post_sim_update_dict)

        do_show_vr_cam_pose = False
        vr_cam_pose = self.get_vr_camera_pose()

        if do_show_vr_cam_pose and vr_cam_pose:
            self._cam_transform = vr_cam_pose
        else:
            self._camera_helper.update(self._cursor_pos, dt)
            self._cam_transform = self._camera_helper.get_cam_transform()

        post_sim_update_dict["cam_transform"] = self._cam_transform

        # draw lookat ring
        self.draw_lookat()

        self.draw_world_origin()

        self._update_help_text()


@hydra.main(version_base=None, config_path="./", config_name="isaacsim_viewer")
def main(config):
    hitl_main(config, lambda app_service: AppStateIsaacSimViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
