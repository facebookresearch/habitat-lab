#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import hydra
import magnum as mn
import os
import json

# import habitat.isaacsim.isaacsim_wrapper as isaacsim_wrapper
# from habitat.isaacsim.usd_visualizer import UsdVisualizer
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
from habitat.isaac_sim.usd_visualizer import UsdVisualizer
from habitat.isaac_sim import isaac_prim_utils

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.core.gui_input import GuiInput

import traceback

import habitat_sim  # unfortunately we can't import this earlier




class SpotPickHelper:

    APPROACH_DIST = 0.12
    APPROACH_DURATION = 3.0

    def __init__(self):

        self._is_ik_active = False
        self._approach_timer = 0.0
        self._was_ik_active = False
        pass

    def update(self, dt, target_rel_pos):

        approach_progress = (self._approach_timer / SpotPickHelper.APPROACH_DURATION)
        # approach_offset drops to zero during approach
        approach_offset = SpotPickHelper.APPROACH_DIST * (1.0 - approach_progress)

        offset_to_shoulder = mn.Vector3(0.29, 0.0, 0.18)
        target_rel_pos -= offset_to_shoulder

        # todo: do this offset in the xy plane. Otherwise approaches to high or low
        # targets don't come in horizontally.
        target_rel_pos_len = target_rel_pos.length()
        target_rel_pos_norm = target_rel_pos.normalized()
        eps = 0.02
        adjusted_target_rel_pos_len = max(eps, 
            target_rel_pos_len - approach_offset)
        adjusted_target_rel_pos = target_rel_pos_norm * adjusted_target_rel_pos_len
    
        adjusted_target_rel_pos += offset_to_shoulder

        # Make it less likely to activate ik when we're early in our approach.
        stickiness_distance_fudge = approach_offset * 1.2

        from habitat.isaac_sim.spot_arm_ik_helper import spot_arm_ik_helper # type: ignore
        is_ik_active, target_arm_joint_positions = spot_arm_ik_helper(adjusted_target_rel_pos, stickiness_distance_fudge)

        self._was_ik_active = is_ik_active

        if is_ik_active:
            self._approach_timer = min(self._approach_timer + dt, SpotPickHelper.APPROACH_DURATION)
        else:
            self._approach_timer = 0.0

        should_close_grasp = self._approach_timer == 0.0 or self._approach_timer == SpotPickHelper.APPROACH_DURATION
        target_arm_joint_positions[7] = 0.0 if should_close_grasp else -1.67

        if approach_progress > 0.0:
            print(f"approach_progress: {approach_progress}")

        return target_arm_joint_positions



def multiply_transforms(rot_a, pos_a, rot_b, pos_b):
    out_pos = rot_b.transform_vector(pos_a) + pos_b
    out_rot = rot_b * rot_a
    return (out_rot, out_pos)

def load_function_from_file(filepath, function_name, globals_dict=None, do_print_errors=False):
    """
    Dynamically load a function from a Python script file.

    :param filepath: Path to the Python file containing the function.
    :param function_name: Name of the function to load from the file.
    :param globals_dict: Optional dictionary to use as the global namespace.
    :return: Loaded function or None if there was an error.
    """
    if not os.path.exists(filepath):
        if do_print_errors:
            print(f"Error: File {filepath} does not exist.")
        return None

    try:
        # Read the content of the file
        with open(filepath, 'r') as file:
            code = file.read()

        # Create a new namespace for the code
        namespace = globals_dict if globals_dict is not None else {}
        
        # Compile and execute the code
        exec(compile(code, filepath, 'exec'), namespace)
        
        # Extract the function
        if function_name not in namespace:
            if do_print_errors:
                print(f"Error: Function '{function_name}' not found in {filepath}.")
            return None

        return namespace[function_name]

    except Exception as e:
        if do_print_errors:
            print(f"Error loading function from {filepath}: {e}")
            traceback.print_exc()
        return None

class HandRecord:
    def __init__(self, idx):
        self._recent_remote_pos = None
        self._idx = idx
        self._recent_receive_count = 0
        self._recorded_positions_rotations = []
        self._write_count = 0

        self._playback_file_count = 0
        self._playback_frame_idx = 0
        self._playback_json = None


# Habitat Spot Wrapper for reference
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

        self._camera_lookat_base_pos = mn.Vector3(-7, 1.0, -3.0)
        self._camera_lookat_y_offset = 0.0
        self._do_camera_follow_spot = False
        self._camera_helper.update(self._get_camera_lookat_pos(), 0.0)

        # self._app_service.reconfigure_sim("data/fpss/hssd-hab-siro.scene_dataset_config.json", "102817140.scene_instance.json")

        self._spot = SpotWrapper(self._sim)

        # Either the HITL app is headless or Isaac is headless. They can't both spawn a window.
        do_isaac_headless = not self._app_service.hitl_config.experimental.headless.do_headless

        # self._isaac_wrapper = isaacsim_wrapper.IsaacSimWrapper(headless=do_isaac_headless)
        self._isaac_wrapper = IsaacAppWrapper(self._sim, headless=do_isaac_headless)
        isaac_world = self._isaac_wrapper.service.world
        self._usd_visualizer = self._isaac_wrapper.service.usd_visualizer

        asset_path = "/home/eric/projects/habitat-lab/data/usd/scenes/102817140.usda"
        from omni.isaac.core.utils.stage import add_reference_to_stage
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/test_scene")
        self._usd_visualizer.on_add_reference_to_stage(usd_path=asset_path, prim_path="/World/test_scene")

        from habitat.isaac_sim._internal.spot_robot_wrapper import SpotRobotWrapper
        self._spot_wrapper = SpotRobotWrapper(self._isaac_wrapper.service)

        from habitat.isaac_sim._internal.metahand_robot_wrapper import MetahandRobotWrapper
        self._metahand_wrapper = MetahandRobotWrapper(self._isaac_wrapper.service)

        self.add_rigid_objects()

        if False:
            from pxr import UsdGeom, Gf
            test_scene_root_prim = isaac_world.stage.GetPrimAtPath("/World/test_scene")
            test_scene_root_xform = UsdGeom.Xform(test_scene_root_prim)
            translate_op = test_scene_root_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f([0.0, 0.0, 0.1]))

        isaac_world.reset()
        self._spot_wrapper.post_reset()
        self._metahand_wrapper.post_reset()

        pos_usd = isaac_prim_utils.habitat_to_usd_position([-7.0, 1.0, -2.0])
        self._spot_wrapper._robot.set_world_pose(pos_usd, [1.0, 0.0, 0.0, 0.0])

        self._hand_records = [HandRecord(idx=0), HandRecord(idx=1)]
        self._did_function_load_fail = False

        self._spot_pick_helper = SpotPickHelper()
        pass


    def add_rigid_objects(self):

        objects_to_add = [
            ("data/objects/ycb/configs/002_master_chef_can.object_config.json", mn.Vector3(-10.0, 1.0, -2.2)),
            ("data/objects/ycb/configs/003_cracker_box.object_config.json", mn.Vector3(-10.0, 1.1, -2.2)),
            ("data/objects/ycb/configs/004_sugar_box.object_config.json", mn.Vector3(-10.0, 1.2, -2.2)),
            ("data/objects/ycb/configs/008_pudding_box.object_config.json", mn.Vector3(-10.0, 1.3, -2.2)),
            ("data/objects/ycb/configs/010_potted_meat_can.object_config.json", mn.Vector3(-10.0, 1.3, -2.2)),
        ]

        # sim = self._sim
        # rigid_obj_mgr = sim.get_rigid_object_manager()
        from habitat.isaac_sim.isaac_rigid_object_manager import IsaacRigidObjectManager
        self._isaac_rom = IsaacRigidObjectManager(self._isaac_wrapper.service)
        rigid_obj_mgr = self._isaac_rom

        for handle, position in objects_to_add:
            ro = rigid_obj_mgr.add_object_by_template_handle(handle)
            ro.translation = position

    def draw_lookat(self):
        line_render = self._app_service.line_render
        lookat_ring_radius = 0.01
        lookat_ring_color = mn.Color3(1, 0.75, 0)
        self._app_service.line_render.draw_circle(
            self._get_camera_lookat_pos(),
            lookat_ring_radius,
            lookat_ring_color,
        )
        
        # trans = mn.Matrix4.translation(self._camera_lookat_base_pos)
        # line_render.push_transform(trans)
        # self.draw_axis(0.1)
        # line_render.pop_transform()    


    def draw_world_origin(self):

        line_render = self._app_service.line_render

        line_render.draw_transformed_line(mn.Vector3(0, 0, 0), mn.Vector3(1, 0, 0), 
            from_color=mn.Color3(255, 0, 0), to_color=mn.Color3(255, 0, 0))
        line_render.draw_transformed_line(mn.Vector3(0, 0, 0), mn.Vector3(0, 1, 0), 
            from_color=mn.Color3(0, 255, 0), to_color=mn.Color3(0, 255, 0))
        line_render.draw_transformed_line(mn.Vector3(0, 0, 0), mn.Vector3(0, 0, 1), 
            from_color=mn.Color3(0, 0, 255), to_color=mn.Color3(0, 0, 255))
                                         

    def _get_controls_text(self):
        controls_str: str = ""
        controls_str += "ESC: exit\n"
        controls_str += "R + mousemove: rotate camera\n"
        controls_str += "mousewheel: cam zoom\n"
        controls_str += "WASD: move Spot\n"
        controls_str += "N: next hand recording\n"
        return controls_str

    def _get_status_text(self):
        status_str = ""
        lookat_pos = self._get_camera_lookat_pos()
        status_str += f"({lookat_pos.x:.1f}, {lookat_pos.y:.1f}, {lookat_pos.z:.1f})\n"
        status_str += f"Hand playback: {self._hand_records[0]._playback_file_count}, {self._hand_records[1]._playback_file_count}"
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
        y_speed = 0.02
        if gui_input.get_key_down(GuiInput.KeyNS.Z):
            self._camera_lookat_y_offset -= y_speed
        if gui_input.get_key_down(GuiInput.KeyNS.X):
            self._camera_lookat_y_offset += y_speed

        xz_forward = self._camera_helper.get_xz_forward()
        xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
        speed = self._app_cfg.camera_move_speed * self._camera_helper.cam_zoom_dist
        if gui_input.get_key(GuiInput.KeyNS.W):
            self._camera_lookat_base_pos += (
                xz_forward * speed
            )
        if gui_input.get_key(GuiInput.KeyNS.S):
            self._camera_lookat_base_pos -= (
                xz_forward * speed
            )
        if gui_input.get_key(GuiInput.KeyNS.E):
            self._camera_lookat_base_pos += (
                xz_right * speed
            )
        if gui_input.get_key(GuiInput.KeyNS.Q):
            self._camera_lookat_base_pos -= (
                xz_right * speed
            )

    def update_isaac(self, post_sim_update_dict):
        if self._isaac_wrapper:
            sim_app = self._isaac_wrapper.service.simulation_app
            if not sim_app.is_running():
                post_sim_update_dict["application_exit"] = True
            else:
                self._isaac_wrapper.step()

    def update_spot_base(self):

        if not self._do_camera_follow_spot:
            return

        # robot_pos = isaac_prim_utils.get_pos(self._spot_wrapper._robot)
        robot_forward = isaac_prim_utils.get_forward(self._spot_wrapper._robot)

        curr_linear_vel_usd = self._spot_wrapper._robot.get_linear_velocity()
        linear_speed = 10.0
        gui_input = self._app_service.gui_input
        if gui_input.get_key(GuiInput.KeyNS.W):
            linear_vel = robot_forward * linear_speed
        elif gui_input.get_key(GuiInput.KeyNS.S):
            linear_vel = robot_forward * -linear_speed
        else:
            linear_vel = mn.Vector3(0.0, 0.0, 0.0)
        linear_vel_usd = isaac_prim_utils.habitat_to_usd_position([linear_vel.x, linear_vel.y, linear_vel.z])
        # only set usd xy vel (vel in ground plane)
        linear_vel_usd[2] = curr_linear_vel_usd[2]
        self._spot_wrapper._robot.set_linear_velocity(linear_vel_usd)

        curr_ang_vel = self._spot_wrapper._robot.get_angular_velocity()
        angular_speed = 10.0
        gui_input = self._app_service.gui_input
        if gui_input.get_key(GuiInput.KeyNS.A):
            angular_vel_z = angular_speed
        elif gui_input.get_key(GuiInput.KeyNS.D):
            angular_vel_z = -angular_speed
        else:
            angular_vel_z = 0.0
        self._spot_wrapper._robot.set_angular_velocity([curr_ang_vel[0], curr_ang_vel[1], angular_vel_z])

    def update_spot_pre_step(self, dt):

        self.update_spot_base()

        self.update_spot_arm(dt)


    def update_record_remote_hands(self):

        remote_gui_input = self._app_service.remote_gui_input
        if not remote_gui_input:
            return
        
        for hand_record in self._hand_records:

            hand_idx = hand_record._idx

            def abort_recording(hand_record):
                hand_record._recent_receive_count = remote_gui_input.get_receive_count()
                if len(hand_record._recorded_positions_rotations):
                    print(f"aborted recording for hand {hand_idx}")
                hand_record._recorded_positions_rotations = []
                hand_record._recent_remote_pos = None

            if remote_gui_input.get_receive_count() == hand_record._recent_receive_count:
                continue

            if remote_gui_input.get_receive_count() == 0:
                if hand_record._recent_receive_count != 0:
                    print("remote_gui_input.get_receive_count() == 0")
                abort_recording(hand_record)
                continue

            if remote_gui_input.get_receive_count() > hand_record._recent_receive_count + 1:
                print(f"remote_gui_input.get_receive_count(): {remote_gui_input.get_receive_count()}, hand_record._recent_receive_count: {hand_record._recent_receive_count}")
                abort_recording(hand_record)
                continue

            hand_record._recent_receive_count = remote_gui_input.get_receive_count()

            positions, rotations = remote_gui_input.get_articulated_hand_pose(hand_idx)
            assert positions and rotations

            # handle case where remote input is not updating (slow app or user is not moving their hand)
            if positions[0] == hand_record._recent_remote_pos:
                continue

            if hand_record._recent_remote_pos is not None:
                max_dist = 0.1
                dist = (positions[0] - hand_record._recent_remote_pos).length()
                if dist > max_dist:
                    print(f"dist: {dist}")
                    abort_recording(hand_record)
                    continue

            hand_record._recent_remote_pos = positions[0]

            if len(hand_record._recorded_positions_rotations) == 0:
                print(f"starting recording for hand {hand_idx}...")

            def to_position_float_list(positions):
                position_floats = []
                for pos in positions:
                    position_floats += list(pos)
                return position_floats

            def to_rotation_float_list(positions):
                rotation_floats_wxyz = []
                for rot_quat in rotations:
                    rotation_floats_wxyz += [rot_quat.scalar, *list(rot_quat.vector)]
                return rotation_floats_wxyz

            hand_record._recorded_positions_rotations.append(
                (to_position_float_list(positions), 
                to_rotation_float_list(rotations)))

            max_frames_to_record = 100
            if len(hand_record._recorded_positions_rotations) == max_frames_to_record:
                filepath = f"hand{hand_idx}_trajectory{hand_record._write_count}_{max_frames_to_record}frames.json"
                with open(filepath, 'w') as file:
                    json.dump(hand_record._recorded_positions_rotations, file, indent=4)
                hand_record._recorded_positions_rotations = []
                hand_record._write_count += 1
                hand_record._recent_remote_pos = None
                print(f"wrote {filepath}")


    def draw_axis(self, length, transform_mat=None):

        line_render = self._app_service.line_render
        if transform_mat:
            line_render.push_transform(transform_mat)
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(length, 0, 0),
            mn.Color4(1, 0, 0, 1),
            mn.Color4(1, 0, 0, 0)
        )
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(0, length, 0),
            mn.Color4(0, 1, 0, 1),
            mn.Color4(0, 1, 0, 0)
        )
        line_render.draw_transformed_line(
            mn.Vector3(0, 0, 0),
            mn.Vector3(0, 0, length),
            mn.Color4(0, 0, 1, 1),
            mn.Color4(0, 0, 1, 0)
        )
        if transform_mat:
            line_render.pop_transform()  


    def draw_hand(self, art_hand_positions, art_hand_rotations):

        line_render = self._app_service.line_render
        num_bones = len(art_hand_positions)
        for i in range(num_bones):
            bone_pos = art_hand_positions[i]
            bone_rot_quat = art_hand_rotations[i]
            trans = mn.Matrix4.from_(bone_rot_quat.to_matrix(), bone_pos)
            self.draw_axis(0.08 if i == 0 else 0.02, trans)

    def get_hand_positions_rotations(self, hand_idx):

        hand_record = self._hand_records[hand_idx]
        assert hand_record._playback_json
        position_floats = hand_record._playback_json[hand_record._playback_frame_idx][0]
        positions = [mn.Vector3(position_floats[i], position_floats[i+1], position_floats[i+2]) 
            for i in range(0, len(position_floats), 3)]
        rotation_floats = hand_record._playback_json[hand_record._playback_frame_idx][1]
        rotations = [mn.Quaternion((rotation_floats[i+1], rotation_floats[i+2], rotation_floats[i+3]), rotation_floats[i+0]) 
            for i in range(0, len(rotation_floats), 4)]
        return positions, rotations
    

    def draw_hand_helper(self, hand_idx, use_identify_root_transform=False, 
        extra_rot=mn.Quaternion.identity_init(), extra_pos=mn.Vector3(0.0, 0.0, 0.0)):

        positions, rotations = self.get_hand_positions_rotations(hand_idx)
        
        composite_rot = mn.Quaternion.identity_init()
        composite_pos = mn.Vector3(0, 0, 0)
        if use_identify_root_transform:
            root_rot_quat = rotations[0]
            root_pos = positions[0]

            composite_rot = root_rot_quat.inverted()
            composite_pos = composite_rot.transform_vector(-root_pos)

        composite_rot, composite_pos = multiply_transforms(composite_rot, composite_pos, extra_rot, extra_pos)

        for i in range(len(positions)):
            old_rot, old_pos = rotations[i], positions[i]
            new_rot, new_pos = multiply_transforms(old_rot, old_pos, composite_rot, composite_pos)
            positions[i], rotations[i] = new_pos, new_rot

        self.draw_hand(positions, rotations)

    def update_play_back_remote_hands(self):

        do_next_file = False
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(GuiInput.KeyNS.N):
            do_next_file = True

        do_pause = not gui_input.get_key(GuiInput.KeyNS.M)

        # temp display only right hand
        for hand_record in self._hand_records:

            hand_idx = hand_record._idx
            first_filepath = f"hand{hand_idx}_trajectory0_100frames.json"
            if not os.path.exists(first_filepath):
                continue

            if do_next_file:
                hand_record._playback_json = None
                hand_record._playback_file_count += 1

            if hand_record._playback_json == None:
                filepath = f"hand{hand_idx}_trajectory{hand_record._playback_file_count}_100frames.json"
                if not os.path.exists(filepath):
                    # loop to first file
                    hand_record._playback_file_count = 0
                    filepath = first_filepath

                with open(filepath, 'r') as file:
                    hand_record._playback_json = json.load(file)

                hand_record._playback_frame_idx = 0
            else:
                if not do_pause:
                    hand_record._playback_frame_idx += 1
                    if hand_record._playback_frame_idx >= len(hand_record._playback_json):
                        hand_record._playback_frame_idx = 0


    def update_spot_arm_baked(self, target_rel_pos):

        from habitat.isaac_sim.spot_arm_ik_helper import spot_arm_ik_helper
        is_ik_active, self._spot_wrapper._target_arm_joint_positions = spot_arm_ik_helper(target_rel_pos)

    def update_spot_arm(self, dt):

        target_pos = self._get_camera_lookat_pos()

        link_positions, link_rotations = self._spot_wrapper.get_link_world_poses()

        for i in range(len(link_positions)):
            rot, pos = link_rotations[i], link_positions[i]
            trans = mn.Matrix4.from_(rot.to_matrix(), pos)
            self.draw_axis(0.1, trans)

            # if i > 0:
            #     prev_pos = link_positions[i - 1]
            #     line_render.draw_transformed_line(prev_pos, pos, 
            #         from_color=mn.Color3(255, 0, 255), to_color=mn.Color3(255, 255, 0))

        def inverse_transform(pos_a, rot_b, pos_b):
            inv_pos = rot_b.inverted().transform_vector(pos_a - pos_b)    
            return inv_pos
        target_rel_pos = inverse_transform(target_pos, link_rotations[0], link_positions[0])

        # self.update_spot_arm_hot_reload(target_rel_pos)
        # self.update_spot_arm_baked(target_rel_pos)

        self._spot_wrapper._target_arm_joint_positions = self._spot_pick_helper.update(dt, target_rel_pos)

        pass

    def update_spot_arm_hot_reload(self, link_positions, link_rotations, target_pos):

        num_dof = len(self._spot_wrapper._arm_joint_indices)
        do_print_errors = not self._did_function_load_fail
        self._did_function_load_fail = False

        filepath = "./habitat-lab/habitat/isaac_sim/spot_arm_ik_helper.py"
        function_name = "spot_arm_ik_helper"  # The function you want to load

        # Load the function
        spot_arm_ik_helper = load_function_from_file(filepath, function_name, do_print_errors=do_print_errors)

        if not spot_arm_ik_helper:
            self._did_function_load_fail = True
            return

        try:
            # Call the loaded function with test arguments
            is_ik_active, result = spot_arm_ik_helper(link_positions, link_rotations, target_pos)

            if not isinstance(result, list) or not isinstance(result[0], float) or len(result) != num_dof:
                raise ValueError(f"spot_arm_ik_helper invalid return value: {result}")
            self._spot_wrapper._target_arm_joint_positions = result

        except Exception as e:
            if do_print_errors:
                print(f"Error calling the function: {e}")
                traceback.print_exc()  
            self._did_function_load_fail = True            

    def update_metahand_bones_from_art_hand_pose(self, art_hand_positions, art_hand_rotations):

        num_dof = 16
        self._metahand_wrapper._target_joint_positions = [0.0] * num_dof

        from habitat.isaac_sim.map_articulated_hand import map_articulated_hand_to_metahand_joint_positions

        result = map_articulated_hand_to_metahand_joint_positions(art_hand_positions, art_hand_rotations)
        self._metahand_wrapper._target_joint_positions = result

    def update_metahand_bones_from_art_hand_pose_hot_reload(self, art_hand_positions, art_hand_rotations):

        num_dof = 16
        self._metahand_wrapper._target_joint_positions = [0.0] * num_dof

        # remote root transform?
        if False:
            root_rot_quat = art_hand_rotations[0]
            root_pos = art_hand_positions[0]

            composite_rot = root_rot_quat.inverted()
            composite_pos = composite_rot.transform_vector(-root_pos)

            for i in range(len(art_hand_positions)):
                old_rot, old_pos = art_hand_rotations[i], art_hand_positions[i]
                new_rot, new_pos = multiply_transforms(old_rot, old_pos, composite_rot, composite_pos)
                art_hand_positions[i], art_hand_rotations[i] = new_pos, new_rot

        do_print_errors = not self._did_function_load_fail
        self._did_function_load_fail = False

        filepath = "./habitat-lab/habitat/isaac_sim/map_articulated_hand.py"
        function_name = "map_articulated_hand_to_metahand_joint_positions"  # The function you want to load

        # Load the function
        map_articulated_hand_to_metahand_joint_positions = load_function_from_file(filepath, function_name, do_print_errors=do_print_errors)

        if not map_articulated_hand_to_metahand_joint_positions:
            self._did_function_load_fail = True
            return

        try:
            # Call the loaded function with test arguments
            result = map_articulated_hand_to_metahand_joint_positions(art_hand_positions, art_hand_rotations)

            if not isinstance(result, list) or not isinstance(result[0], float) or len(result) != num_dof:
                raise ValueError(f"map_articulated_hand_to_metahand_joint_positions invalid return value: {result}")
            self._metahand_wrapper._target_joint_positions = result

        except Exception as e:
            if do_print_errors:
                print(f"Error calling the function: {e}")
                traceback.print_exc()  
            self._did_function_load_fail = True      

    def update_metahand_from_remote_hand(self, extra_rot, extra_pos):

        hand_idx = 1  # right hand
        art_hand_positions, art_hand_rotations = self.get_hand_positions_rotations(hand_idx=hand_idx)

        for i in range(len(art_hand_positions)):
            old_rot, old_pos = art_hand_rotations[i], art_hand_positions[i]
            new_rot, new_pos = multiply_transforms(old_rot, old_pos, extra_rot, extra_pos)
            art_hand_positions[i], art_hand_rotations[i] = new_pos, new_rot

        target_base_pos = art_hand_positions[0]
        target_base_rot = art_hand_rotations[0]

        c = 0.70710678118
        base_fixup_rot = mn.Quaternion([0.0, 0.0, c], -c)  # mn.Quaternion([0.5, 0.5, 0.5], -0.5)
        fixed_target_base_rot = target_base_rot * base_fixup_rot

        # trans = mn.Matrix4.from_(fixed_target_base_rot.to_matrix(), target_base_pos)
        # self.draw_axis(0.25, trans)

        self._metahand_wrapper.set_target_base_position(target_base_pos)
        self._metahand_wrapper.set_target_base_rotation(fixed_target_base_rot)

        self.update_metahand_bones_from_art_hand_pose(art_hand_positions, art_hand_rotations)





    def sim_update(self, dt, post_sim_update_dict):
        
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(GuiInput.KeyNS.ESC):
            post_sim_update_dict["application_exit"] = True

        new_joint_pos = (self._app_service.get_anim_fraction() - 0.5) * 0.3
        self._spot.set_all_joints(new_joint_pos)

        if gui_input.get_key(GuiInput.KeyNS.SPACE):
            self._sim.step_physics(dt=1.0/60)

        if not self._do_camera_follow_spot:
            self._update_camera_lookat_base_pos()

        # self.update_record_remote_hands()

        self.update_play_back_remote_hands()

        # extra_rot = mn.Quaternion([0.0, 0.0, 0.0], 1.0)  # mn.Quaternion([0.5, 0.5, 0.5], 0.5)
        # extra_pos = [-7.0, 1.0, -2.75]
        # self.draw_hand_helper(hand_idx=1, use_identify_root_transform=True,
        #     extra_rot=extra_rot,
        #     extra_pos=extra_pos)

        extra_rot = mn.Quaternion([0.0, 0.0, 0.0], 1.0)  # mn.Quaternion([0.5, 0.5, 0.5], -0.5)
        extra_pos = [-7.0, -0.5, -3.25]
        # self.draw_hand_helper(hand_idx=0, use_identify_root_transform=False,
        #     extra_rot=extra_rot,
        #     extra_pos=extra_pos)

        self.draw_hand_helper(hand_idx=1, use_identify_root_transform=False,
            extra_rot=extra_rot,
            extra_pos=extra_pos)
        
        extra_rot = mn.Quaternion([0.0, 0.0, 0.0], 1.0)  # mn.Quaternion([0.5, 0.5, 0.5], 0.5)
        extra_pos = [-7.0, -0.5, -3.0]
        self.update_metahand_from_remote_hand(extra_rot, extra_pos)

        self.update_spot_pre_step(dt)

        self.update_isaac(post_sim_update_dict)

        if self._do_camera_follow_spot:
            self._camera_lookat_base_pos = isaac_prim_utils.get_pos(self._spot_wrapper._robot)

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)
        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        # draw lookat ring
        self.draw_lookat()

        self.draw_world_origin()

        if True:
            robot_pos = isaac_prim_utils.get_pos(self._spot_wrapper._robot)
            robot_forward = isaac_prim_utils.get_forward(self._spot_wrapper._robot)

            line_render = self._app_service.line_render
            line_render.draw_transformed_line(robot_pos, robot_pos + robot_forward, 
                from_color=mn.Color3(255, 255, 0), to_color=mn.Color3(255, 0, 255))
            

        self._update_help_text()


@hydra.main(
    version_base=None, config_path="./", config_name="isaacsim_viewer"
)
def main(config):
    hitl_main(config, lambda app_service: AppStateIsaacSimViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
