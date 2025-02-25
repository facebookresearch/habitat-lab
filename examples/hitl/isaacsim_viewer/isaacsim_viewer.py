#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import traceback

import hydra
import magnum as mn
import numpy as np

import habitat_sim  # unfortunately we can't import this earlier
from habitat.isaac_sim import isaac_prim_utils

# import habitat.isaacsim.isaacsim_wrapper as isaacsim_wrapper
# from habitat.isaacsim.usd_visualizer import UsdVisualizer
from habitat.isaac_sim.isaac_app_wrapper import IsaacAppWrapper
from habitat.isaac_sim.usd_visualizer import UsdVisualizer
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import (
    omegaconf_to_object,
    register_hydra_plugins,
)
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper


def bind_physics_material_to_hierarchy(
    stage,
    root_prim,
    material_name,
    static_friction,
    dynamic_friction,
    restitution,
):

    from omni.isaac.core.materials.physics_material import PhysicsMaterial
    from pxr import UsdPhysics, UsdShade

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


class SpotStateMachine:

    def __init__(self, spot_wrapper, line_render):

        self._spot_wrapper = spot_wrapper
        self._spot_pick_helper = None
        self._raise_timer = None
        self._pause_timer = None
        self._get_pick_target_pos = None
        self._line_render = line_render
        pass

    def reset(self):
        self._spot_pick_helper = None
        self._raise_timer = None
        self._pause_timer = None
        self._get_pick_target_pos = None

        self._spot_wrapper._target_arm_joint_positions = [
            0.0,
            -2.36,
            0.0,
            2.25,
            0.0,
            1.67,
            0.0,
        ]

        pos_usd = isaac_prim_utils.habitat_to_usd_position(
            [-4.0, 0.8, -3.5]
        )
        self._spot_wrapper._robot.set_world_pose(pos_usd, [1.0, 0.0, 0.0, 0.0])

    def set_pick_target(self, get_pick_target_pos):

        self.reset()
        self._get_pick_target_pos = get_pick_target_pos

    def _update_nav(self, robot_pos, target_pos):

        # temp disable because we can't hook this up to _hide_gui easily
        # self._line_render.draw_transformed_line(robot_pos, target_pos,
        #     from_color=mn.Color3(255, 0, 0), to_color=mn.Color3(0, 255, 0))

        robot_forward = isaac_prim_utils.get_forward(self._spot_wrapper._robot)

        robot_to_target = target_pos - robot_pos
        robot_to_target_xz_norm = mn.Vector3(
            robot_to_target.x, 0.0, robot_to_target.z
        ).normalized()

        # assumes both are unit length and in xz plane
        def get_angle_rads(v1, v2):
            # assert mn.math.isclose(v1.y, 0.0), "v1 must be in the XZ plane."
            # assert mn.math.isclose(v2.y, 0.0), "v2 must be in the XZ plane."
            # assert mn.math.isclose(v1.length(), 1.0), "v1 must be normalized."
            # assert mn.math.isclose(v2.length(), 1.0), "v2 must be normalized."
            dot_product = mn.math.clamp(
                mn.math.dot(v1, v2), -1.0, 1.0
            )  # Clamp for safety
            cross_y = (
                v1.x * v2.z - v1.z * v2.x
            )  # Cross product Y component in XZ plane
            # Compute the signed angle
            angle = mn.math.acos(dot_product)
            return float(angle if cross_y >= 0 else -angle)

        angle = get_angle_rads(robot_to_target_xz_norm, robot_forward)

        curr_ang_vel = self._spot_wrapper._robot.get_angular_velocity()
        angle_scalar = 10.0
        max_angular_speed = 2.0
        self._spot_wrapper._robot.set_angular_velocity(
            [
                curr_ang_vel[0],
                curr_ang_vel[1],
                mn.math.clamp(
                    angle * angle_scalar, -max_angular_speed, max_angular_speed
                ),
            ]
        )

        angle_threshold = 5 * mn.math.pi / 180.0
        if abs(angle) < angle_threshold:
            linear_speed = 2.5
        else:
            linear_speed = 0.0

        curr_linear_vel_usd = self._spot_wrapper._robot.get_linear_velocity()
        linear_vel = robot_forward * linear_speed
        linear_vel_usd = isaac_prim_utils.habitat_to_usd_position(
            [linear_vel.x, linear_vel.y, linear_vel.z]
        )
        # only set usd xy vel (vel in ground plane)
        linear_vel_usd[2] = curr_linear_vel_usd[2]
        self._spot_wrapper._robot.set_linear_velocity(linear_vel_usd)

    def _update_pick(self, dt, target_pos):

        base_pos, base_rot = self._spot_wrapper.get_root_pose()

        def inverse_transform(pos_a, rot_b, pos_b):
            inv_pos = rot_b.inverted().transform_vector(pos_a - pos_b)
            return inv_pos

        target_rel_pos = inverse_transform(target_pos, base_rot, base_pos)

        self._spot_wrapper._target_arm_joint_positions = (
            self._spot_pick_helper.update(dt, target_rel_pos)
        )

    def _dist_xz(self, robot_pos, target_pos):

        robot_to_target = target_pos - robot_pos
        robot_to_target_xz = mn.Vector3(
            robot_to_target.x, 0.0, robot_to_target.z
        )
        dist_xz = robot_to_target_xz.length()
        return dist_xz

    def update(self, dt):

        if not self._get_pick_target_pos:
            return

        if self._pause_timer is None:
            self._pause_timer = 0.0
        self._pause_timer += dt
        pause_duration = 2.0
        if self._pause_timer < pause_duration:
            return

        target_pos = self._get_pick_target_pos()

        robot_pos = isaac_prim_utils.get_pos(self._spot_wrapper._robot)

        dist_xz = self._dist_xz(robot_pos, target_pos)

        approach_threshold = 0.8
        if not self._spot_pick_helper and dist_xz > approach_threshold:
            self._update_nav(robot_pos, target_pos)
            return

        if not self._spot_pick_helper:
            self._spot_pick_helper = SpotPickHelper(
                len(self._spot_wrapper._arm_joint_indices)
            )

        if not self._spot_pick_helper.is_done():
            self._update_pick(dt, target_pos)
            return

        if self._raise_timer is None:
            # save arm position
            self._raise_timer = 0.0

        self._raise_timer += dt
        raise_duration = 2.0
        if self._raise_timer < raise_duration:

            # rotate elbow up
            inc1 = -dt / raise_duration * 30 / 180 * mn.math.pi
            self._spot_wrapper._target_arm_joint_positions[3] += inc1

            # rotate wrist up
            inc2 = -dt / raise_duration * 0 / 180 * mn.math.pi
            self._spot_wrapper._target_arm_joint_positions[5] += inc2
            return

        # spot_spawn_pos = mn.Vector3(-1.2, 1.0, -5.2)

        waypoint_pos = mn.Vector3(-3.0, 0.8, -4.8)
        dist_xz = self._dist_xz(robot_pos, waypoint_pos)
        approach_threshold = 0.5
        if dist_xz > approach_threshold:
            self._update_nav(robot_pos, waypoint_pos)
            return

        pass


class SpotPickHelper:

    APPROACH_DIST = 0.16
    APPROACH_DURATION = 50.0

    def __init__(self, num_dof):

        self._approach_timer = 0.0
        self._was_ik_active = False
        assert num_dof > 0
        self._num_dof = num_dof
        self._did_function_load_fail = False
        self.reset()
        pass

    def is_done(self):
        return self._approach_timer == SpotPickHelper.APPROACH_DURATION

    def spot_arm_ik_helper_wrapper(
        self, target_rel_pos, approach_offset_len, use_conservative_reach
    ):

        hot_reload = False
        if hot_reload:
            return self.spot_arm_ik_helper_wrapper_hot_reload(
                target_rel_pos, approach_offset_len, use_conservative_reach
            )
        else:
            from habitat.isaac_sim.spot_arm_ik_helper import (
                spot_arm_ik_helper,  # type: ignore
            )

            return spot_arm_ik_helper(
                target_rel_pos, approach_offset_len, use_conservative_reach
            )

    def spot_arm_ik_helper_wrapper_hot_reload(
        self, target_rel_pos, approach_offset_len, use_conservative_reach
    ):

        num_dof = self._num_dof
        fallback_ret_val = (False, [0.0] * num_dof)

        do_print_errors = not self._did_function_load_fail
        self._did_function_load_fail = False

        filepath = "./habitat-lab/habitat/isaac_sim/spot_arm_ik_helper.py"
        function_name = "spot_arm_ik_helper"  # The function you want to load

        # Load the function
        spot_arm_ik_helper = load_function_from_file(
            filepath, function_name, do_print_errors=do_print_errors
        )

        if not spot_arm_ik_helper:
            self._did_function_load_fail = True
            return fallback_ret_val

        try:
            # Call the loaded function with test arguments
            is_ik_active, result = spot_arm_ik_helper(
                target_rel_pos, approach_offset_len, use_conservative_reach
            )

            if (
                not isinstance(result, list)
                or not isinstance(result[0], float)
                or len(result) != num_dof
            ):
                raise ValueError(
                    f"spot_arm_ik_helper invalid return value: {result}"
                )
            return is_ik_active, result

        except Exception as e:
            if do_print_errors:
                print(f"Error calling the function: {e}")
                traceback.print_exc()
            self._did_function_load_fail = True

            return fallback_ret_val

    def reset(self):
        self._approach_timer = 0.0
        self._was_ik_active = False

    def update(self, dt, target_rel_pos):

        approach_progress = (
            self._approach_timer / SpotPickHelper.APPROACH_DURATION
        )
        # approach_offset drops to zero during approach
        approach_offset_len = SpotPickHelper.APPROACH_DIST * (
            1.0 - approach_progress
        )

        # offset_to_shoulder = mn.Vector3(0.29, 0.0, 0.18)
        # target_rel_pos -= offset_to_shoulder

        # # todo: do this offset in the xy plane. Otherwise approaches to high or low
        # # targets don't come in horizontally.
        # target_rel_pos_len = target_rel_pos.length()
        # target_rel_pos_norm = target_rel_pos.normalized()
        # eps = 0.02
        # adjusted_target_rel_pos_len = max(eps,
        #     target_rel_pos_len - approach_offset)
        # adjusted_target_rel_pos = target_rel_pos_norm * adjusted_target_rel_pos_len

        # adjusted_target_rel_pos += offset_to_shoulder

        use_conservative_reach = not self._was_ik_active

        is_ik_active, target_arm_joint_positions = (
            self.spot_arm_ik_helper_wrapper(
                target_rel_pos, approach_offset_len, use_conservative_reach
            )
        )

        self._was_ik_active = is_ik_active

        if is_ik_active:
            self._approach_timer = min(
                self._approach_timer + dt, SpotPickHelper.APPROACH_DURATION
            )
        else:
            self._approach_timer = 0.0

        should_close_grasp = (
            self._approach_timer == 0.0
            or self._approach_timer == SpotPickHelper.APPROACH_DURATION
        )
        target_arm_joint_positions[7] = 0.5 if should_close_grasp else -1.67

        # temp leave grasp open
        # target_arm_joint_positions[7] = -1.0

        if approach_progress > 0.0:
            print(
                f"approach_progress: {approach_progress}",
                target_arm_joint_positions[7],
            )

        return target_arm_joint_positions


def multiply_transforms(rot_a, pos_a, rot_b, pos_b):
    out_pos = rot_b.transform_vector(pos_a) + pos_b
    out_rot = rot_b * rot_a
    return (out_rot, out_pos)


def load_function_from_file(
    filepath, function_name, globals_dict=None, do_print_errors=False
):
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
        with open(filepath, "r") as file:
            code = file.read()

        # Create a new namespace for the code
        namespace = globals_dict if globals_dict is not None else {}

        # Compile and execute the code
        exec(compile(code, filepath, "exec"), namespace)

        # Extract the function
        if function_name not in namespace:
            if do_print_errors:
                print(
                    f"Error: Function '{function_name}' not found in {filepath}."
                )
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
            self.sim_obj.update_joint_motor(joint_motor[0], joint_motor[1])


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

        self._cursor_pos = mn.Vector3(
            -7.2, 0.8, -7.7
        )  # mn.Vector3(-7.0, 1.0, -2.75)
        self._do_camera_follow_spot = False
        self._camera_helper.update(self._cursor_pos, 0.0)

        # self._app_service.reconfigure_sim("data/fpss/hssd-hab-siro.scene_dataset_config.json", "102817140.scene_instance.json")

        # self._spot = SpotWrapper(self._sim)

        # Either the HITL app is headless or Isaac is headless. They can't both spawn a window.
        do_isaac_headless = True  # not self._app_service.hitl_config.experimental.headless.do_headless

        # self._isaac_wrapper = isaacsim_wrapper.IsaacSimWrapper(headless=do_isaac_headless)
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

        # asset_path = "/home/eric/projects/habitat-lab/data/usd/scenes/102817140.usda"
        asset_path = "/home/joanne/habitat-lab/data/usd/scenes/fremont_static_objects.usda"  # YOUR_PATH
        from omni.isaac.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(
            usd_path=asset_path, prim_path="/World/test_scene"
        )
        self._usd_visualizer.on_add_reference_to_stage(
            usd_path=asset_path, prim_path="/World/test_scene"
        )

        from habitat.isaac_sim._internal.murp_robot_wrapper import (
            MurpRobotWrapper,
        )

        self._spot_wrapper = MurpRobotWrapper(self._isaac_wrapper.service)

        from habitat.isaac_sim._internal.metahand_robot_wrapper import (
            MetahandRobotWrapper,
        )

        self._metahand_wrapper = MetahandRobotWrapper(
            self._isaac_wrapper.service
        )

        self._rigid_objects = []
        self.add_or_reset_rigid_objects()
        self._pick_target_rigid_object_idx = None

        if False:
            from pxr import Gf, UsdGeom

            test_scene_root_prim = isaac_world.stage.GetPrimAtPath(
                "/World/test_scene"
            )
            test_scene_root_xform = UsdGeom.Xform(test_scene_root_prim)
            translate_op = test_scene_root_xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f([0.0, 0.0, 0.1]))

        stage = self._isaac_wrapper.service.world.stage
        prim = stage.GetPrimAtPath("/World")
        # bind_physics_material_to_hierarchy(
        #     stage=stage,
        #     root_prim=prim,
        #     material_name="my_material",
        #     static_friction=0.1,
        #     dynamic_friction=0.1,
        #     restitution=0.0,
        # )

        isaac_world.reset()
        self._spot_wrapper.post_reset()
        self._metahand_wrapper.post_reset()
        self._isaac_rom.post_reset()

        # position spot near table
        # pos_usd = isaac_prim_utils.habitat_to_usd_position([-7.9, 1.0, -6.4])
        pos_usd = isaac_prim_utils.habitat_to_usd_position([-1.2, 1.0, -5.2])
        self._spot_wrapper._robot.set_world_pose(pos_usd, [1.0, 0.0, 0.0, 0.0])

        self._hand_records = [HandRecord(idx=0), HandRecord(idx=1)]
        self._did_function_load_fail = False

        # self._spot_pick_helper = SpotPickHelper(len(self._spot_wrapper._arm_joint_indices))
        self._spot_state_machine = SpotStateMachine(
            self._spot_wrapper, self._app_service.line_render
        )
        self._spot_state_machine.reset()
        self._hide_gui = False
        self._is_recording = False

        # arbitrary spot for VR avatar (near table)
        human_pos = mn.Vector3(-3.1, 0.0, -7.6)  # mn.Vector3(-7.5, 0.0, -8.0)

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.change_humanoid_position(human_pos)
            # client_message_manager.signal_scene_change()
            # client_message_manager.update_navmesh_triangles(
            #     self._get_navmesh_triangle_vertices()
            # )

        # this seems to break set_world_pose for robots
        # if not do_isaac_headless:
        #     self.set_physics_paused(True)

        # self.set_physics_paused(True)

        self._sps_tracker = AverageRateTracker(2.0)
        self._do_pause_physics = False
        self._timer = 0.0
        self.init_mouse_raycaster()
        pass

    def add_or_reset_rigid_objects(self):

        # on dining table
        drop_pos = mn.Vector3(-3.6, 0.8, -7.22)  # mn.Vector3(-7.4, 0.8, -7.5)
        offset_vec = mn.Vector3(1.3, 0.0, 0.0)

        # above coffee table
        # drop_pos = mn.Vector3(-8.1, 0.5, -3.9)

        # middle of room
        # drop_pos = mn.Vector3(-5.4, 1.2, -3.9)

        up_vec = mn.Vector3(0.0, 1.0, 0.0)
        path_to_configs = "data/objects/ycb/configs"

        do_add = len(self._rigid_objects) == 0

        # for coffee table
        if False:
            objects_to_add = []
            object_names = [
                "024_bowl",
                "013_apple",
                "011_banana",
                "010_potted_meat_can",
                "077_rubiks_cube",
                "036_wood_block",
                "004_sugar_box",
            ]
            next_obj_idx = 0
            sp = 0.25
            for cell_y in range(5):
                for cell_x in range(3):
                    for cell_z in range(3):
                        offset_vec = mn.Vector3(
                            cell_x * sp - sp, cell_y * sp, cell_z * sp - sp
                        )
                        objects_to_add.append(
                            (
                                f"{path_to_configs}/{object_names[next_obj_idx]}.object_config.json",
                                drop_pos + offset_vec,
                            )
                        )
                        next_obj_idx = (next_obj_idx + 1) % len(object_names)

        if True:
            # for dining table
            objects_to_add = [
                (
                    f"{path_to_configs}/024_bowl.object_config.json",
                    drop_pos + offset_vec * 0.0 + up_vec * 0.0,
                ),
                #            (f"{path_to_configs}/011_banana.object_config.json", drop_pos + offset_vec * 0.01 + up_vec * 0.05),
                (
                    f"{path_to_configs}/013_apple.object_config.json",
                    drop_pos + offset_vec * -0.01 + up_vec * 0.05,
                ),
                #             (f"{path_to_configs}/011_banana.object_config.json", drop_pos + offset_vec * 0.02 + up_vec * 0.12),
                (
                    f"{path_to_configs}/013_apple.object_config.json",
                    drop_pos + offset_vec * 0.01 + up_vec * 0.1,
                ),
                (
                    f"{path_to_configs}/010_potted_meat_can.object_config.json",
                    drop_pos + offset_vec * 0.3 + up_vec * 0.0,
                ),
                (
                    f"{path_to_configs}/077_rubiks_cube.object_config.json",
                    drop_pos + offset_vec * 0.6 + up_vec * 0.1,
                ),
                (
                    f"{path_to_configs}/036_wood_block.object_config.json",
                    drop_pos + offset_vec * 0.6 + up_vec * 0.0,
                ),
                (
                    f"{path_to_configs}/004_sugar_box.object_config.json",
                    drop_pos + offset_vec * 0.9,
                ),
                (
                    f"{path_to_configs}/004_sugar_box.object_config.json",
                    drop_pos + offset_vec * 1.0,
                ),
                (
                    f"{path_to_configs}/004_sugar_box.object_config.json",
                    drop_pos + offset_vec * 1.1,
                ),
                (
                    f"{path_to_configs}/004_sugar_box.object_config.json",
                    drop_pos + offset_vec * 0.8,
                ),
                (
                    f"{path_to_configs}/010_potted_meat_can.object_config.json",
                    drop_pos + offset_vec * 0.22 + up_vec * 0.0,
                ),
                (
                    f"{path_to_configs}/010_potted_meat_can.object_config.json",
                    drop_pos + offset_vec * 0.38 + up_vec * 0.0,
                ),
            ]

        from habitat.isaac_sim.isaac_rigid_object_manager import (
            IsaacRigidObjectManager,
        )

        self._isaac_rom = IsaacRigidObjectManager(self._isaac_wrapper.service)
        rigid_obj_mgr = self._isaac_rom

        for i, (handle, position) in enumerate(objects_to_add):
            if do_add:
                ro = rigid_obj_mgr.add_object_by_template_handle(handle)
                self._rigid_objects.append(ro)
            else:
                ro = self._rigid_objects[i]

            rotation = mn.Quaternion.rotation(-mn.Deg(90), mn.Vector3.x_axis())
            trans = mn.Matrix4.from_(rotation.to_matrix(), position)
            ro.transformation = trans

    def draw_lookat(self):
        if self._hide_gui:
            return

        line_render = self._app_service.line_render
        lookat_ring_radius = 0.01
        lookat_ring_color = mn.Color3(1, 0.75, 0)
        self._app_service.line_render.draw_circle(
            self._cursor_pos,
            lookat_ring_radius,
            lookat_ring_color,
        )

        # trans = mn.Matrix4.translation(self._cursor_pos)
        # line_render.push_transform(trans)
        # self.draw_axis(0.1)
        # line_render.pop_transform()

    def draw_world_origin(self):

        if self._hide_gui:
            return

        line_render = self._app_service.line_render

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
        controls_str += "WASD: move Spot\n"
        controls_str += "N: next hand recording\n"
        controls_str += "G: toggle Spot control\n"
        controls_str += "H: toggle GUI\n"
        controls_str += "P: pause physics\n"
        controls_str += "J: reset rigid objects\n"
        controls_str += "K: start recording\n"
        controls_str += "L: stop recording\n"
        controls_str += "Y: apply force at mouse\n"
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
            status_str += self._recent_mouse_ray_hit_info['rigidBody'] + "\n"
        # status_str += f"Hand playback: {self._hand_records[0]._playback_file_count}, {self._hand_records[1]._playback_file_count}"
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
        if gui_input.get_key_down(GuiInput.KeyNS.Z):
            self._cursor_pos.y -= y_speed
        if gui_input.get_key_down(GuiInput.KeyNS.X):
            self._cursor_pos.y += y_speed

        xz_forward = self._camera_helper.get_xz_forward()
        xz_right = mn.Vector3(-xz_forward.z, 0.0, xz_forward.x)
        speed = (
            self._app_cfg.camera_move_speed * self._camera_helper.cam_zoom_dist
        )
        if gui_input.get_key(GuiInput.KeyNS.W):
            self._cursor_pos += xz_forward * speed
        if gui_input.get_key(GuiInput.KeyNS.S):
            self._cursor_pos -= xz_forward * speed
        if gui_input.get_key(GuiInput.KeyNS.D):
            self._cursor_pos += xz_right * speed
        if gui_input.get_key(GuiInput.KeyNS.A):
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

    def update_spot_base(self):

        if not self._do_camera_follow_spot:
            return

        # robot_pos = isaac_prim_utils.get_pos(self._spot_wrapper._robot)
        robot_forward = isaac_prim_utils.get_forward(self._spot_wrapper._robot)

        curr_linear_vel_usd = self._spot_wrapper._robot.get_linear_velocity()
        linear_speed = 3.5
        gui_input = self._app_service.gui_input
        if gui_input.get_key(GuiInput.KeyNS.W):
            linear_vel = robot_forward * linear_speed
        elif gui_input.get_key(GuiInput.KeyNS.S):
            linear_vel = robot_forward * -linear_speed
        else:
            linear_vel = mn.Vector3(0.0, 0.0, 0.0)
        linear_vel_usd = isaac_prim_utils.habitat_to_usd_position(
            [linear_vel.x, linear_vel.y, linear_vel.z]
        )
        # only set usd xy vel (vel in ground plane)
        linear_vel_usd[2] = curr_linear_vel_usd[2]
        self._spot_wrapper._robot.set_linear_velocity(linear_vel_usd)

        curr_ang_vel = self._spot_wrapper._robot.get_angular_velocity()
        angular_speed = 4.0
        gui_input = self._app_service.gui_input
        if gui_input.get_key(GuiInput.KeyNS.A):
            angular_vel_z = angular_speed
        elif gui_input.get_key(GuiInput.KeyNS.D):
            angular_vel_z = -angular_speed
        else:
            angular_vel_z = 0.0
        self._spot_wrapper._robot.set_angular_velocity(
            [curr_ang_vel[0], curr_ang_vel[1], angular_vel_z]
        )

    def update_spot_pre_step(self, dt):

        # enable these two, plus turn down friction, to demo spot base controller
        # self.update_spot_base()
        # self._spot_wrapper._target_arm_joint_positions = [0.0, -1.18, 0.0, 1.12, 0.0, 0.83, 0.0, 0.0]

        # self.update_spot_arm(dt)

        self._spot_state_machine.update(dt)

    def update_record_remote_hands(self):

        remote_gui_input = self._app_service.remote_gui_input
        if not remote_gui_input:
            return

        for hand_record in self._hand_records:

            hand_idx = hand_record._idx

            def abort_recording(hand_record):
                hand_record._recent_receive_count = (
                    remote_gui_input.get_receive_count()
                )
                if len(hand_record._recorded_positions_rotations):
                    print(f"aborted recording for hand {hand_idx}")
                hand_record._recorded_positions_rotations = []
                hand_record._recent_remote_pos = None

            if (
                remote_gui_input.get_receive_count()
                == hand_record._recent_receive_count
            ):
                continue

            if remote_gui_input.get_receive_count() == 0:
                if hand_record._recent_receive_count != 0:
                    print("remote_gui_input.get_receive_count() == 0")
                abort_recording(hand_record)
                continue

            if (
                remote_gui_input.get_receive_count()
                > hand_record._recent_receive_count + 1
            ):
                print(
                    f"remote_gui_input.get_receive_count(): {remote_gui_input.get_receive_count()}, hand_record._recent_receive_count: {hand_record._recent_receive_count}"
                )
                abort_recording(hand_record)
                continue

            hand_record._recent_receive_count = (
                remote_gui_input.get_receive_count()
            )

            positions, rotations = remote_gui_input.get_articulated_hand_pose(
                hand_idx
            )
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
                    rotation_floats_wxyz += [
                        rot_quat.scalar,
                        *list(rot_quat.vector),
                    ]
                return rotation_floats_wxyz

            hand_record._recorded_positions_rotations.append(
                (
                    to_position_float_list(positions),
                    to_rotation_float_list(rotations),
                )
            )

            max_frames_to_record = 100
            if (
                len(hand_record._recorded_positions_rotations)
                == max_frames_to_record
            ):
                filepath = f"hand{hand_idx}_trajectory{hand_record._write_count}_{max_frames_to_record}frames.json"
                with open(filepath, "w") as file:
                    json.dump(
                        hand_record._recorded_positions_rotations,
                        file,
                        indent=4,
                    )
                hand_record._recorded_positions_rotations = []
                hand_record._write_count += 1
                hand_record._recent_remote_pos = None
                print(f"wrote {filepath}")

    def draw_axis(self, length, transform_mat=None):

        if self._hide_gui:
            return

        line_render = self._app_service.line_render
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

    def draw_hand(self, art_hand_positions, art_hand_rotations):

        line_render = self._app_service.line_render
        num_bones = len(art_hand_positions)
        for i in range(num_bones):
            bone_pos = art_hand_positions[i]
            bone_rot_quat = art_hand_rotations[i]
            trans = mn.Matrix4.from_(bone_rot_quat.to_matrix(), bone_pos)
            self.draw_axis(0.08 if i == 0 else 0.02, trans)

    def get_art_hand_positions_rotations_from_playback(self, hand_idx):

        hand_record = self._hand_records[hand_idx]
        assert hand_record._playback_json
        position_floats = hand_record._playback_json[
            hand_record._playback_frame_idx
        ][0]
        positions = [
            mn.Vector3(
                position_floats[i],
                position_floats[i + 1],
                position_floats[i + 2],
            )
            for i in range(0, len(position_floats), 3)
        ]
        rotation_floats = hand_record._playback_json[
            hand_record._playback_frame_idx
        ][1]
        rotations = [
            mn.Quaternion(
                (
                    rotation_floats[i + 1],
                    rotation_floats[i + 2],
                    rotation_floats[i + 3],
                ),
                rotation_floats[i + 0],
            )
            for i in range(0, len(rotation_floats), 4)
        ]
        return positions, rotations

    def get_art_hand_positions_rotations(self, hand_idx):

        use_recorded = False

        if use_recorded:
            return self.get_art_hand_positions_rotations_from_playback(
                hand_idx
            )
        else:

            remote_gui_input = self._app_service.remote_gui_input
            if not remote_gui_input:
                return None, None

            positions, rotations = remote_gui_input.get_articulated_hand_pose(
                hand_idx
            )
            if not positions:
                return None, None

            return positions, rotations

    # def draw_hand_helper(self, hand_idx):

    #     positions, rotations = self.get_art_hand_positions_rotations(hand_idx)
    #     if not positions:
    #         return

    #     self.draw_hand(positions, rotations)

    def update_play_back_remote_hands(self):

        do_next_file = False
        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(GuiInput.KeyNS.N):
            do_next_file = True

        do_pause = not gui_input.get_key(GuiInput.KeyNS.M)

        # temp display only right hand
        for hand_record in self._hand_records:

            hand_idx = hand_record._idx
            first_filepath = (
                f"hand_trajectories/hand{hand_idx}_trajectory0_100frames.json"
            )

            if do_next_file:
                hand_record._playback_json = None
                hand_record._playback_file_count += 1

            if hand_record._playback_json == None:
                if not os.path.exists(first_filepath):
                    continue
                filepath = f"hand_trajectories/hand{hand_idx}_trajectory{hand_record._playback_file_count}_100frames.json"
                if not os.path.exists(filepath):
                    # loop to first file
                    hand_record._playback_file_count = 0
                    filepath = first_filepath

                with open(filepath, "r") as file:
                    hand_record._playback_json = json.load(file)

                hand_record._playback_frame_idx = 0
            else:
                if not do_pause:
                    hand_record._playback_frame_idx += 1
                    if hand_record._playback_frame_idx >= len(
                        hand_record._playback_json
                    ):
                        hand_record._playback_frame_idx = 0

    def update_spot_arm(self, dt):

        use_cursor = False
        if use_cursor:
            target_pos = self._cursor_pos
        else:
            resting_arm_joint_positions = [
                0.0
            ] * self._spot_pick_helper._num_dof

            if not self._pick_target_rigid_object_idx is not None:
                self._spot_pick_helper.reset()
                return resting_arm_joint_positions

            ro = self._rigid_objects[self._pick_target_rigid_object_idx]
            target_pos = ro.translation
            pass

        self.draw_axis(0.1, mn.Matrix4.translation(target_pos))

        base_pos, base_rot = self._spot_wrapper.get_root_pose()

        def inverse_transform(pos_a, rot_b, pos_b):
            inv_pos = rot_b.inverted().transform_vector(pos_a - pos_b)
            return inv_pos

        target_rel_pos = inverse_transform(target_pos, base_rot, base_pos)

        self._spot_wrapper._target_arm_joint_positions = (
            self._spot_pick_helper.update(dt, target_rel_pos)
        )

        pass

    def update_metahand_bones_from_art_hand_pose(
        self, art_hand_positions, art_hand_rotations
    ):

        num_dof = 16
        self._metahand_wrapper._target_joint_positions = [0.0] * num_dof

        from habitat.isaac_sim.map_articulated_hand import (
            map_articulated_hand_to_metahand_joint_positions,
        )

        result = map_articulated_hand_to_metahand_joint_positions(
            art_hand_positions, art_hand_rotations
        )
        self._metahand_wrapper._target_joint_positions = result

    def update_metahand_bones_from_art_hand_pose_hot_reload(
        self, art_hand_positions, art_hand_rotations
    ):

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
                new_rot, new_pos = multiply_transforms(
                    old_rot, old_pos, composite_rot, composite_pos
                )
                art_hand_positions[i], art_hand_rotations[i] = new_pos, new_rot

        do_print_errors = not self._did_function_load_fail
        self._did_function_load_fail = False

        filepath = "./habitat-lab/habitat/isaac_sim/map_articulated_hand.py"
        function_name = "map_articulated_hand_to_metahand_joint_positions"  # The function you want to load

        # Load the function
        map_articulated_hand_to_metahand_joint_positions = (
            load_function_from_file(
                filepath, function_name, do_print_errors=do_print_errors
            )
        )

        if not map_articulated_hand_to_metahand_joint_positions:
            self._did_function_load_fail = True
            return

        try:
            # Call the loaded function with test arguments
            result = map_articulated_hand_to_metahand_joint_positions(
                art_hand_positions, art_hand_rotations
            )

            if (
                not isinstance(result, list)
                or not isinstance(result[0], float)
                or len(result) != num_dof
            ):
                raise ValueError(
                    f"map_articulated_hand_to_metahand_joint_positions invalid return value: {result}"
                )
            self._metahand_wrapper._target_joint_positions = result

        except Exception as e:
            if do_print_errors:
                print(f"Error calling the function: {e}")
                traceback.print_exc()
            self._did_function_load_fail = True

    def update_metahand_from_art_hand(
        self, use_identify_root_transform=False, extra_rot=None, extra_pos=None
    ):

        hand_idx = 1  # right hand
        art_hand_positions, art_hand_rotations = (
            self.get_art_hand_positions_rotations(hand_idx=hand_idx)
        )
        if not art_hand_positions:
            return

        if (
            use_identify_root_transform
            or extra_rot is not None
            or extra_pos is not None
        ):
            composite_rot = mn.Quaternion.identity_init()
            composite_pos = mn.Vector3(0, 0, 0)
            if use_identify_root_transform:
                root_rot_quat = art_hand_rotations[0]
                root_pos = art_hand_positions[0]

                composite_rot = root_rot_quat.inverted()
                composite_pos = composite_rot.transform_vector(-root_pos)

            if extra_rot is None:
                extra_rot = mn.Quaternion.identity_init()
            if extra_pos is None:
                extra_pos = mn.Vector3(0, 0, 0)

            composite_rot, composite_pos = multiply_transforms(
                composite_rot, composite_pos, extra_rot, extra_pos
            )

            for i in range(len(art_hand_positions)):
                old_rot, old_pos = art_hand_rotations[i], art_hand_positions[i]
                new_rot, new_pos = multiply_transforms(
                    old_rot, old_pos, composite_rot, composite_pos
                )
                art_hand_positions[i], art_hand_rotations[i] = new_pos, new_rot

        target_base_pos = art_hand_positions[0]
        target_base_rot = art_hand_rotations[0]

        c = 0.70710678118
        base_fixup_rot = mn.Quaternion(
            [0.0, 0.0, c], -c
        )  # mn.Quaternion([0.5, 0.5, 0.5], -0.5)
        fixed_target_base_rot = target_base_rot * base_fixup_rot

        self._metahand_wrapper.set_target_base_position(target_base_pos)
        self._metahand_wrapper.set_target_base_rotation(fixed_target_base_rot)

        self.update_metahand_bones_from_art_hand_pose(
            art_hand_positions, art_hand_rotations
        )
        # self.update_metahand_bones_from_art_hand_pose_hot_reload(art_hand_positions, art_hand_rotations)

        visual_offset = mn.Vector3(0.0, 0.0, 0.25)
        for i in range(len(art_hand_positions)):
            art_hand_positions[i] += visual_offset

        self.draw_hand(art_hand_positions, art_hand_rotations)

    def set_physics_paused(self, do_pause_physics):
        self._do_pause_physics = do_pause_physics
        world = self._isaac_wrapper.service.world
        if do_pause_physics:
            world.pause()
        else:
            world.play()

    def get_vr_camera_pose(self):

        remote_gui_input = self._app_service.remote_gui_input
        if not remote_gui_input:
            return None

        pos, rot_quat = remote_gui_input.get_head_pose()
        if not pos:
            return None

        extra_rot = mn.Quaternion.rotation(mn.Deg(180), mn.Vector3.y_axis())

        # change from forward=z+ to forward=z-
        rot_quat = rot_quat * extra_rot

        return mn.Matrix4.from_(rot_quat.to_matrix(), pos)

    def handle_keys(self, dt, post_sim_update_dict):

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(GuiInput.KeyNS.ESC):
            post_sim_update_dict["application_exit"] = True

        # new_joint_pos = (self._app_service.get_anim_fraction() - 0.5) * 0.3
        # self._spot.set_all_joints(new_joint_pos)

        if gui_input.get_key(GuiInput.KeyNS.SPACE):
            self._sim.step_physics(dt=1.0 / 60)

        if gui_input.get_key_down(GuiInput.KeyNS.P):
            self.set_physics_paused(not self._do_pause_physics)

        if gui_input.get_key_down(GuiInput.KeyNS.G):
            self._do_camera_follow_spot = not self._do_camera_follow_spot

        if gui_input.get_key_down(GuiInput.KeyNS.H):
            self._hide_gui = not self._hide_gui

        if gui_input.get_key_down(GuiInput.KeyNS.J):
            self.add_or_reset_rigid_objects()

        def set_spot_pick_target(rigid_object_idx):
            self._pick_target_rigid_object_idx = rigid_object_idx

            def get_pick_target_pos():
                ro = self._rigid_objects[self._pick_target_rigid_object_idx]
                com_world = isaac_prim_utils.get_com_world(ro._rigid_prim)
                self.draw_axis(0.05, mn.Matrix4.translation(com_world))
                return com_world

            self._spot_state_machine.set_pick_target(get_pick_target_pos)

        self._timer += dt
        reset_period = 25.0
        if self._timer > reset_period:
            self._timer = 0.0
            self.add_or_reset_rigid_objects()
            # self._spot_state_machine.reset()
            # self._pick_target_rigid_object_idx = 2
            # # if self._pick_target_rigid_object_idx is None:
            # #     self._pick_target_rigid_object_idx = 2
            # # self._pick_target_rigid_object_idx += 1
            # # # iterate over range [2, 6]
            # # if self._pick_target_rigid_object_idx > 6:
            # #     self._pick_target_rigid_object_idx = 2
            # print(f"setting pick target = {self._pick_target_rigid_object_idx}")
            # set_spot_pick_target(self._pick_target_rigid_object_idx)

        # self._timer += dt
        # reset_period = 3.0
        # if self._timer > reset_period:
        #     world = self._isaac_wrapper.service.world
        #     if world.is_playing():
        #         self.add_or_reset_rigid_objects()
        #     self._timer = 0.0

        pick_target_keys = [
            GuiInput.KeyNS.ONE,
            GuiInput.KeyNS.TWO,
            GuiInput.KeyNS.THREE,
            GuiInput.KeyNS.FOUR,
            GuiInput.KeyNS.FIVE,
            GuiInput.KeyNS.SIX,
            GuiInput.KeyNS.SEVEN,
            GuiInput.KeyNS.EIGHT,
        ]
        assert len(pick_target_keys) <= len(self._rigid_objects)
        for i, key in enumerate(pick_target_keys):
            if gui_input.get_key_down(key):
                set_spot_pick_target(i)
                break

        if gui_input.get_key_down(GuiInput.KeyNS.ZERO):
            self._pick_target_rigid_object_idx = None
            self._spot_state_machine.reset()

        if gui_input.get_key_down(GuiInput.KeyNS.K):
            self._app_service.video_recorder.start_recording()
            self._is_recording = True
            self._hide_gui = True
        elif gui_input.get_key_down(GuiInput.KeyNS.L):
            self._app_service.video_recorder.stop_recording_and_save_video(
                self._video_output_prefix
            )
            self._is_recording = False
            self._hide_gui = False

    def debug_draw_rigid_objects(self):

        for ro in self._rigid_objects:
            com_world = isaac_prim_utils.get_com_world(ro._rigid_prim)
            self.draw_axis(0.05, mn.Matrix4.translation(com_world))
    def init_mouse_raycaster(self):

        self._recent_mouse_ray_hit_info = None
        pass

    def update_mouse_raycaster(self, dt):

        self._recent_mouse_ray_hit_info = None

        mouse_ray = self._app_service.gui_input.mouse_ray

        if not mouse_ray:
            return

        origin_usd = isaac_prim_utils.habitat_to_usd_position(mouse_ray.origin)
        dir_usd = isaac_prim_utils.habitat_to_usd_position(mouse_ray.direction)

        from pxr import Gf
        from omni.physx import get_physx_scene_query_interface
        hit_info = get_physx_scene_query_interface().raycast_closest(
            isaac_prim_utils.to_gf_vec3(origin_usd), 
            isaac_prim_utils.to_gf_vec3(dir_usd), 1000.0)

        if not hit_info["hit"]:
            return

        # dist = hit_info['distance']
        hit_pos_usd = hit_info['position']
        hit_normal_usd = hit_info['normal']
        hit_pos_habitat = mn.Vector3(*isaac_prim_utils.usd_to_habitat_position(hit_pos_usd))
        hit_normal_habitat = mn.Vector3(*isaac_prim_utils.usd_to_habitat_position(hit_normal_usd))
        # collision_name = hit_info['collision']
        body_name = hit_info['rigidBody']

        line_render = self._app_service.line_render

        hit_radius = 0.05        
        line_render.draw_circle(hit_pos_habitat, hit_radius, mn.Color3(255, 0, 255), 16, hit_normal_habitat)

        self._recent_mouse_ray_hit_info = hit_info

        gui_input = self._app_service.gui_input
        if gui_input.get_key_down(GuiInput.KeyNS.Y):
            force_mag = 600.0
            import carb
            # instead of hit_normal_usd, consider dir_usd
            force_vec = carb.Float3(hit_normal_usd[0] * force_mag, hit_normal_usd[1] * force_mag, hit_normal_usd[2] * force_mag)
            from omni.physx import get_physx_interface
            get_physx_interface().apply_force_at_pos(body_name, force_vec, hit_pos_usd)



    def sim_update(self, dt, post_sim_update_dict):

        self._sps_tracker.increment()

        self.handle_keys(dt, post_sim_update_dict)

        if not self._do_camera_follow_spot:
            self._update_cursor_pos()

        # self.update_record_remote_hands()

        # self.update_play_back_remote_hands()

        # extra_rot = mn.Quaternion([0.0, 0.0, 0.0], 1.0)  # mn.Quaternion([0.5, 0.5, 0.5], 0.5)
        # self.draw_hand_helper(hand_idx=1, use_identify_root_transform=True,
        #     extra_rot=extra_rot,
        #     extra_pos=extra_pos)

        # self.draw_hand_helper(hand_idx=0, use_identify_root_transform=False,
        #     extra_rot=extra_rot,
        #     extra_pos=extra_pos)

        # self.draw_hand_helper(hand_idx=1)

        # extra_pos = [-7.0, 1.0, -2.75]
        # self.update_metahand_from_art_hand(use_identify_root_transform=True, extra_pos=extra_pos)
        # self.update_metahand_from_art_hand(use_identify_root_transform=False, extra_pos=mn.Vector3(0.2, 0.00, 0.0))
        self.update_metahand_from_art_hand(
            use_identify_root_transform=False, extra_pos=None
        )

        self.update_spot_pre_step(dt)
        self.update_mouse_raycaster(dt)

        self.update_isaac(post_sim_update_dict)

        do_show_vr_cam_pose = False
        vr_cam_pose = self.get_vr_camera_pose()

        if (
            do_show_vr_cam_pose
            and not self._do_camera_follow_spot
            and vr_cam_pose
        ):
            self._cam_transform = vr_cam_pose
        else:
            lookat = (
                isaac_prim_utils.get_pos(self._spot_wrapper._robot)
                if self._do_camera_follow_spot
                else self._cursor_pos
            )

            self._camera_helper.update(lookat, dt)
            self._cam_transform = self._camera_helper.get_cam_transform()

        post_sim_update_dict["cam_transform"] = self._cam_transform

        # draw lookat ring
        self.draw_lookat()

        self.draw_world_origin()

        # draw line for Spot forward vec
        if False:
            robot_pos = isaac_prim_utils.get_pos(self._spot_wrapper._robot)
            robot_forward = isaac_prim_utils.get_forward(
                self._spot_wrapper._robot
            )

            line_render = self._app_service.line_render
            line_render.draw_transformed_line(
                robot_pos,
                robot_pos + robot_forward,
                from_color=mn.Color3(255, 255, 0),
                to_color=mn.Color3(255, 0, 255),
            )

        # self.debug_draw_rigid_objects()

        self._update_help_text()


@hydra.main(version_base=None, config_path="./", config_name="isaacsim_viewer")
def main(config):
    hitl_main(config, lambda app_service: AppStateIsaacSimViewer(app_service))


if __name__ == "__main__":
    register_hydra_plugins()
    main()
