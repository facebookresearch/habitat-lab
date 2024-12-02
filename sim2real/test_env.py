#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pickle as pkl
import time
from os import path as osp

import magnum as mn
import numpy as np
import pytest
from omegaconf import DictConfig
from scipy.spatial.transform import Slerp as slerp

import habitat.articulated_agents.robots.spot_robot as spot_robot
import habitat.articulated_agents.robots.spot_robot_real as spot_robot
import habitat_sim
import habitat_sim.agent
from habitat.tasks.rearrange.utils import (
    IkHelper,
    batch_transform_point,
    is_pb_installed,
    make_render_only,
    set_agent_base_via_obj_trans,
)
from habitat.utils.rotation_utils import *

default_sim_settings = {
    # settings shared by example.py and benchmark.py
    "max_frames": 1000,
    "width": 640,
    "height": 480,
    "default_agent": 0,
    "sensor_height": 1.5,
    "hfov": 90,
    "color_sensor": True,  # RGB sensor (default: ON)
    "semantic_sensor": False,  # semantic sensor (default: OFF)
    "depth_sensor": False,  # depth sensor (default: OFF)
    "ortho_rgba_sensor": False,  # Orthographic RGB sensor (default: OFF)
    "ortho_depth_sensor": False,  # Orthographic depth sensor (default: OFF)
    "ortho_semantic_sensor": False,  # Orthographic semantic sensor (default: OFF)
    "fisheye_rgba_sensor": False,
    "fisheye_depth_sensor": False,
    "fisheye_semantic_sensor": False,
    "equirect_rgba_sensor": False,
    "equirect_depth_sensor": False,
    "equirect_semantic_sensor": False,
    "seed": 1,
    "silent": False,  # do not print log info (default: OFF)
    # settings exclusive to example.py
    "save_png": False,  # save the pngs to disk (default: OFF)
    "print_semantic_scene": False,
    "print_semantic_mask_stats": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "scene": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "test_scene_data_url": "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip",
    "goal_position": [5.047, 0.199, 11.145],
    "enable_physics": False,
    "enable_gfx_replay_save": False,
    "physics_config_file": "./data/default.physics_config.json",
    "num_objects": 10,
    "test_object_index": 0,
    "frustum_culling": True,
}


class TestEnv:
    def __init__(self, fixed_base, make_video, filepath):
        self.fixed_base = fixed_base
        self.produce_debug_video = make_video
        self.real_filepath = filepath
        self.observations = []
        self.start_js = np.array([0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0])
        # self.start_js = np.array([0, -180, 180, 90, 0, -90])

        self.init_scene_and_robot()

    # build SimulatorConfiguration
    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        if "scene_dataset_config_file" in settings:
            sim_cfg.scene_dataset_config_file = settings[
                "scene_dataset_config_file"
            ]
        sim_cfg.frustum_culling = settings.get("frustum_culling", False)
        if "enable_physics" in settings:
            sim_cfg.enable_physics = settings["enable_physics"]
        if "physics_config_file" in settings:
            sim_cfg.physics_config_file = settings["physics_config_file"]
        if not settings["silent"]:
            print(
                "sim_cfg.physics_config_file = " + sim_cfg.physics_config_file
            )
        if "scene_light_setup" in settings:
            sim_cfg.scene_light_setup = settings["scene_light_setup"]
        sim_cfg.gpu_device_id = 0
        if not hasattr(sim_cfg, "scene_id"):
            raise RuntimeError(
                "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
            )
        sim_cfg.scene_id = settings["scene"]

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []

        def create_camera_spec(**kw_args):
            camera_sensor_spec = habitat_sim.CameraSensorSpec()
            camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            camera_sensor_spec.resolution = [
                settings["height"],
                settings["width"],
            ]

            camera_sensor_spec.position = [-1.5, 2.5, 1.5]
            camera_sensor_spec.orientation = [
                np.deg2rad(-20.0),
                np.deg2rad(-20.0),
                0.0,
            ]
            # camera_sensor_spec.position = [1.5, 2.5, 0.0]
            # camera_sensor_spec.orientation = [
            #     np.deg2rad(-20.0),
            #     np.deg2rad(90.0),
            #     0.0,
            # ]

            ## camera on spot's left
            # camera_sensor_spec.position = [-2.0, settings["sensor_height"], 2.0]
            for k in kw_args:
                setattr(camera_sensor_spec, k, kw_args[k])
            return camera_sensor_spec

        if settings["color_sensor"]:
            color_sensor_spec = create_camera_spec(
                uuid="color_sensor",
                hfov=settings["hfov"],
                sensor_type=habitat_sim.SensorType.COLOR,
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = create_camera_spec(
                uuid="depth_sensor",
                hfov=settings["hfov"],
                sensor_type=habitat_sim.SensorType.DEPTH,
                channels=1,
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = create_camera_spec(
                uuid="semantic_sensor",
                hfov=settings["hfov"],
                sensor_type=habitat_sim.SensorType.SEMANTIC,
                channels=1,
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(semantic_sensor_spec)

        # create agent specifications
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }

        # override action space to no-op to test physics
        if sim_cfg.enable_physics:
            agent_cfg.action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
                )
            }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def simulate(self, dt, get_observations=True):
        r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
        target_time = self.sim.get_world_time() + dt
        while self.sim.get_world_time() < target_time:
            self.sim.step_physics(1.0 / 60.0)
            if get_observations:
                self.observations.append(self.sim.get_sensor_observations())

    def visualize_position(self, position, r=0.05):
        template_mgr = self.sim.get_object_template_manager()
        rom = self.sim.get_rigid_object_manager()
        template = template_mgr.get_template_by_handle(
            template_mgr.get_template_handles("sphere")[0]
        )
        template.scale = mn.Vector3(r, r, r)
        viz_template = template_mgr.register_template(
            template, "ball_new_viz_" + str(r)
        )
        viz_obj = rom.add_object_by_template_id(viz_template)
        viz_obj.translation = mn.Vector3(*position)
        return viz_obj.object_id

    def draw_axes(self, axis_len=1.0):
        opacity = 1.0
        red = mn.Color4(1.0, 0.0, 0.0, opacity)
        green = mn.Color4(0.0, 1.0, 0.0, opacity)
        blue = mn.Color4(0.0, 0.0, 1.0, opacity)
        lr = self.sim.get_debug_line_render()
        # draw axes with x+ = red, y+ = green, z+ = blue
        lr.draw_transformed_line(self.origin, mn.Vector3(axis_len, 0, 0), red)
        lr.draw_transformed_line(
            self.origin, mn.Vector3(0, axis_len, 0), green
        )
        lr.draw_transformed_line(self.origin, mn.Vector3(0, 0, axis_len), blue)

    def set_robot_pose(self, real_position, real_rotation):
        sim_robot_pos, sim_robot_rot_matrix = transform_3d_coordinates(
            real_position, real_rotation, "real_to_sim", "matrix"
        )

        position = (
            sim_robot_pos
            - self.spot.sim_obj.transformation.transform_vector(
                self.spot.params.base_offset
            )
        )
        mn_matrix = mn.Matrix3(sim_robot_rot_matrix)

        # mn_matrix = mn.Matrix3(spot.get_rotation_matrix(real_robot_rot))
        target_trans = mn.Matrix4.from_(mn_matrix, mn.Vector3(*position))
        self.spot.sim_obj.transformation = target_trans

    def setup_scene(self):
        obj_template_mgr = self.sim.get_object_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        # add a ground plane
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(
            cube_handle
        )
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        obj_template_mgr.register_template(cube_template_cpy)
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

        # compute a navmesh on the ground plane
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        self.sim.navmesh_visualization = True
        return ground_plane

    def create_robot(self):
        # add the robot to the world via the wrapper
        robot_path = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
        agent_config = DictConfig({"articulated_agent_urdf": robot_path})

        spot = spot_robot.SpotRobotReal(
            agent_config, self.sim, fixed_base=self.fixed_base
        )
        spot.reconfigure()
        spot.update()
        assert spot.get_robot_sim_id() == self.ground_plane.object_id + 1
        return spot

    def visualize_position_sim(self, real_position, real_rotation):
        sim_pos = transform_position(real_position, direction="real_to_sim")
        print("ball sim pos: ", sim_pos)
        self.visualize_position(sim_pos, r=0.1)

    def create_video(self):
        # produce some test debug video
        if self.produce_debug_video:
            print("making video!")
            from habitat_sim.utils import viz_utils as vut

            print("# obs: ", len(self.observations))
            vut.make_video(
                self.observations,
                "color_sensor",
                "color",
                f"sim2real/output/test_{int(time.time())*1000}",
                open_vid=False,
            )
            print("finished making video")

    def init_scene_and_robot(self):
        # set this to output test results as video for easy investigation
        cfg_settings = default_sim_settings.copy()
        cfg_settings["scene"] = "NONE"
        cfg_settings["enable_physics"] = True

        # loading the physical scene
        hab_cfg = self.make_cfg(cfg_settings)
        self.origin = mn.Vector3(0.0, 0.0, 0.0)
        if is_pb_installed:
            arm_urdf = "/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/data/versioned_data/hab_spot_arm/urdf/spot_onlyarm.urdf"
            self.ik_helper = IkHelper(
                arm_urdf,
                np.deg2rad(self.start_js),
            )

        self.sim = habitat_sim.Simulator(hab_cfg)
        # setup the camera for debug video (looking at 0,0,0)
        self.sim.agents[0].scene_node.translation = [2.0, -1.0, 0.0]
        self.ground_plane = self.setup_scene()
        self.spot = self.create_robot()
        self.draw_axes()

        real_pos = [1.0, 1.0, 0.0]
        real_rot = [0.0, 0.0, 0.0]
        self.visualize_position_sim(real_pos, real_rot)
        self.simulate(1.0)

        self.spot.set_base_position(0.0, 0.0, 0.0)
        self.spot.set_arm_joint_positions(self.start_js)
        self.spot.close_gripper()
        self.spot.leg_joint_pos = [0.0, 0.7, -1.5] * 4
        self.simulate(1.0)

    def test_position(self):
        self.reset_spot()
        ## position test
        real_robot_positions = [
            [0.0, 0.0, 0.0],  # origin
            [0.0, 0.0, 1.0],  # up
            [1.0, 0.0, 0.0],  # forward
            [-1.0, 0.0, 0.0],  # back
            [0.0, 1.0, 0.0],  # left
            [0.0, -1.0, 0.0],  # right
            [1.0, 1.0, 0.0],  # front left
            [1.0, -1.0, 0.0],  # front right
            [-1.0, 1.0, 0.0],  # back left
            [-1.0, -1.0, 0.0],  # back right
        ]
        for real_robot_pos in real_robot_positions:
            self.spot.set_base_position(
                real_robot_pos[0], real_robot_pos[1], 0
            )
            # set base ground position from navmesh
            self.simulate(1.0)

            print(
                "obj translation: ",
                self.spot.sim_obj.transformation.translation,
            )

    def test_rotation(self):
        self.reset_spot()

        ## rotation test
        real_robot_rotations = [
            [0.0, 0.0, 0.0],  # origin
            [0.0, 0.0, 45.0],  # yaw 45 deg right
            [0.0, 0.0, -45.0],  # yaw 45 deg left
            [45.0, 0.0, 0.0],  # roll 45 deg right
            [-45.0, 0.0, 0.0],  # roll 45 deg left
            [0.0, 45.0, 0.0],  # pitch 45 deg down
            [0.0, -45.0, 0.0],  # pitch 45 deg up
        ]
        for real_robot_rot in real_robot_rotations:
            real_robot_pos = [0.0, 0.0, 0.0]
            self.set_robot_pose(
                self.spot, real_robot_pos, np.deg2rad(real_robot_rot)
            )

            # set base ground position from navmesh
            self.simulate(1.0)
            print(
                "obj translation: ",
                self.spot.sim_obj.transformation.translation,
            )
            sim_obj_rot = matrix_to_euler(
                self.spot.sim_obj.transformation.rotation()
            )
            print("obj rotation: ", np.rad2deg(sim_obj_rot))
            print(
                "base translation: ", self.spot.base_transformation.translation
            )
            base_rot = matrix_to_euler(
                self.spot.base_transformation.rotation()
            )
            print("base rotation: ", np.rad2deg(base_rot))

    def test_gripper(self, sim, spot, produce_debug_video):
        self.simulate(1.0)
        spot.open_gripper()
        self.simulate(1.0)
        spot.close_gripper()
        self.simulate(1.0)

    def test_arm_joints(self, sim, spot, produce_debug_video):
        self.reset_spot()

        arm_joint_positions = [
            # [0.0, -120.0, 0.0, 60.0, 0.0, 88.0, 0.0],
            # [0.0, -120.0, 0.0, 60.0, 0.0, 88.0, 90.0],  # roll right
            # [0.0, -120.0, 0.0, 60.0, 0.0, 88.0, -90.0],  # roll left
            # [0.0, -120.0, 0.0, 60.0, 0.0, 120.0, 0.0],  # pitch down
            # [0.0, -120.0, 0.0, 60.0, 0.0, 30.0, 0.0],  # pitch up
            [0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0],  # zero
            # [0.0, -180, 0.0, 180.0, 0.0, 0.0, 90.0],  # roll right
            # [0.0, -180, 0.0, 180.0, 0.0, 30.0, 0.0],  # pitch up
            [90.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0],  # shoulder yaw right -
            [60.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0],  # shoulder yaw right -
            [90.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0],  # shoulder yaw right -
            [120.0, -180, 0.0, 171.0, 0.0, 0.0, 0.0],  # shoulder yaw right -
            [0.0, -120.0, 0.0, 60.0, 0.0, 88.0, 0.0],  # zero
            [-30.0, -180, 0.0, 171.0, 0.0, 0.0, 0.0],  # shoulder yaw right +
            [-60.0, -180, 0.0, 171.0, 0.0, 0.0, 0.0],  # shoulder yaw right +
            [-90.0, -180, 0.0, 171.0, 0.0, 0.0, 0.0],  # shoulder yaw right +
            [-120.0, -180, 0.0, 171.0, 0.0, 0.0, 0.0],  # shoulder yaw right +
            # [
            #     -120.0,
            #     -120.0,
            #     0.0,
            #     60.0,
            #     0.0,
            #     88.0,
            #     0.0,
            # ],  # shoulder yaw left +
            # [
            #     -150.0,
            #     -120.0,
            #     0.0,
            #     60.0,
            #     0.0,
            #     88.0,
            #     0.0,
            # ],  # shoulder yaw left +
            # [
            #     -180.0,
            #     -120.0,
            #     0.0,
            #     60.0,
            #     0.0,
            #     88.0,
            #     0.0,
            # ],  # shoulder yaw left +
            # [
            #     -210.0,
            #     -120.0,
            #     0.0,
            #     60.0,
            #     0.0,
            #     88.0,
            #     0.0,
            # ],  # shoulder yaw right +
            # [0.0, -120.0, 0.0, 60.0, 0.0, 88.0, 0.0],  # shoulder yaw left +
            # [0.0, 60.0, 0.0, 60.0, 0.0, 88.0, 0.0],  # shoulder pitch down
            # [0.0, -60.0, 0.0, 60.0, 0.0, 88.0, 0.0],  # shoulder pitch up
            # [0.0, -120.0, 0.0, 60.0, 0.0, 88.0, 0.0],  # shoulder pitch up
            # [0.0, -120.0, 60.0, 60.0, 0.0, 88.0, 0.0],  # shoulder pitch up
            # [0.0, -120.0, -60.0, 60.0, 0.0, 88.0, 0.0],  # shoulder pitch up
        ]
        for arm_joint_pos in arm_joint_positions:
            self.spot.set_arm_joint_positions(arm_joint_pos)
            self.simulate(1.0)
            obj_T_ee = self.test_obj_to_goal()
            local_ee_pos, local_ee_rpy = self.spot.get_ee_pos_in_body_frame()
            print(
                f"local_ee_pos: {local_ee_pos}, local_ee_rpy: {np.rad2deg(local_ee_rpy)}"
            )
            print("--------------")

        return observations

    def reset_spot(self):
        self.spot.set_base_position(0.0, 0.0, 0.0)

        self.spot.set_arm_joint_positions(
            [0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0]
        )
        self.spot.leg_joint_pos = [0.0, 0.7, -1.5] * 4

        self.simulate(1.0)

    def test_obj_to_goal(self):
        rom = self.sim.get_rigid_object_manager()
        sim_global_T_obj_pos = rom.get_object_by_handle(
            "ball_new_viz_0_:0000"
        ).translation
        global_T_obj_pos = transform_position(
            sim_global_T_obj_pos, direction="sim_to_real"
        )
        global_T_ee = self.spot.get_ee_global_pose()

        global_T_base = self.spot.global_T_body()
        global_T_ee_base_rot = mn.Matrix4.from_(
            global_T_base.rotation(), global_T_ee.translation
        )
        global_T_ee = global_T_ee_base_rot
        obj_T_ee_pos = batch_transform_point(
            [global_T_obj_pos], global_T_ee.inverted(), np.float32
        )[[0]].reshape(-1)
        print("obj_to_goal: ", obj_T_ee_pos)
        return obj_T_ee_pos

    def load_data(self, filepath):
        print("load data: ", filepath)
        with open(os.path.join(filepath), "rb") as handle:
            log_packet_list = pkl.load(handle)
        return log_packet_list

    def apply_ee_constraints(self, ee_target):
        ee_index = 0
        ee_target = np.clip(
            ee_target,
            self.spot.params.ee_constraint[ee_index, :, 0],
            self.spot.params.ee_constraint[ee_index, :, 1],
        )
        return ee_target

    def test_real_replay(self, debug=False):
        # dict_keys(['timestamp', 'datetime', 'camera_data',
        # 'vision_T_base', 'base_pose_xyt', 'arm_pose',
        # 'is_gripper_holding_item', 'gripper_open_percentage', 'gripper_force_in_hand'])
        real_data = self.load_data(self.real_filepath)
        print("len data: ", len(real_data))
        ctr = 0
        abs_error_bullet_xyz_ee_xyz = []
        abs_error_bullet_rpy_ee_rpy = []
        abs_error_hab_xyz_ee_xyz = []
        abs_error_hab_rpy_ee_rpy = []
        abs_error_joint_angles = []
        for step_data in real_data:
            x, y, t = step_data["base_pose_xyt"]
            print("theta: ", t)
            # self.spot.set_base_position(x, y, t)
            # if "ee_pose" in step_data.keys():
            #     joint_pos = np.array(self.spot.arm_joint_pos)
            #     joint_vel = np.zeros(joint_pos.shape)
            #     self.ik_helper.set_arm_state(joint_pos, joint_vel)

            #     cur_sim_ee_xyz, curr_sim_ee_rpy = self.ik_helper.calc_fk(
            #         np.array(self.spot.get_arm_joint_positions())
            #     )
            #     local_ee_pos, local_ee_rpy = (
            #         self.spot.get_ee_pos_in_body_frame()
            #     )
            #     ee_xyz, ee_rpy = step_data["ee_pose"]

            #     # ee_xyz, ee_rpy = self.apply_ee_constraints(
            #     # np.array([ee_xyz, ee_rpy])
            #     # )
            #     # print("CALC IK WITH POSITION")
            #     arm_joints = self.ik_helper.calc_ik(ee_xyz)
            #     print("CALC IK WITH POSITION + ROTATION")
            #     # arm_joints = self.ik_helper.calc_ik(ee_xyz, ee_rpy)

            #     sh0, sh1, el0, el1, wr0, wr1 = step_data["arm_pose"]
            #     # real_arm_joints = np.array([sh0, sh1, el0, el1, wr0, wr1])
            #     real_arm_joints = np.array([sh0, sh1, 0.0, el0, el1, wr0, wr1])
            #     abs_error_joint_angles.append(
            #         np.abs(np.array(real_arm_joints) - np.array(arm_joints))
            #     )
            #     self.spot.set_arm_joint_positions(arm_joints, "radians")
            #     self.ik_helper.set_arm_state(arm_joints)
            # else:
            sh0, sh1, el0, el1, wr0, wr1 = step_data["arm_pose"]
            arm_joints = np.array([sh0, sh1, 0.0, el0, el1, wr0, wr1])
            print("arm_joints: ", arm_joints)
            # arm_joints = np.array([sh0, sh1, el0, el1, wr0, wr1])
            self.spot.set_arm_joint_positions(arm_joints, "radians")
            self.simulate(1.0)

    def test_spot_robot_wrapper(self):
        self.test_position()
        # self.test_rotation()
        # self.test_gripper()
        # self.test_arm_joints()
        # self.test_obj_to_goal()
        # self.test_real_replay()

        self.create_video()

        self.sim.close(destroy=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-video", dest="make_video", action="store_false")
    parser.add_argument("--fix-base", action="store_false")
    parser.add_argument("--filepath", default="")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(dir_path, "output/")

    show_video = args.display
    display = args.display
    make_video = args.make_video
    fixed_base = args.fix_base
    filepath = args.filepath

    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    TE = TestEnv(fixed_base, make_video, filepath)
    TE.test_spot_robot_wrapper()
    # test_spot_robot_wrapper(False, make_video)
