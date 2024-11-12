#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from os import path as osp

import magnum as mn
import numpy as np
import pytest
from omegaconf import DictConfig

import habitat.articulated_agents.robots.spot_robot as spot_robot
import habitat.articulated_agents.robots.spot_robot_real as spot_robot
import habitat_sim
import habitat_sim.agent
from habitat.tasks.rearrange.utils import (
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


# build SimulatorConfiguration
def make_cfg(settings):
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
        print("sim_cfg.physics_config_file = " + sim_cfg.physics_config_file)
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
        camera_sensor_spec.resolution = [settings["height"], settings["width"]]

        camera_sensor_spec.position = [2.0, 2.5, 0.0]
        camera_sensor_spec.orientation = [
            np.deg2rad(-20.0),
            np.deg2rad(90.0),
            0.0,
        ]

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


def simulate(sim, dt, get_observations=False):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    observations = []
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(1.0 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations


def visualize_position(sim, position, r=0.05):
    template_mgr = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
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


def draw_axes(sim, translation, axis_len=1.0):
    opacity = 1.0
    red = mn.Color4(1.0, 0.0, 0.0, opacity)
    green = mn.Color4(0.0, 1.0, 0.0, opacity)
    blue = mn.Color4(0.0, 0.0, 1.0, opacity)
    lr = sim.get_debug_line_render()
    # draw axes with x+ = red, y+ = green, z+ = blue
    lr.draw_transformed_line(translation, mn.Vector3(axis_len, 0, 0), red)
    lr.draw_transformed_line(translation, mn.Vector3(0, axis_len, 0), green)
    lr.draw_transformed_line(translation, mn.Vector3(0, 0, axis_len), blue)


def set_robot_pose(spot, real_position, real_rotation):
    sim_robot_pos, sim_robot_rot_matrix = transform_3d_coordinates(
        real_position, real_rotation, "real_to_sim", "matrix"
    )

    position = sim_robot_pos - spot.sim_obj.transformation.transform_vector(
        spot.params.base_offset
    )
    mn_matrix = mn.Matrix3(sim_robot_rot_matrix)

    # mn_matrix = mn.Matrix3(spot.get_rotation_matrix(real_robot_rot))
    target_trans = mn.Matrix4.from_(mn_matrix, mn.Vector3(*position))
    spot.sim_obj.transformation = target_trans
    # print(
    #     "real_position: ",
    #     real_position,
    #     "base_position: ",
    #     spot.get_body_position(),
    #     "base_rotation: ",
    #     spot.get_body_rotation(),
    #     # "base_trans: ",
    #     # spot.base_transformation,
    # )


def setup_scene(sim):
    obj_template_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()
    # add a ground plane
    cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
    cube_template_cpy = obj_template_mgr.get_template_by_handle(cube_handle)
    cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
    obj_template_mgr.register_template(cube_template_cpy)
    ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
    ground_plane.translation = [0.0, -0.2, 0.0]
    ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

    # compute a navmesh on the ground plane
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.include_static_objects = True
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    sim.navmesh_visualization = True
    return ground_plane


def create_robot(sim, fixed_base, ground_plane):
    # add the robot to the world via the wrapper
    robot_path = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
    agent_config = DictConfig({"articulated_agent_urdf": robot_path})

    spot = spot_robot.SpotRobotReal(agent_config, sim, fixed_base=fixed_base)
    spot.reconfigure()
    spot.update()
    assert spot.get_robot_sim_id() == ground_plane.object_id + 1
    return spot


def visualize_position_sim(sim, spot, real_position, real_rotation):
    sim_pos = transform_position(real_position, direction="real_to_sim")
    visualize_position(sim, sim_pos, r=0.1)


def create_video(produce_debug_video, observations):
    # produce some test debug video
    if produce_debug_video:
        print("making video!")
        from habitat_sim.utils import viz_utils as vut

        vut.make_video(
            observations,
            "color_sensor",
            "color",
            f"sim2real/test_{int(time.time())*1000}",
            open_vid=False,
        )

def test_position(sim, spot, produce_debug_video):
    observations = []
    observations += reset_spot(sim ,spot, produce_debug_video)
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
        real_robot_rot = [0.0, 0.0, 0.0]
        set_robot_pose(spot, real_robot_pos, np.deg2rad(real_robot_rot))

        # set base ground position from navmesh
        observations += simulate(sim, 1.0, produce_debug_video)

        print('obj translation: ', spot.sim_obj.transformation.translation)
        sim_obj_rot = matrix_to_euler(spot.sim_obj.transformation.rotation())
        print('obj rotation: ', np.rad2deg(sim_obj_rot))
        print('base translation: ', spot.base_transformation.translation)
        base_rot = matrix_to_euler(spot.base_transformation.rotation())
        print('base rotation: ', np.rad2deg(base_rot))
    return observations

def test_rotation(sim, spot, produce_debug_video):
    observations = []
    observations += reset_spot(sim ,spot, produce_debug_video)

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
        set_robot_pose(spot, real_robot_pos, np.deg2rad(real_robot_rot))

        # set base ground position from navmesh
        observations += simulate(sim, 1.0, produce_debug_video)
        print('obj translation: ', spot.sim_obj.transformation.translation)
        sim_obj_rot = matrix_to_euler(spot.sim_obj.transformation.rotation())
        print('obj rotation: ', np.rad2deg(sim_obj_rot))
        print('base translation: ', spot.base_transformation.translation)
        base_rot = matrix_to_euler(spot.base_transformation.rotation())
        print('base rotation: ', np.rad2deg(base_rot))
    return observations

def test_gripper(sim, spot, produce_debug_video):
    observations = []
    observations += reset_spot(sim ,spot, produce_debug_video)
    spot.open_gripper() 
    observations += simulate(sim, 1.0, produce_debug_video)
    spot.close_gripper()
    observations += simulate(sim, 1.0, produce_debug_video)
    return observations

def test_arm_joints(sim ,spot, produce_debug_video):
    observations = []
    observations += reset_spot(sim ,spot, produce_debug_video)

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
        spot.set_arm_joint_positions(arm_joint_pos)
        observations += simulate(sim, 1.0, produce_debug_video)
        local_ee_pos, local_ee_rpy = spot.get_ee_pos_in_body_frame()
        print(f'local_ee_pos: {local_ee_pos}, local_ee_rpy: {np.rad2deg(local_ee_rpy)}')
        print('--------------')

    return observations

def reset_spot(sim ,spot, produce_debug_video):
    observations = []
    set_robot_pose(spot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    spot.set_arm_joint_positions([0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0])
    spot.leg_joint_pos = [0.0, 0.7, -1.5] * 4

    observations += simulate(sim, 1.0, produce_debug_video)
    return observations

def test_spot_robot_wrapper(fixed_base, produce_debug_video=False):
    # set this to output test results as video for easy investigation
    produce_debug_video = True
    observations = []
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    # loading the physical scene
    hab_cfg = make_cfg(cfg_settings)
    origin = mn.Vector3(0.0, 0.0, 0.0)

    with habitat_sim.Simulator(hab_cfg) as sim:
        # setup the camera for debug video (looking at 0,0,0)
        sim.agents[0].scene_node.translation = [2.0, -1.0, 0.0]
        ground_plane = setup_scene(sim)
        spot = create_robot(sim, fixed_base, ground_plane)
        draw_axes(sim, origin)

        real_pos = [0.0, 0.0, 0.0]
        real_rot = [0.0, 0.0, 0.0]
        visualize_position_sim(sim, spot, real_pos, real_rot)

        set_robot_pose(spot, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        spot.set_arm_joint_positions([0.0, -180, 0.0, 180.0, 0.0, 0.0, 0.0])
        observations += simulate(sim, 1.0, produce_debug_video)

        # set the motor angles
        spot.leg_joint_pos = [0.0, 0.7, -1.5] * 4
        observations += test_position(sim, spot, produce_debug_video)
        observations += test_rotation(sim, spot, produce_debug_video)
        observations += test_gripper(sim, spot, produce_debug_video)
        observations += test_arm_joints(sim, spot, produce_debug_video)

        create_video(produce_debug_video, observations)

        sim.close(destroy=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--no-make-video", dest="make_video", action="store_false"
    )
    parser.add_argument("--fix-base", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(dir_path, "output/")

    show_video = args.display
    display = args.display
    make_video = args.make_video

    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    test_spot_robot_wrapper(True, make_video)
    # test_spot_robot_wrapper(False, make_video)
