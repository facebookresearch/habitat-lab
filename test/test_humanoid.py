#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from os import path as osp

import gym
import magnum as mn
import numpy as np
import pytest
from omegaconf import DictConfig

import habitat.articulated_agents.humanoids.kinematic_humanoid as kinematic_humanoid
import habitat_sim
import habitat_sim.agent
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)

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
        camera_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
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
        sim.step_physics(0.1 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations


@pytest.mark.parametrize(
    "humanoid_name",
    ["female_2"],
)
def test_humanoid_controller(humanoid_name):
    """Test the humanoid controller"""

    # loading the physical scene
    produce_debug_video = False
    num_steps = 100
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True
    epsilon = 1e-4

    observations = []
    hab_cfg = make_cfg(cfg_settings)
    with habitat_sim.Simulator(hab_cfg) as sim:
        obj_template_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()

        # setup the camera for debug video (looking at 0,0,0)
        sim.agents[0].scene_node.translation = [0.0, -1.0, 2.0]

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
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        sim.navmesh_visualization = True

        # add the humanoid to the world via the wrapper
        humanoid_path = f"data/humanoids/humanoid_data/{humanoid_name}/{humanoid_name}.urdf"
        walk_pose_path = f"data/humanoids/humanoid_data/{humanoid_name}/{humanoid_name}_motion_data_smplx.pkl"

        agent_config = DictConfig(
            {
                "articulated_agent_urdf": humanoid_path,
                "motion_data_path": walk_pose_path,
                "auto_update_sensor_transform": True,
            }
        )
        if not osp.exists(humanoid_path):
            pytest.skip(f"No humanoid file {humanoid_path}")
        kin_humanoid = kinematic_humanoid.KinematicHumanoid(agent_config, sim)
        kin_humanoid.reconfigure()
        kin_humanoid.update()
        assert (
            kin_humanoid.get_robot_sim_id() == 2
        )  # 0 is the stage and 1 is the ground plane
        print(kin_humanoid.get_link_and_joint_names())
        observations += simulate(sim, 1.0, produce_debug_video)

        # set base ground position from navmesh
        # NOTE: because the navmesh floats above the collision geometry we should see a pop/settle with dynamics and no fixed base
        target_base_pos = sim.pathfinder.snap_point(
            kin_humanoid.sim_obj.translation
        )
        kin_humanoid.base_pos = target_base_pos
        assert kin_humanoid.base_pos == target_base_pos
        observations += simulate(sim, 1.0, produce_debug_video)

        # Test controller
        humanoid_controller = HumanoidRearrangeController(walk_pose_path)

        init_pos = kin_humanoid.base_pos
        base_trans = kin_humanoid.base_transformation
        target_pos = init_pos + mn.Vector3(1.5, 0, 0)
        step_count = 0
        humanoid_controller.reset(base_trans)
        while step_count < num_steps:
            pose_diff = target_pos - kin_humanoid.base_pos
            humanoid_controller.calculate_walk_pose(pose_diff)
            new_pose = humanoid_controller.get_pose()

            new_joints = new_pose[:-16]
            new_pos_transform_base = new_pose[-16:]
            new_pos_transform_offset = new_pose[-32:-16]

            # When the array is all 0, this indicates we are not setting
            # the human joint
            if np.array(new_pos_transform_offset).sum() != 0:
                vecs_base = [
                    mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                vecs_offset = [
                    mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                new_transform_offset = mn.Matrix4(*vecs_offset)
                new_transform_base = mn.Matrix4(*vecs_base)
                kin_humanoid.set_joint_transform(
                    new_joints, new_transform_offset, new_transform_base
                )
                observations += simulate(sim, 0.01, produce_debug_video)
            step_count += 1

        assert pose_diff.length() < epsilon
        # produce some test debug video
        if produce_debug_video:
            from habitat_sim.utils import viz_utils as vut

            vut.make_video(
                observations,
                "color_sensor",
                "color",
                "test_humanoid_wrapper",
                open_vid=True,
            )


@pytest.mark.skipif(
    not osp.exists("data/humanoids/humanoid_data/female_2/female_2.urdf"),
    reason="Test requires female 2 humanoid URDF and assets.",
)
@pytest.mark.skipif(
    not habitat_sim.built_with_bullet,
    reason="Robot wrapper API requires Bullet physics.",
)
def test_base_rot():
    # set this to output test results as video for easy investigation
    produce_debug_video = False
    observations = []
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    # loading the physical scene
    hab_cfg = make_cfg(cfg_settings)

    with habitat_sim.Simulator(hab_cfg) as sim:
        # setup the camera for debug video (looking at 0,0,0)
        sim.agents[0].scene_node.translation = [0.0, -1.0, 2.0]

        # add the humanoid to the world via the wrapper
        humanoid_path = "data/humanoids/humanoid_data/female_2/female_2.urdf"
        walk_pose_path = "data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl"

        agent_config = DictConfig(
            {
                "articulated_agent_urdf": humanoid_path,
                "motion_data_path": walk_pose_path,
                "auto_update_sensor_transform": True,
            }
        )
        if not osp.exists(humanoid_path):
            pytest.skip(f"No humanoid file {humanoid_path}")
        kin_humanoid = kinematic_humanoid.KinematicHumanoid(agent_config, sim)
        kin_humanoid.reconfigure()
        kin_humanoid.update()

        # check that for range (-pi, pi) getter and setter are consistent
        num_samples = 100
        for i in range(1, num_samples):
            angle = -math.pi + (math.pi * 2 * i) / num_samples
            kin_humanoid.base_rot = angle
            assert np.allclose(angle, kin_humanoid.base_rot, atol=1e-5)
            if produce_debug_video:
                observations.append(sim.get_sensor_observations())

        # check that for range [-2pi, 2pi] increment is accurate
        kin_humanoid.base_rot = -math.pi * 2
        inc = (math.pi * 4) / num_samples
        rot_check = -math.pi * 2
        for _ in range(0, num_samples):
            kin_humanoid.base_rot = kin_humanoid.base_rot + inc
            rot_check += inc
            # NOTE: here we check that the angle is accurate (allow an offset of one full rotation to cover redundant options)
            accurate_angle = False
            for offset in [0, math.pi * 2, math.pi * -2]:
                if np.allclose(
                    rot_check, kin_humanoid.base_rot + offset, atol=1e-5
                ):
                    accurate_angle = True
                    break
            assert (
                accurate_angle
            ), f"should be {rot_check}, but was {kin_humanoid.base_rot}."
            if produce_debug_video:
                observations.append(sim.get_sensor_observations())

        # produce some test debug video
        if produce_debug_video:
            from habitat_sim.utils import viz_utils as vut

            vut.make_video(
                observations,
                "color_sensor",
                "color",
                "test_base_rot",
                open_vid=True,
            )


@pytest.mark.skipif(
    not osp.exists("data/humanoids/humanoid_data/female_2/female_2.urdf"),
    reason="Test requires a human armature.",
)
@pytest.mark.skipif(
    not osp.exists(
        "data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl"
    ),
    reason="Test requires motion files.",
)
def test_gym_humanoid():
    """Test Gym with the humanoid"""
    config_file = "benchmark/rearrange/skills/pick.yaml"
    overrides = [
        "~habitat.task.actions.arm_action",
        "~habitat.task.actions.base_velocity",
        "++habitat.task.actions={humanoidjoint_action:{type:HumanoidJointAction}}",
        "++habitat.task.actions.humanoidjoint_action.num_joints=17",
        "habitat.simulator.agents.main_agent.articulated_agent_urdf=data/humanoids/humanoid_data/female_2/female_2.urdf",
        "habitat.simulator.agents.main_agent.articulated_agent_type=KinematicHumanoid",
        "habitat.simulator.agents.main_agent.motion_data_path=data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl",
        "habitat.simulator.ac_freq_ratio=1",
        "habitat.task.measurements.force_terminate.max_accum_force=-1.0",
        "habitat.task.measurements.force_terminate.max_instant_force=-1.0",
        "habitat.simulator.kinematic_mode=True",
        "habitat.simulator.ac_freq_ratio=1",
    ]

    hab_gym = gym.make(
        "Habitat-v0",
        cfg_file_path=config_file,
        override_options=overrides,
        use_render_mode=True,
    )
    hab_gym.reset()
    hab_gym.step(hab_gym.action_space.sample())
    hab_gym.close()

    hab_gym = gym.make(
        "HabitatRender-v0",
        cfg_file_path=config_file,
    )
    hab_gym.reset()
    hab_gym.step(hab_gym.action_space.sample())
    hab_gym.render("rgb_array")
    hab_gym.close()


@pytest.mark.parametrize(
    "humanoid_name",
    ["female_2"],
)
def test_humanoid_seqpose_controller(humanoid_name):
    """Test the humanoid controller"""

    # loading the physical scene
    produce_debug_video = False
    num_steps = 1000
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    observations = []
    hab_cfg = make_cfg(cfg_settings)
    with habitat_sim.Simulator(hab_cfg) as sim:
        obj_template_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()

        # setup the camera for debug video (looking at 0,0,0)
        sim.agents[0].scene_node.translation = [0.0, 0.0, 2.0]

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
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        sim.navmesh_visualization = True

        # add the humanoid to the world via the wrapper
        humanoid_path = f"data/humanoids/humanoid_data/{humanoid_name}/{humanoid_name}.urdf"
        walk_pose_path = f"data/humanoids/humanoid_data/{humanoid_name}/{humanoid_name}_motion_data_smplx.pkl"

        agent_config = DictConfig(
            {
                "articulated_agent_urdf": humanoid_path,
                "motion_data_path": walk_pose_path,
                "auto_update_sensor_transform": True,
            }
        )
        if not osp.exists(humanoid_path):
            pytest.skip(f"No humanoid file {humanoid_path}")
        kin_humanoid = kinematic_humanoid.KinematicHumanoid(agent_config, sim)
        kin_humanoid.reconfigure()
        kin_humanoid.update()
        assert (
            kin_humanoid.get_robot_sim_id() == 2
        )  # 0 is the stage and 1 is the ground plane

        # set base ground position from navmesh
        # NOTE: because the navmesh floats above the collision geometry we should see a pop/settle with dynamics and no fixed base
        target_base_pos = sim.pathfinder.snap_point(
            kin_humanoid.sim_obj.translation
        )
        kin_humanoid.base_pos = target_base_pos
        assert kin_humanoid.base_pos == target_base_pos
        observations += simulate(sim, 0.01, produce_debug_video)

        # Test controller
        motion_path = (
            "data/humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.pkl"
        )

        if not osp.exists(motion_path):
            pytest.skip(
                f"No motion file {motion_path}. You can create this file by using humanoid_utils.py"
            )
        humanoid_controller = HumanoidSeqPoseController(motion_path)

        base_trans = kin_humanoid.base_transformation
        step_count = 0
        humanoid_controller.reset(base_trans)
        while step_count < num_steps:
            humanoid_controller.calculate_pose()
            humanoid_controller.next_pose(cycle=True)
            new_pose = humanoid_controller.get_pose()

            new_joints = new_pose[:-16]
            new_pos_transform_base = new_pose[-16:]
            new_pos_transform_offset = new_pose[-32:-16]

            # When the array is all 0, this indicates we are not setting
            # the human joint
            if np.array(new_pos_transform_offset).sum() != 0:
                vecs_base = [
                    mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                vecs_offset = [
                    mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                new_transform_offset = mn.Matrix4(*vecs_offset)
                new_transform_base = mn.Matrix4(*vecs_base)
                kin_humanoid.set_joint_transform(
                    new_joints, new_transform_offset, new_transform_base
                )
                observations += simulate(sim, 0.0001, produce_debug_video)

            step_count += 1

        # produce some test debug video
        if produce_debug_video:
            from habitat_sim.utils import viz_utils as vut

            vut.make_video(
                observations,
                "color_sensor",
                "color",
                "test_humanoid_wrapper",
                open_vid=True,
            )
