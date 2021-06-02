#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import numpy as np
import pytest
import quaternion

import habitat
from habitat.config.default import get_config
from habitat.tasks.nav.nav import (
    MoveForwardAction,
    NavigationEpisode,
    NavigationGoal,
)
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_rotate_vector,
)
from habitat.utils.test_utils import sample_non_stop_action
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)


def _random_episode(env, config):
    random_location = env._sim.sample_navigable_point()
    random_heading = np.random.uniform(-np.pi, np.pi)
    random_rotation = [
        0,
        np.sin(random_heading / 2),
        0,
        np.cos(random_heading / 2),
    ]
    env.episode_iterator = iter(
        [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=random_location,
                start_rotation=random_rotation,
                goals=[],
            )
        ]
    )


def test_state_sensors():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["HEADING_SENSOR", "COMPASS_SENSOR", "GPS_SENSOR"]
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:
        env.reset()
        random.seed(123)
        np.random.seed(123)

        for _ in range(100):
            random_heading = np.random.uniform(-np.pi, np.pi)
            random_rotation = [
                0,
                np.sin(random_heading / 2),
                0,
                np.cos(random_heading / 2),
            ]
            env.episode_iterator = iter(
                [
                    NavigationEpisode(
                        episode_id="0",
                        scene_id=config.SIMULATOR.SCENE,
                        start_position=[03.00611, 0.072_447, -2.67867],
                        start_rotation=random_rotation,
                        goals=[],
                    )
                ]
            )

            obs = env.reset()
            heading = obs["heading"]
            assert np.allclose(heading, [random_heading])
            assert np.allclose(obs["compass"], [0.0], atol=1e-5)
            assert np.allclose(obs["gps"], [0.0, 0.0], atol=1e-5)


def test_tactile():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["PROXIMITY_SENSOR"]
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:
        env.reset()
        random.seed(1234)

        for _ in range(20):
            _random_episode(env, config)
            env.reset()

            for _ in range(10):
                obs = env.step(action=MoveForwardAction.name)
                proximity = obs["proximity"]
                assert 0.0 <= proximity
                assert 2.0 >= proximity


def test_collisions():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.MEASUREMENTS = ["COLLISIONS"]
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:
        env.reset()

        for _ in range(20):
            _random_episode(env, config)

            env.reset()
            assert env.get_metrics()["collisions"] is None

            prev_collisions = 0
            prev_loc = env.sim.get_agent_state().position
            for _ in range(50):
                action = sample_non_stop_action(env.action_space)
                env.step(action)
                collisions = env.get_metrics()["collisions"]["count"]
                loc = env.sim.get_agent_state().position
                if (
                    np.linalg.norm(loc - prev_loc)
                    < 0.9 * config.SIMULATOR.FORWARD_STEP_SIZE
                    and action["action"] == MoveForwardAction.name
                ):
                    # Check to see if the new method of doing collisions catches
                    # all the same collisions as the old method
                    assert collisions == prev_collisions + 1

                # We can _never_ collide with standard turn actions
                if action["action"] != MoveForwardAction.name:
                    assert collisions == prev_collisions

                prev_loc = loc
                prev_collisions = collisions


def test_pointgoal_sensor():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["POINTGOAL_SENSOR"]
    config.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:

        # start position is checked for validity for the specific test scene
        valid_start_position = [-1.3731, 0.08431, 8.60692]
        expected_pointgoal = [0.1, 0.2, 0.3]
        goal_position = np.add(valid_start_position, expected_pointgoal)

        # starting quaternion is rotated 180 degree along z-axis, which
        # corresponds to simulator using z-negative as forward action
        start_rotation = [0, 0, 0, 1]

        env.episode_iterator = iter(
            [
                NavigationEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation,
                    goals=[NavigationGoal(position=goal_position)],
                )
            ]
        )

        env.reset()
        for _ in range(100):
            obs = env.step(sample_non_stop_action(env.action_space))
            pointgoal = obs["pointgoal"]
            # check to see if taking non-stop actions will affect static point_goal
            assert np.allclose(pointgoal, expected_pointgoal)


def test_pointgoal_with_gps_compass_sensor():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = [
        "POINTGOAL_WITH_GPS_COMPASS_SENSOR",
        "COMPASS_SENSOR",
        "GPS_SENSOR",
        "POINTGOAL_SENSOR",
    ]
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "CARTESIAN"

    config.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"

    config.TASK.GPS_SENSOR.DIMENSIONALITY = 3

    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:
        # start position is checked for validity for the specific test scene
        valid_start_position = [-1.3731, 0.08431, 8.60692]
        expected_pointgoal = [0.1, 0.2, 0.3]
        goal_position = np.add(valid_start_position, expected_pointgoal)

        # starting quaternion is rotated 180 degree along z-axis, which
        # corresponds to simulator using z-negative as forward action
        start_rotation = [0, 0, 0, 1]

        env.episode_iterator = iter(
            [
                NavigationEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation,
                    goals=[NavigationGoal(position=goal_position)],
                )
            ]
        )

        env.reset()
        for _ in range(100):
            obs = env.step(sample_non_stop_action(env.action_space))
            pointgoal = obs["pointgoal"]
            pointgoal_with_gps_compass = obs["pointgoal_with_gps_compass"]
            compass = float(obs["compass"][0])
            gps = obs["gps"]
            # check to see if taking non-stop actions will affect static point_goal
            assert np.allclose(
                pointgoal_with_gps_compass,
                quaternion_rotate_vector(
                    quaternion.from_rotation_vector(
                        compass * np.array([0, 1, 0])
                    ).inverse(),
                    pointgoal - gps,
                ),
                atol=1e-5,
            )


def test_imagegoal_sensor():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["IMAGEGOAL_SENSOR"]
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:

        # start position is checked for validity for the specific test scene
        valid_start_position = [-1.3731, 0.08431, 8.60692]
        pointgoal = [0.1, 0.2, 0.3]
        goal_position = np.add(valid_start_position, pointgoal)

        pointgoal_2 = [0.3, 0.2, 0.1]
        goal_position_2 = np.add(valid_start_position, pointgoal_2)

        # starting quaternion is rotated 180 degree along z-axis, which
        # corresponds to simulator using z-negative as forward action
        start_rotation = [0, 0, 0, 1]

        env.episode_iterator = iter(
            [
                NavigationEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation,
                    goals=[NavigationGoal(position=goal_position)],
                ),
                NavigationEpisode(
                    episode_id="1",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation,
                    goals=[NavigationGoal(position=goal_position_2)],
                ),
            ]
        )
        obs = env.reset()
        for _ in range(100):
            new_obs = env.step(sample_non_stop_action(env.action_space))
            # check to see if taking non-stop actions will affect static image_goal
            assert np.allclose(obs["imagegoal"], new_obs["imagegoal"])
            assert np.allclose(obs["rgb"].shape, new_obs["imagegoal"].shape)

        previous_episode_obs = obs
        _ = env.reset()
        for _ in range(10):
            new_obs = env.step(sample_non_stop_action(env.action_space))
            # check to see if taking non-stop actions will affect static image_goal
            assert not np.allclose(
                previous_episode_obs["imagegoal"], new_obs["imagegoal"]
            )
            assert np.allclose(
                previous_episode_obs["rgb"].shape, new_obs["imagegoal"].shape
            )


def test_get_observations_at():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = []
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.freeze()
    with habitat.Env(config=config, dataset=None) as env:

        # start position is checked for validity for the specific test scene
        valid_start_position = [-1.3731, 0.08431, 8.60692]
        expected_pointgoal = [0.1, 0.2, 0.3]
        goal_position = np.add(valid_start_position, expected_pointgoal)

        # starting quaternion is rotated 180 degree along z-axis, which
        # corresponds to simulator using z-negative as forward action
        start_rotation = [0, 0, 0, 1]

        env.episode_iterator = iter(
            [
                NavigationEpisode(
                    episode_id="0",
                    scene_id=config.SIMULATOR.SCENE,
                    start_position=valid_start_position,
                    start_rotation=start_rotation,
                    goals=[NavigationGoal(position=goal_position)],
                )
            ]
        )

        obs = env.reset()
        start_state = env.sim.get_agent_state()
        for _ in range(100):
            # Note, this test will not currently work for camera change actions
            # (look up/down), only for movement actions.
            new_obs = env.step(sample_non_stop_action(env.action_space))
            for key, val in new_obs.items():
                agent_state = env.sim.get_agent_state()
                if not (
                    np.allclose(agent_state.position, start_state.position)
                    and np.allclose(agent_state.rotation, start_state.rotation)
                ):
                    assert not np.allclose(val, obs[key])
            obs_at_point = env.sim.get_observations_at(
                start_state.position,
                start_state.rotation,
                keep_agent_at_new_pose=False,
            )
            for key, val in obs_at_point.items():
                assert np.allclose(val, obs[key])

        obs_at_point = env.sim.get_observations_at(
            start_state.position,
            start_state.rotation,
            keep_agent_at_new_pose=True,
        )
        for key, val in obs_at_point.items():
            assert np.allclose(val, obs[key])
        agent_state = env.sim.get_agent_state()
        assert np.allclose(agent_state.position, start_state.position)
        assert np.allclose(agent_state.rotation, start_state.rotation)


def smoke_test_sensor(config, N_STEPS=100):
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    valid_start_position = [-1.3731, 0.08431, 8.60692]

    expected_pointgoal = [0.1, 0.2, 0.3]
    goal_position = np.add(valid_start_position, expected_pointgoal)

    # starting quaternion is rotated 180 degree along z-axis, which
    # corresponds to simulator using z-negative as forward action
    start_rotation = [0, 0, 0, 1]
    test_episode = NavigationEpisode(
        episode_id="0",
        scene_id=config.SIMULATOR.SCENE,
        start_position=valid_start_position,
        start_rotation=start_rotation,
        goals=[NavigationGoal(position=goal_position)],
    )

    with habitat.Env(config=config, dataset=None) as env:

        env.episode_iterator = iter([test_episode])
        no_noise_obs = env.reset()
        assert no_noise_obs is not None

        actions = [
            sample_non_stop_action(env.action_space) for _ in range(N_STEPS)
        ]
        for action in actions:
            assert env.step(action) is not None


@pytest.mark.parametrize(
    "sensors",
    [
        ["FISHEYE_RGB_SENSOR"],
        ["FISHEYE_DEPTH_SENSOR"],
        ["FISHEYE_SEMANTIC_SENSOR"],
        ["EQUIRECT_RGB_SENSOR"],
        ["EQUIRECT_DEPTH_SENSOR"],
        ["EQUIRECT_SEMANTIC_SENSOR"],
    ],
)
@pytest.mark.parametrize("cuda", [True, False])
def test_smoke_not_pinhole_sensors(sensors, cuda):
    habitat_sim = pytest.importorskip("habitat_sim")
    if not habitat_sim.cuda_enabled and cuda:
        pytest.skip("habitat_sim must be built with CUDA to test G2P2GPU")
    config = get_config()
    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = cuda

    config.SIMULATOR.SCENE = (
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    )
    config.SIMULATOR.AGENT_0.SENSORS = sensors
    config.freeze()
    smoke_test_sensor(config)


@pytest.mark.parametrize(
    "sensor", ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
)
@pytest.mark.parametrize("sensor_subtype", ["ORTHOGRAPHIC", "PINHOLE"])
@pytest.mark.parametrize("cuda", [True, False])
def test_smoke_pinhole_sensors(sensor, sensor_subtype, cuda):
    habitat_sim = pytest.importorskip("habitat_sim")
    if not habitat_sim.cuda_enabled and cuda:
        pytest.skip("habitat_sim must be built with CUDA")
    config = get_config()
    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = cuda
    config.SIMULATOR.SCENE = (
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    )
    config.SIMULATOR.AGENT_0.SENSORS = [sensor]
    getattr(config.SIMULATOR, sensor).SENSOR_SUBTYPE = sensor_subtype
    config.freeze()
    smoke_test_sensor(config)


def test_noise_models_rgbd():
    DEMO_MODE = False
    N_STEPS = 100

    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = (
        "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    )
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.freeze()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    valid_start_position = [-1.3731, 0.08431, 8.60692]

    expected_pointgoal = [0.1, 0.2, 0.3]
    goal_position = np.add(valid_start_position, expected_pointgoal)

    # starting quaternion is rotated 180 degree along z-axis, which
    # corresponds to simulator using z-negative as forward action
    start_rotation = [0, 0, 0, 1]
    test_episode = NavigationEpisode(
        episode_id="0",
        scene_id=config.SIMULATOR.SCENE,
        start_position=valid_start_position,
        start_rotation=start_rotation,
        goals=[NavigationGoal(position=goal_position)],
    )

    print(f"{test_episode}")
    with habitat.Env(config=config, dataset=None) as env:

        env.episode_iterator = iter([test_episode])
        no_noise_obs = [env.reset()]
        no_noise_states = [env.sim.get_agent_state()]

        actions = [
            sample_non_stop_action(env.action_space) for _ in range(N_STEPS)
        ]
        for action in actions:
            no_noise_obs.append(env.step(action))
            no_noise_states.append(env.sim.get_agent_state())

    config.defrost()

    config.SIMULATOR.RGB_SENSOR.NOISE_MODEL = "GaussianNoiseModel"
    config.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS = habitat.Config()
    config.SIMULATOR.RGB_SENSOR.NOISE_MODEL_KWARGS.INTENSITY_CONSTANT = 0.5
    config.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL = "RedwoodDepthNoiseModel"

    config.SIMULATOR.ACTION_SPACE_CONFIG = "pyrobotnoisy"
    config.SIMULATOR.NOISE_MODEL = habitat.Config()
    config.SIMULATOR.NOISE_MODEL.ROBOT = "LoCoBot"
    config.SIMULATOR.NOISE_MODEL.CONTROLLER = "Proportional"
    config.SIMULATOR.NOISE_MODEL.NOISE_MULTIPLIER = 0.5

    config.freeze()

    with habitat.Env(config=config, dataset=None) as env:

        env.episode_iterator = iter([test_episode])

        obs = env.reset()
        assert np.linalg.norm(
            obs["rgb"].astype(np.float)
            - no_noise_obs[0]["rgb"].astype(np.float)
        ) > 1.5e-2 * np.linalg.norm(
            no_noise_obs[0]["rgb"].astype(np.float)
        ), "No RGB noise detected."

        assert np.linalg.norm(
            obs["depth"].astype(np.float)
            - no_noise_obs[0]["depth"].astype(np.float)
        ) > 1.5e-2 * np.linalg.norm(
            no_noise_obs[0]["depth"].astype(np.float)
        ), "No Depth noise detected."

        images = []
        state = env.sim.get_agent_state()
        angle_diffs = []
        pos_diffs = []
        for action in actions:
            prev_state = state
            obs = env.step(action)
            state = env.sim.get_agent_state()
            position_change = np.linalg.norm(
                np.array(state.position) - np.array(prev_state.position), ord=2
            )

            if action["action"][:5] == "TURN_":
                angle_diff = abs(
                    angle_between_quaternions(
                        state.rotation, prev_state.rotation
                    )
                    - np.deg2rad(config.SIMULATOR.TURN_ANGLE)
                )
                angle_diffs.append(angle_diff)
            else:
                pos_diffs.append(
                    abs(position_change - config.SIMULATOR.FORWARD_STEP_SIZE)
                )

            if DEMO_MODE:
                images.append(observations_to_image(obs, {}))

        if DEMO_MODE:
            images_to_video(images, "data/video/test_noise", "test_noise")

        assert (
            np.mean(angle_diffs) > 0.025
        ), "No turn action actuation noise detected."
        assert (
            np.mean(pos_diffs) > 0.025
        ), "No forward action actuation noise detected."
