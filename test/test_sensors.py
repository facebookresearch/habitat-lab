#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import numpy as np
import pytest

import habitat
import numpy as np
import pytest
from habitat.config.default import get_config
from habitat.sims.habitat_simulator import SimulatorActions
from habitat.tasks.nav.nav_task import (
    COLLISION_PROXIMITY_TOLERANCE,
    NavigationEpisode,
    NavigationGoal,
)

NON_STOP_ACTIONS = [
    v for v in range(len(SimulatorActions)) if v != SimulatorActions.STOP.value
]

MOVEMENT_ACTIONS = [
    SimulatorActions.MOVE_FORWARD.value,
    SimulatorActions.TURN_LEFT.value,
    SimulatorActions.TURN_RIGHT.value,
]


def _random_episode(env, config):
    random_location = env._sim.sample_navigable_point()
    random_heading = np.random.uniform(-np.pi, np.pi)
    random_rotation = [
        0,
        np.sin(random_heading / 2),
        0,
        np.cos(random_heading / 2),
    ]
    env.episodes = [
        NavigationEpisode(
            episode_id="0",
            scene_id=config.SIMULATOR.SCENE,
            start_position=random_location,
            start_rotation=random_rotation,
            goals=[],
        )
    ]


def test_heading_sensor():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["HEADING_SENSOR"]
    config.freeze()
    env = habitat.Env(config=config, dataset=None)
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
        env.episodes = [
            NavigationEpisode(
                episode_id="0",
                scene_id=config.SIMULATOR.SCENE,
                start_position=[03.00611, 0.072447, -2.67867],
                start_rotation=random_rotation,
                goals=[],
            )
        ]

        obs = env.reset()
        heading = obs["heading"]
        assert np.allclose(heading, random_heading)

    env.close()


def test_tactile():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["PROXIMITY_SENSOR"]
    config.TASK.MEASUREMENTS = ["COLLISIONS"]
    config.freeze()
    env = habitat.Env(config=config, dataset=None)
    env.reset()
    random.seed(1234)

    for _ in range(20):
        _random_episode(env, config)
        env.reset()
        assert env.get_metrics()["collisions"] is None

        my_collisions_count = 0
        action = env._sim.index_forward_action
        for _ in range(10):
            obs = env.step(action)
            collisions = env.get_metrics()["collisions"]
            proximity = obs["proximity"]
            if proximity < COLLISION_PROXIMITY_TOLERANCE:
                my_collisions_count += 1

            assert my_collisions_count == collisions

    env.close()


def test_collisions():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.MEASUREMENTS = ["COLLISIONS"]
    config.freeze()
    env = habitat.Env(config=config, dataset=None)
    env.reset()
    random.seed(123)
    np.random.seed(123)

    actions = [
        SimulatorActions.MOVE_FORWARD.value,
        SimulatorActions.TURN_LEFT.value,
        SimulatorActions.TURN_RIGHT.value,
    ]

    for _ in range(20):
        _random_episode(env, config)

        env.reset()
        assert env.get_metrics()["collisions"] is None

        prev_collisions = 0
        prev_loc = env.sim.get_agent_state().position
        for _ in range(50):
            action = np.random.choice(actions)
            env.step(action)
            collisions = env.get_metrics()["collisions"]

            loc = env.sim.get_agent_state().position
            if (
                np.linalg.norm(loc - prev_loc)
                < 0.9 * config.SIMULATOR.FORWARD_STEP_SIZE
                and action == actions[0]
            ):
                # Check to see if the new method of doing collisions catches
                # all the same collisions as the old method
                assert collisions == prev_collisions + 1

            prev_loc = loc
            prev_collisions = collisions

    env.close()


def test_static_pointgoal_sensor():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = ["STATIC_POINTGOAL_SENSOR"]
    config.freeze()
    env = habitat.Env(config=config, dataset=None)

    # start position is checked for validity for the specific test scene
    valid_start_position = [-1.3731, 0.08431, 8.60692]
    expected_static_pointgoal = [0.1, 0.2, 0.3]
    goal_position = np.add(valid_start_position, expected_static_pointgoal)

    # starting quaternion is rotated 180 degree along z-axis, which
    # corresponds to simulator using z-negative as forward action
    start_rotation = [0, 0, 0, 1]

    env.episodes = [
        NavigationEpisode(
            episode_id="0",
            scene_id=config.SIMULATOR.SCENE,
            start_position=valid_start_position,
            start_rotation=start_rotation,
            goals=[NavigationGoal(position=goal_position)],
        )
    ]

    env.reset()
    for _ in range(100):
        obs = env.step(np.random.choice(NON_STOP_ACTIONS))
        static_pointgoal = obs["static_pointgoal"]
        # check to see if taking non-stop actions will affect static point_goal
        assert np.allclose(static_pointgoal, expected_static_pointgoal)

    env.close()


def test_get_observations_at():
    config = get_config()
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config.defrost()
    config.TASK.SENSORS = []
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.freeze()
    env = habitat.Env(config=config, dataset=None)

    # start position is checked for validity for the specific test scene
    valid_start_position = [-1.3731, 0.08431, 8.60692]
    expected_static_pointgoal = [0.1, 0.2, 0.3]
    goal_position = np.add(valid_start_position, expected_static_pointgoal)

    # starting quaternion is rotated 180 degree along z-axis, which
    # corresponds to simulator using z-negative as forward action
    start_rotation = [0, 0, 0, 1]

    env.episodes = [
        NavigationEpisode(
            episode_id="0",
            scene_id=config.SIMULATOR.SCENE,
            start_position=valid_start_position,
            start_rotation=start_rotation,
            goals=[NavigationGoal(position=goal_position)],
        )
    ]

    obs = env.reset()
    start_state = env.sim.get_agent_state()
    for _ in range(100):
        # Note, this test will not currently work for camera change actions
        # (look up/down), only for movement actions.
        new_obs = env.step(np.random.choice(MOVEMENT_ACTIONS))
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
        start_state.position, start_state.rotation, keep_agent_at_new_pose=True
    )
    for key, val in obs_at_point.items():
        assert np.allclose(val, obs[key])
    agent_state = env.sim.get_agent_state()
    assert np.allclose(agent_state.position, start_state.position)
    assert np.allclose(agent_state.rotation, start_state.rotation)
    env.close()
