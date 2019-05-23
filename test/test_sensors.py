#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import pytest
import random

import habitat
from habitat.config.default import get_config
from habitat.tasks.nav.nav_task import (
    NavigationEpisode,
    COLLISION_PROXIMITY_TOLERANCE,
)
from habitat.sims.habitat_simulator import SimulatorActions

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"


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
    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config = get_config()
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
    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config = get_config()
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
    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config = get_config()
    config.defrost()
    config.TASK.MEASUREMENTS = ["COLLISIONS"]
    config.freeze()
    env = habitat.Env(config=config, dataset=None)
    env.reset()
    random.seed(123)
    np.random.seed(123)

    actions = [
        SimulatorActions.FORWARD.value,
        SimulatorActions.LEFT.value,
        SimulatorActions.RIGHT.value,
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
