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
from habitat.tasks.nav.nav_task import NavigationEpisode

CFG_TEST = "test/habitat_all_sensors_test.yaml"


def test_heading_sensor():
    config = get_config(CFG_TEST)
    if not os.path.exists(config.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")
    config = get_config()
    config.defrost()
    config.TASK.SENSORS = ['HEADING_SENSOR']
    config.freeze()
    env = habitat.Env(config=config, dataset=None)
    env.reset()
    random.seed(1234)

    for _ in range(100):
        random_heading = np.random.uniform(-np.pi, np.pi)
        random_rotation = [0, np.sin(random_heading / 2), 0, np.cos(random_heading / 2)]
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
        heading = obs['heading']
        assert np.allclose(heading, random_heading)

    env.close()
