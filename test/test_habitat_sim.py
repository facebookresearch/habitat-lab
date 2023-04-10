#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import pytest

from habitat.config import read_write
from habitat.config.default import get_agent_config, get_config
from habitat.sims import make_sim


def test_sim_no_sensors():
    config = get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    with read_write(config):
        agent_config = get_agent_config(config.habitat.simulator)
        agent_config.sim_sensors = {}
        if not os.path.exists(config.habitat.simulator.scene):
            pytest.skip("Please download Habitat test data to data folder.")
        with make_sim(
            config.habitat.simulator.type, config=config.habitat.simulator
        ) as sim:
            sim.reset()


def test_sim_geodesic_distance():
    config = get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
    if not os.path.exists(config.habitat.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")
    with make_sim(
        config.habitat.simulator.type, config=config.habitat.simulator
    ) as sim:
        sim.reset()

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "test-sim-geodesic-distance-test-golden.json",
            ),
            "r",
        ) as f:
            test_data = json.load(f)

        for test_case in test_data["single_end"]:
            assert np.isclose(
                sim.geodesic_distance(test_case["start"], test_case["end"]),
                test_case["expected"],
            ), "Geodesic distance mechanism has been changed"

        for test_case in test_data["multi_end"]:
            assert np.isclose(
                sim.geodesic_distance(test_case["start"], test_case["ends"]),
                test_case["expected"],
            ), "Geodesic distance mechanism has been changed"

            assert np.isclose(
                sim.geodesic_distance(test_case["start"], test_case["ends"]),
                np.min(
                    [
                        sim.geodesic_distance(test_case["start"], end)
                        for end in test_case["ends"]
                    ]
                ),
            ), "Geodesic distance for multi target setup isn't equal to separate single target calls."
