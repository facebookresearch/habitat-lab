#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os

import pytest

import habitat
from habitat.config.default import get_agent_config

try:
    from habitat_baselines.agents import ppo_agents, simple_agents

    baseline_installed = True
except ImportError:
    baseline_installed = False

CFG_TEST = "test/config/habitat/habitat_all_sensors_test.yaml"


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "input_type,resolution",
    [
        (i_type, resolution)
        for i_type, resolution in itertools.product(
            ["blind", "rgb", "depth", "rgbd"], [256, 384]
        )
    ],
)
def test_ppo_agents(input_type, resolution):
    ppo_agent_config = ppo_agents.get_default_config()
    ppo_agent_config.MODEL_PATH = ""
    ppo_agent_config.RESOLUTION = resolution
    ppo_agent_config.INPUT_TYPE = input_type

    with habitat.config.read_write(ppo_agent_config):
        config_env = habitat.get_config(config_path=CFG_TEST)
        if not os.path.exists(config_env.habitat.simulator.scene):
            pytest.skip("Please download Habitat test data to data folder.")

        benchmark = habitat.Benchmark(config_paths=CFG_TEST)
        with habitat.config.read_write(config_env):
            agent_config = get_agent_config(config_env.habitat.simulator)
            agent_config.sim_sensors["rgb_sensor"].update(
                {
                    "height": resolution,
                    "width": resolution,
                }
            )
            agent_config.sim_sensors["depth_sensor"].update(
                {
                    "height": resolution,
                    "width": resolution,
                }
            )
            if input_type in ["depth", "blind"]:
                del agent_config.sim_sensors["rgb_sensor"]
            if input_type in ["rgb", "blind"]:
                del agent_config.sim_sensors["depth_sensor"]

        del benchmark._env
        benchmark._env = habitat.Env(config=config_env)
        agent = ppo_agents.PPOAgent(ppo_agent_config)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=10))
        benchmark._env.close()


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_simple_agents():
    config_env = habitat.get_config(config_path=CFG_TEST)

    if not os.path.exists(config_env.habitat.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")

    benchmark = habitat.Benchmark(config_paths=CFG_TEST)

    for agent_class in [
        simple_agents.ForwardOnlyAgent,
        simple_agents.GoalFollower,
        simple_agents.RandomAgent,
        simple_agents.RandomForwardAgent,
    ]:
        agent = agent_class(
            config_env.habitat.task.measurements.success.success_distance,
            config_env.habitat.task.goal_sensor_uuid,
        )
        habitat.logger.info(agent_class.__name__)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=100))

    benchmark._env.close()
