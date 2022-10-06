#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os

import pytest

import habitat

try:
    from habitat_baselines.agents import ppo_agents, simple_agents

    baseline_installed = True
except ImportError:
    baseline_installed = False

CFG_TEST = "test/habitat_all_sensors_test.yaml"


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

    agent_config = ppo_agents.get_default_config()
    agent_config.MODEL_PATH = ""
    with habitat.config.read_write(agent_config):
        config_env = habitat.get_config(config_paths=CFG_TEST)
        if not os.path.exists(config_env.habitat.simulator.scene):
            pytest.skip("Please download Habitat test data to data folder.")

        benchmark = habitat.Benchmark(config_paths=CFG_TEST)
        with habitat.config.read_write(config_env):
            config_env.habitat.simulator.agent_0.sensors = []
            if input_type in ["rgb", "rgbd"]:
                config_env.habitat.simulator.agent_0.sensors += ["rgb_sensor"]
                agent_config.RESOLUTION = resolution
                config_env.habitat.simulator.rgb_sensor.width = resolution
                config_env.habitat.simulator.rgb_sensor.height = resolution
            if input_type in ["depth", "rgbd"]:
                config_env.habitat.simulator.agent_0.sensors += [
                    "depth_sensor"
                ]
                agent_config.RESOLUTION = resolution
                config_env.habitat.simulator.depth_sensor.width = resolution
                config_env.habitat.simulator.depth_sensor.height = resolution

        del benchmark._env
        benchmark._env = habitat.Env(config=config_env)
        agent_config.INPUT_TYPE = input_type

        agent = ppo_agents.PPOAgent(agent_config)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=10))
        benchmark._env.close()


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_simple_agents():
    config_env = habitat.get_config(config_paths=CFG_TEST)

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
            config_env.habitat.task.success.success_distance,
            config_env.habitat.task.goal_sensor_uuid,
        )
        habitat.logger.info(agent_class.__name__)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=100))

    benchmark._env.close()
