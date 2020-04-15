#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

import habitat
from habitat.config import Config as CN

try:
    from habitat_baselines.agents import ppo_agents
    from habitat_baselines.agents import simple_agents

    baseline_installed = True
except ImportError:
    baseline_installed = False

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_ppo_agents():

    agent_config = ppo_agents.get_default_config()
    agent_config.MODEL_PATH = ""
    agent_config.defrost()
    config_env = habitat.get_config(config_paths=CFG_TEST)
    if not os.path.exists(config_env.habitat.simulator.scene):
        pytest.skip("Please download Habitat test data to data folder.")

    benchmark = habitat.Benchmark(config_paths=CFG_TEST)

    for input_type in ["blind", "rgb", "depth", "rgbd"]:
        for resolution in [256, 384]:
            config_env.defrost()
            config_env.habitat.simulator.agent_0.sensors = []
            if input_type in ["rgb", "rgbd"]:
                config_env.habitat.simulator.agent_0.sensors += ["rgb_sensor"]
                config_env.habitat.simulator.rgb_sensor.width = resolution
                config_env.habitat.simulator.rgb_sensor.height = resolution
            if input_type in ["depth", "rgbd"]:
                config_env.habitat.simulator.agent_0.sensors += [
                    "depth_sensor"
                ]
                config_env.habitat.simulator.depth_sensor.width = resolution
                config_env.habitat.simulator.depth_sensor.height = resolution
            config_env.freeze()
            del benchmark._env
            benchmark._env = habitat.Env(config=config_env)
            agent_config.INPUT_TYPE = input_type

            agent = ppo_agents.PPOAgent(agent_config)
            habitat.logger.info(benchmark.evaluate(agent, num_episodes=10))


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
            config_env.habitat.task.success_distance,
            config_env.habitat.task.goal_sensor_uuid,
        )
        habitat.logger.info(agent_class.__name__)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=100))
