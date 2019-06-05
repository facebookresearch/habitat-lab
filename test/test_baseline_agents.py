#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

import habitat
from habitat_baselines.agents import simple_agents

try:
    import torch  # noqa # pylint: disable=unused-import

    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from habitat_baselines.agents import ppo_agents

CFG_TEST = "configs/test/habitat_all_sensors_test.yaml"


@pytest.mark.skipif(not has_torch, reason="Test needs torch")
def test_ppo_agents():
    config = ppo_agents.get_defaut_config()
    config.MODEL_PATH = ""
    config_env = habitat.get_config(config_paths=CFG_TEST)
    config_env.defrost()
    if not os.path.exists(config_env.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    benchmark = habitat.Benchmark(config_paths=CFG_TEST)

    for input_type in ["blind", "rgb", "depth", "rgbd"]:
        config_env.defrost()
        config_env.SIMULATOR.AGENT_0.SENSORS = []
        if input_type in ["rgb", "rgbd"]:
            config_env.SIMULATOR.AGENT_0.SENSORS += ["RGB_SENSOR"]
        if input_type in ["depth", "rgbd"]:
            config_env.SIMULATOR.AGENT_0.SENSORS += ["DEPTH_SENSOR"]
        config_env.freeze()
        del benchmark._env
        benchmark._env = habitat.Env(config=config_env)
        config.INPUT_TYPE = input_type

        agent = ppo_agents.PPOAgent(config)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=10))


def test_simple_agents():
    config_env = habitat.get_config(config_paths=CFG_TEST)

    if not os.path.exists(config_env.SIMULATOR.SCENE):
        pytest.skip("Please download Habitat test data to data folder.")

    benchmark = habitat.Benchmark(config_paths=CFG_TEST)

    for agent_class in [
        simple_agents.ForwardOnlyAgent,
        simple_agents.GoalFollower,
        simple_agents.RandomAgent,
        simple_agents.RandomForwardAgent,
    ]:
        agent = agent_class(config_env.TASK.SUCCESS_DISTANCE)
        habitat.logger.info(agent_class.__name__)
        habitat.logger.info(benchmark.evaluate(agent, num_episodes=100))
