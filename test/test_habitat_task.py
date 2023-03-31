#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import pytest

import habitat
from habitat.config.default_structured_configs import TeleportActionConfig
from habitat.utils.test_utils import sample_non_stop_action

CFG_TEST = "test/config/habitat/habitat_all_sensors_test.yaml"
teleport_position = np.array(
    [-3.2890449, 0.15067159, 11.124366], dtype=np.float32
)
teleport_ROTATION = np.array([0.92035, 0, -0.39109465, 0], dtype=np.float32)


def test_task_actions():
    config = habitat.get_config(config_path=CFG_TEST)
    with habitat.config.read_write(config):
        config.habitat.task.actions["teleport"] = TeleportActionConfig()

    with habitat.Env(config=config) as env:
        env.reset()
        action = {
            "action": "teleport",
            "action_args": {
                "position": teleport_position,
                "rotation": teleport_ROTATION,
            },
        }
        assert env.action_space.contains(action)
        env.step(action)
        agent_state = env.sim.get_agent_state()
        assert np.allclose(
            np.array(teleport_position, dtype=np.float32), agent_state.position
        ), "mismatch in position after teleport"
        assert np.allclose(
            np.array(teleport_ROTATION, dtype=np.float32),
            np.array([*agent_state.rotation.imag, agent_state.rotation.real]),
        ), "mismatch in rotation after teleport"
        env.step("turn_right")


def test_task_actions_sampling_for_teleport():
    config = habitat.get_config(config_path=CFG_TEST)
    with habitat.config.read_write(config):
        config.habitat.task.actions["teleport"] = TeleportActionConfig()

    with habitat.Env(config=config) as env:
        env.reset()
        while not env.episode_over:
            action = sample_non_stop_action(env.action_space)
            assert env.action_space.contains(action)
            habitat.logger.info(
                f"Action : "
                f"{action['action']}, "
                f"args: {action['action_args']}."
            )
            env.step(action)
            agent_state = env.sim.get_agent_state()
            habitat.logger.info(agent_state)


@pytest.mark.parametrize(
    "config_file",
    [
        CFG_TEST,
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        "test/config/habitat/habitat_mp3d_eqa_test.yaml",
    ],
)
def test_task_actions_sampling(config_file):
    config = habitat.get_config(config_path=config_file)
    if not os.path.exists(
        config.habitat.dataset.data_path.format(
            split=config.habitat.dataset.split
        )
    ):
        pytest.skip(
            f"Please download dataset to data folder "
            f"{config.habitat.dataset.data_path}."
        )

    with habitat.Env(config=config) as env:
        env.reset()
        while not env.episode_over:
            action = sample_non_stop_action(env.action_space)
            assert env.action_space.contains(action)
            habitat.logger.info(
                f"Action : "
                f"{action['action']}, "
                f"args: {action['action_args']}."
            )
            env.step(action)
            agent_state = env.sim.get_agent_state()
            habitat.logger.info(agent_state)
