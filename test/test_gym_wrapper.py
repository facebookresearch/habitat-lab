#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob

import gym
import gym.spaces as spaces
import numpy as np
import pytest

import habitat_baselines.utils.env_utils
import habitat_baselines.utils.gym_definitions
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config as baselines_get_config
from habitat_baselines.utils.gym_adapter import HabGymWrapper
from habitat_baselines.utils.render_wrapper import HabRenderWrapper


@pytest.mark.parametrize(
    "config_file,overrides,expected_action_dim,expected_obs_type",
    [
        (
            "habitat_baselines/config/rearrange/ddppo_reach_state.yaml",
            [],
            7,
            np.ndarray,
        ),
        (
            "habitat_baselines/config/rearrange/ddppo_pick.yaml",
            [],
            8,
            dict,
        ),
        (
            "habitat_baselines/config/rearrange/ddppo_pick.yaml",
            [
                "TASK_CONFIG.TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER",
                "SuctionGraspAction",
            ],
            7,
            dict,
        ),
    ],
)
def test_gym_wrapper_contract(
    config_file, overrides, expected_action_dim, expected_obs_type
):
    """
    Test the Gym wrapper returns the right things and works with overrides.
    """
    config = baselines_get_config(config_file, overrides)
    env_class = get_env_class(config.ENV_NAME)

    env = habitat_baselines.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )
    env = HabGymWrapper(env)
    env = HabRenderWrapper(env)
    assert isinstance(env.action_space, spaces.Box)
    assert (
        env.action_space.shape[0] == expected_action_dim
    ), f"Has {env.action_space.shape[0]} action dim but expected {expected_action_dim}"
    obs = env.reset()
    assert isinstance(obs, expected_obs_type), f"Obs {obs}"
    obs, _, _, info = env.step(env.action_space.sample())
    assert isinstance(obs, expected_obs_type), f"Obs {obs}"

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3 and frame.shape[-1] == 3

    for _, v in info.items():
        assert not isinstance(v, dict)
    env.close()


@pytest.mark.parametrize(
    "config_file,override_options",
    [
        [
            "habitat_baselines/config/rearrange/ddppo_pick.yaml",
            [
                "TASK_CONFIG.TASK.ACTIONS.ARM_ACTION.GRIP_CONTROLLER",
                "SuctionGraspAction",
            ],
        ],
        ["habitat_baselines/config/rearrange/ddppo_pick.yaml", []],
    ],
)
def test_full_gym_wrapper(config_file, override_options):
    """
    Test the Gym wrapper and its Render wrapper work
    """
    hab_gym = gym.make(
        "Habitat-v0",
        cfg_file_path=config_file,
        override_options=override_options,
        use_render_mode=True,
    )
    hab_gym.reset()
    hab_gym.step(hab_gym.action_space.sample())
    hab_gym.close()

    hab_gym = gym.make(
        "HabitatRender-v0",
        cfg_file_path=config_file,
    )
    hab_gym.reset()
    hab_gym.step(hab_gym.action_space.sample())
    hab_gym.render("rgb_array")
    hab_gym.close()


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("habitat_baselines/config/rearrange/**/*.yaml", recursive=True),
    ),
)
def test_auto_gym_wrapper(test_cfg_path):
    """
    Test all defined automatic Gym wrappers work
    """
    config = baselines_get_config(test_cfg_path)
    if "GYM_AUTO_NAME" not in config:
        return
    full_gym_name = f"Habitat{config['GYM_AUTO_NAME']}-v0"

    hab_gym = gym.make(full_gym_name)
    hab_gym.reset()
    hab_gym.step(hab_gym.action_space.sample())
    hab_gym.close()
