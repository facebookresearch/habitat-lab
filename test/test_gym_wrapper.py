#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from glob import glob

import gym
import gym.spaces as spaces
import mock
import numpy as np
import pytest

import habitat.utils.env_utils
import habitat.utils.gym_definitions
from habitat.core.environments import get_env_class
from habitat.utils.gym_definitions import _get_env_name

# using mock for pygame to avoid having a pygame windows
sys.modules["pygame"] = mock.MagicMock()
importlib.reload(habitat.utils.gym_adapter)


@pytest.mark.parametrize(
    "config_file,overrides,expected_action_dim,expected_obs_type",
    [
        (
            "benchmark/rearrange/reach_state.yaml",
            [],
            7,
            np.ndarray,
        ),
        (
            "benchmark/rearrange/pick.yaml",
            [],
            10,  # arm = 7 + base = 2 + grip = 1
            dict,
        ),
        (
            "benchmark/rearrange/tidy_house.yaml",
            [],
            11,  # 7 joints, 1 grip action, 2 base velocity, 1 stop action
            dict,
        ),
    ],
)
def test_gym_wrapper_contract_continuous(
    config_file, overrides, expected_action_dim, expected_obs_type
):
    """
    Test the Gym wrapper returns the right things and works with overrides.
    """
    config = habitat.get_config(config_file, overrides)
    env_class_name = _get_env_name(config.habitat)
    env_class = get_env_class(env_class_name)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )

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

    for _, v in info.items():  # type: ignore[union-attr]
        assert not isinstance(v, dict)
    env.close()


@pytest.mark.parametrize(
    "config_file,overrides,expected_action_dim,expected_obs_type",
    [
        (
            "benchmark/nav/imagenav/imagenav_test.yaml",
            [],
            4,
            dict,
        ),
        (
            "benchmark/nav/pointnav/pointnav_habitat_test.yaml",
            [],
            4,
            dict,
        ),
    ],
)
def test_gym_wrapper_contract_discrete(
    config_file, overrides, expected_action_dim, expected_obs_type
):
    """
    Test the Gym wrapper returns the right things and works with overrides.
    """
    config = habitat.get_config(config_file, overrides)
    env_class_name = _get_env_name(config.habitat)
    env_class = get_env_class(env_class_name)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )
    assert isinstance(env.action_space, spaces.Discrete)
    assert (
        env.action_space.n == expected_action_dim
    ), f"Has {env.action_space.n} action dim but expected {expected_action_dim}"
    obs = env.reset()
    assert isinstance(obs, expected_obs_type), f"Obs {obs}"
    obs, _, _, info = env.step(env.action_space.sample())
    assert isinstance(obs, expected_obs_type), f"Obs {obs}"

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3 and frame.shape[-1] == 3

    for _, v in info.items():  # type: ignore[union-attr]
        assert not isinstance(v, dict)
    env.close()


@pytest.mark.parametrize(
    "config_file,override_options",
    [
        [
            "benchmark/rearrange/pick.yaml",
            [
                "habitat.task.actions.arm_action.grip_controller=SuctionGraspAction",
            ],
        ],
        ["benchmark/rearrange/pick.yaml", []],
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
        glob("habitat-lab/habitat/config/benchmark/**/*.yaml", recursive=True),
    ),
)
def test_auto_gym_wrapper(test_cfg_path):
    """
    Test all defined automatic Gym wrappers work
    """
    config = habitat.get_config(test_cfg_path)
    if "gym" not in config.habitat or config.habitat.gym.auto_name == "":
        pytest.skip(f"Gym environment name isn't set for {test_cfg_path}.")
    pytest.importorskip("pygame")
    for prefix in ["", "Render"]:
        full_gym_name = f"Habitat{prefix}{config.habitat.gym.auto_name}-v0"

        hab_gym = gym.make(
            full_gym_name,
            # Test sometimes fails with concurrent rendering.
            override_options=["habitat.simulator.concur_render=False"],
        )
        hab_gym.reset()
        done = False
        for _ in range(5):
            hab_gym.render(mode="human")
            _, _, done, _ = hab_gym.step(hab_gym.action_space.sample())
            if done:
                hab_gym.reset()

        hab_gym.close()


@pytest.mark.parametrize(
    "name",
    [
        "HabitatPick-v0",
        "HabitatPlace-v0",
        "HabitatCloseCab-v0",
        "HabitatCloseFridge-v0",
        "HabitatOpenCab-v0",
        "HabitatOpenFridge-v0",
        "HabitatNavToObj-v0",
        "HabitatReachState-v0",
        "HabitatTidyHouse-v0",
        "HabitatPrepareGroceries-v0",
        "HabitatSetTable-v0",
    ],
)
def test_gym_premade_envs(name):
    env = gym.make(name)
    env.reset()
    done = False
    for _ in range(10):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            env.reset()
            done = False
    env.close()
