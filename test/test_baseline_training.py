#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random

import numpy as np
import pytest

from habitat.config import read_write
from habitat.config.default import get_agent_config

try:
    import torch
    import torch.distributed

    import habitat_sim.utils.datasets_download as data_downloader
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.config.default import get_config

    baseline_installed = True
except ImportError:
    baseline_installed = False

try:
    import pygame  # noqa: F401

    pygame_installed = True
except ImportError:
    pygame_installed = False


def setup_function(test_trainers):
    # Download the needed datasets
    data_downloader.main(["--uids", "rearrange_task_assets", "--no-replace"])


@pytest.mark.skipif(
    int(os.environ.get("TEST_BASELINE_SMALL", 0)) == 0,
    reason="Full training tests did not run. Need `export TEST_BASELINE_SMALL=1",
)
@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "config_path,num_updates,overrides",
    [
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            ["habitat.dataset.split=minival", "benchmark/rearrange=place"],
        ),
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            ["benchmark/rearrange=open_cab"],
        ),
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            [
                "benchmark/rearrange=open_fridge",
            ],
        ),
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            ["habitat.dataset.split=minival", "benchmark/rearrange=pick"],
        ),
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            [
                "habitat.dataset.split=minival",
                "benchmark/rearrange=nav_to_obj",
            ],
        ),
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            [
                "benchmark/rearrange=close_fridge",
            ],
        ),
        (
            "habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml",
            3,
            ["benchmark/rearrange=close_cab"],
        ),
        (
            "habitat-baselines/habitat_baselines/config/imagenav/ddppo_imagenav_example.yaml",
            3,
            [],
        ),
    ],
)
@pytest.mark.parametrize("trainer_name", ["ddppo", "ver"])
def test_trainers(config_path, num_updates, overrides, trainer_name):
    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)
    # Setup the training
    config = get_config(
        config_path,
        [
            f"habitat_baselines.num_updates={num_updates}",
            "habitat_baselines.total_num_steps=-1.0",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            f"habitat_baselines.trainer_name={trainer_name}",
            *overrides,
        ],
    )
    with read_write(config):
        agent_config = get_agent_config(config.habitat.simulator)
        # Changing the visual observation size for speed
        for sim_sensor_config in agent_config.sim_sensors.values():
            sim_sensor_config.update({"height": 64, "width": 64})
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    torch.cuda.manual_seed(config.habitat.seed)
    torch.backends.cudnn.deterministic = True
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    assert config.habitat_baselines.trainer_name in (
        "ddppo",
        "ver",
    ), "This test can only be used with ddppo/ver trainer"

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    # Train
    trainer.train()
    # Training should complete without raising an error.


@pytest.mark.skipif(
    int(os.environ.get("TEST_BASELINE_SMALL", 0)) == 0,
    reason="Full training tests did not run. Need `export TEST_BASELINE_SMALL=1",
)
@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.skipif(
    not pygame_installed, reason="pygame sub-module not installed"
)
@pytest.mark.parametrize(
    "config_path,num_updates",
    [
        (
            "habitat-baselines/habitat_baselines/config/imagenav/ddppo_imagenav_example.yaml",
            3,
        ),
    ],
)
@pytest.mark.parametrize("trainer_name", ["ddppo", "ver"])
@pytest.mark.parametrize("env_key", ["CartPole-v0"])
@pytest.mark.parametrize("dependencies", [[], ["pygame"]])
def test_trainers_gym_registry(
    config_path, num_updates, trainer_name, env_key, dependencies
):
    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)
    # Setup the training
    config = get_config(
        config_path,
        [
            f"habitat_baselines.num_updates={num_updates}",
            "habitat_baselines.total_num_steps=-1.0",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            f"habitat_baselines.trainer_name={trainer_name}",
            # Overwrite the gym_environment
            "habitat.env_task=GymRegistryEnv",
            f"habitat.env_task_gym_dependencies={dependencies}",
            f"habitat.env_task_gym_id={env_key}",
        ],
    )
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    torch.cuda.manual_seed(config.habitat.seed)
    torch.backends.cudnn.deterministic = True
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    assert config.habitat_baselines.trainer_name in (
        "ddppo",
        "ver",
    ), "This test can only be used with ddppo/ver trainer"

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    # Train
    trainer.train()
    # Training should complete without raising an error.


@pytest.mark.skipif(
    int(os.environ.get("TEST_BASELINE_LARGE", 0)) == 0,
    reason="Full training tests did not run. Need `export TEST_BASELINE_LARGE=1",
)
@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "config_path,num_updates,target_reward",
    [
        (
            "habitat-baselines/habitat_baselines/config/rearrange/ddppo_reach_state.yaml",
            40,
            5.0,
        ),
        (
            "habitat-baselines/habitat_baselines/config/pointnav/ddppo_pointnav.yaml",
            1000,
            2.0,
        ),
    ],
)
@pytest.mark.parametrize("trainer_name", ["ddppo", "ver"])
def test_trainers_large(config_path, num_updates, target_reward, trainer_name):
    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)
    # Setup the training
    config = get_config(
        config_path,
        [
            f"habitat_baselines.num_updates={num_updates}",
            "habitat_baselines.total_num_steps=-1.0",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            f"habitat_baselines.trainer_name={trainer_name}",
        ],
    )
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    torch.cuda.manual_seed(config.habitat.seed)
    torch.backends.cudnn.deterministic = True
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    assert config.habitat_baselines.trainer_name in (
        "ddppo",
        "ver",
    ), "This test can only be used with ddppo/ver trainer"

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    # Train
    trainer.train()

    # Gather the data
    if config.habitat_baselines.trainer_name == "ddppo":
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in trainer.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)
        reward = deltas["reward"] / deltas["count"]
    else:
        reward = trainer.window_episode_stats["reward"].mean

    # Make sure the final reward is greater than the target
    assert (
        reward >= target_reward
    ), f"reward for task {config_path} was {reward} but is expected to be at least {target_reward}"
