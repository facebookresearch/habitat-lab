#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random

import numpy as np
import pytest

try:
    import torch
    import torch.distributed

    import habitat_sim.utils.datasets_download as data_downloader
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.config.default import get_config

    baseline_installed = True
except ImportError:
    baseline_installed = False


def setup_function(test_trainers):
    # Download the needed datasets
    data_downloader.main(["--uids", "rearrange_task_assets", "--no-replace"])


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "config_path,num_updates,target_reward",
    [
        ("habitat_baselines/config/rearrange/ddppo_reach_state.yaml", 40, 5.0),
    ],
)
def test_trainers(config_path, num_updates, target_reward):
    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)
    # Setup the training
    config = get_config(
        config_path,
        [
            "NUM_UPDATES",
            num_updates,
            "TOTAL_NUM_STEPS",
            -1.0,
            "CHECKPOINT_FOLDER",
            "data/test_checkpoints/test_training",
        ],
    )
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.cuda.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    assert (
        config.TRAINER_NAME == "ddppo"
    ), "This test can only be used with ddppo trainer"

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    # Train
    trainer.train()

    # Gather the data
    deltas = {
        k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
        for k, v in trainer.window_episode_stats.items()
    }
    deltas["count"] = max(deltas["count"], 1.0)
    reward = deltas["reward"] / deltas["count"]

    # Make sure the final reward is greater than the target
    assert (
        reward >= target_reward
    ), f"reward for task {config_path} was {reward} but is expected to be at least {target_reward}"
