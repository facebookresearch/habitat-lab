#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
from glob import glob

import pytest

try:
    import torch
    import torch.distributed

    from habitat_baselines.run import run_exp
    from habitat_baselines.common.base_trainer import BaseRLTrainer
    from habitat_baselines.config.default import get_config

    baseline_installed = True
except ImportError:
    baseline_installed = False


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize("task_cfg_path", ["configs/tasks/pointnav.yaml"])
@pytest.mark.parametrize(
    "trainer_cfg_path", glob("habitat_baselines/config/test/*")
)
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("gpu2gpu", [True, False])
def test_trainers(task_cfg_path, trainer_cfg_path, mode, gpu2gpu):
    if gpu2gpu:
        try:
            import habitat_sim
        except ImportError:
            pytest.skip("GPU-GPU requires Habitat-Sim")

        if not habitat_sim.cuda_enabled:
            pytest.skip("GPU-GPU requires CUDA")

    run_exp(
        [task_cfg_path, trainer_cfg_path],
        mode,
        ["habitat.simulator.habitat_sim_v0.gpu_gpu", str(gpu2gpu)],
    )

    # Deinit processes group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_eval_config():
    ckpt_opts = ["habitat_baselines.video_option", "[]"]
    eval_opts = ["habitat_baselines.video_option", "['disk']"]

    ckpt_cfg = get_config(None, ckpt_opts)
    assert ckpt_cfg.habitat_baselines.video_option == []
    assert ckpt_cfg.habitat_baselines.cmd_trailing_opts == [
        "habitat_baselines.video_option",
        "[]",
    ]

    eval_cfg = get_config(None, eval_opts)
    assert eval_cfg.habitat_baselines.video_option == ["disk"]
    assert eval_cfg.habitat_baselines.cmd_trailing_opts == [
        "habitat_baselines.video_option",
        "['disk']",
    ]

    trainer = BaseRLTrainer(get_config())
    assert trainer.config.habitat_baselines.video_option == [
        "disk",
        "tensorboard",
    ]
    returned_config = trainer._setup_eval_config(checkpoint_config=ckpt_cfg)
    assert returned_config.habitat_baselines.video_option == []

    trainer = BaseRLTrainer(eval_cfg)
    returned_config = trainer._setup_eval_config(ckpt_cfg)
    assert returned_config.habitat_baselines.video_option == ["disk"]


def __do_pause_test(num_envs, envs_to_pause):
    class PausableShim:
        def __init__(self, num_envs):
            self._running = list(range(num_envs))

        @property
        def num_envs(self):
            return len(self._running)

        def pause_at(self, idx):
            self._running.pop(idx)

    envs = PausableShim(num_envs)
    test_recurrent_hidden_states = (
        torch.arange(num_envs).view(1, num_envs, 1).expand(4, num_envs, 512)
    )
    not_done_masks = torch.arange(num_envs).view(num_envs, 1)
    current_episode_reward = torch.arange(num_envs).view(num_envs, 1)
    prev_actions = torch.arange(num_envs).view(num_envs, 1)
    batch = {
        k: torch.arange(num_envs)
        .view(num_envs, 1, 1, 1)
        .expand(num_envs, 3, 256, 256)
        for k in ["a", "b"]
    }
    rgb_frames = [[idx] for idx in range(num_envs)]

    (
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    ) = BaseRLTrainer._pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    )

    expected = list(sorted(set(range(num_envs)) - set(envs_to_pause)))

    assert envs._running == expected

    assert list(test_recurrent_hidden_states.size()) == [4, len(expected), 512]
    assert test_recurrent_hidden_states[0, :, 0].numpy().tolist() == expected

    assert not_done_masks[:, 0].numpy().tolist() == expected
    assert current_episode_reward[:, 0].numpy().tolist() == expected
    assert prev_actions[:, 0].numpy().tolist() == expected
    assert [v[0] for v in rgb_frames] == expected

    for _, v in batch.items():
        assert list(v.size()) == [len(expected), 3, 256, 256]
        assert v[:, 0, 0, 0].numpy().tolist() == expected


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_pausing():
    random.seed(0)
    for _ in range(100):
        num_envs = random.randint(1, 13)
        envs_to_pause = list(range(num_envs))

        random.shuffle(envs_to_pause)
        envs_to_pause = envs_to_pause[: random.randint(0, num_envs)]
        # envs_to_pause is assumed to be sorted in the function
        envs_to_pause = sorted(envs_to_pause)

        __do_pause_test(num_envs, envs_to_pause)

    num_envs = 8
    __do_pause_test(num_envs, [])
    __do_pause_test(num_envs, list(range(num_envs)))
