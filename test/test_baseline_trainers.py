#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import random
from copy import deepcopy
from glob import glob

import pytest

try:
    import torch
    import torch.distributed

    from habitat_baselines.common.base_trainer import BaseRLTrainer
    from habitat_baselines.config.default import get_config
    from habitat_baselines.run import execute_exp, run_exp

    baseline_installed = True
except ImportError:
    baseline_installed = False


def _powerset(s):
    return [
        combo
        for r in range(len(s) + 1)
        for combo in itertools.combinations(s, r)
    ]


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path,mode,gpu2gpu,observation_transforms",
    itertools.product(
        glob("habitat_baselines/config/test/*"),
        ["train", "eval"],
        [True, False],
        [
            [],
            [
                "CenterCropper",
                "ResizeShortestEdge",
            ],
        ],
    ),
)
def test_trainers(test_cfg_path, mode, gpu2gpu, observation_transforms):
    if gpu2gpu:
        try:
            import habitat_sim
        except ImportError:
            pytest.skip("GPU-GPU requires Habitat-Sim")

        if not habitat_sim.cuda_enabled:
            pytest.skip("GPU-GPU requires CUDA")

    run_exp(
        test_cfg_path,
        mode,
        [
            "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_GPU",
            str(gpu2gpu),
            "RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS",
            str(tuple(observation_transforms)),
        ],
    )

    # Deinit processes group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path,mode",
    [
        ["habitat_baselines/config/test/ppo_pointnav_test.yaml", "train"],
    ],
)
def test_equirect_stiching(test_cfg_path, mode: str):
    meta_config = get_config(config_paths=test_cfg_path)
    meta_config.defrost()
    config = meta_config.TASK_CONFIG
    CAMERA_NUM = 6
    orient = [
        [0, math.pi, 0],  # Back
        [-math.pi / 2, 0, 0],  # Down
        [0, 0, 0],  # Front
        [0, math.pi / 2, 0],  # Right
        [0, 3 / 2 * math.pi, 0],  # Left
        [math.pi / 2, 0, 0],  # Up
    ]
    sensor_uuids = []

    if "RGB_SENSOR" in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR.RGB_SENSOR.ORIENTATION = orient[0]
        for camera_id in range(1, CAMERA_NUM):
            camera_template = f"RGB_{camera_id}"
            camera_config = deepcopy(config.SIMULATOR.RGB_SENSOR)
            camera_config.ORIENTATION = orient[camera_id]

            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)
            setattr(config.SIMULATOR, camera_template, camera_config)
            config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)

    if "DEPTH_SENSOR" in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR.DEPTH_SENSOR.ORIENTATION = orient[0]
        for camera_id in range(1, CAMERA_NUM):
            camera_template = f"DEPTH_{camera_id}"
            camera_config = deepcopy(config.SIMULATOR.DEPTH_SENSOR)
            camera_config.ORIENTATION = orient[camera_id]
            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)

            setattr(config.SIMULATOR, camera_template, camera_config)
            config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)

    meta_config.TASK_CONFIG = config
    meta_config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
    meta_config.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.SENSOR_UUIDS = tuple(
        sensor_uuids
    )
    meta_config.freeze()
    execute_exp(meta_config, mode)
    # Deinit processes group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_eval_config():
    ckpt_opts = ["VIDEO_OPTION", "[]"]
    eval_opts = ["VIDEO_OPTION", "['disk']"]

    ckpt_cfg = get_config(None, ckpt_opts)
    assert ckpt_cfg.VIDEO_OPTION == []
    assert ckpt_cfg.CMD_TRAILING_OPTS == ["VIDEO_OPTION", "[]"]

    eval_cfg = get_config(None, eval_opts)
    assert eval_cfg.VIDEO_OPTION == ["disk"]
    assert eval_cfg.CMD_TRAILING_OPTS == ["VIDEO_OPTION", "['disk']"]

    trainer = BaseRLTrainer(get_config())
    assert trainer.config.VIDEO_OPTION == ["disk", "tensorboard"]
    returned_config = trainer._setup_eval_config(checkpoint_config=ckpt_cfg)
    assert returned_config.VIDEO_OPTION == []

    trainer = BaseRLTrainer(eval_cfg)
    returned_config = trainer._setup_eval_config(ckpt_cfg)
    assert returned_config.VIDEO_OPTION == ["disk"]


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

    expected = sorted(set(range(num_envs)) - set(envs_to_pause))

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
