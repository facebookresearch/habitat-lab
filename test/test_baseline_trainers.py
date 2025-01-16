#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import itertools
import math
import os
import random
from copy import deepcopy

import pytest

from habitat.config.default import get_agent_config
from habitat.core.vector_env import VectorEnv

try:
    import torch
    import torch.distributed

    from habitat_baselines.common.base_trainer import BaseRLTrainer
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.config.default import get_config
    from habitat_baselines.rl.ddppo.ddp_utils import find_free_port
    from habitat_baselines.run import execute_exp
    from habitat_baselines.utils.common import batch_obs

    baseline_installed = True
except ImportError:
    baseline_installed = False

from habitat import make_dataset
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    HeadDepthSensorConfig,
    HeadRGBSensorConfig,
)
from habitat.gym import make_gym_from_config
from habitat_baselines.config.default_structured_configs import (
    Cube2EqConfig,
    Cube2FishConfig,
)
from habitat_baselines.rl.ppo.evaluator import pause_envs


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path,gpu2gpu,observation_transforms_overrides,mode",
    list(
        itertools.product(
            ["test/config/habitat_baselines/ppo_pointnav_test.yaml"],
            [True, False],
            [
                [],
                [
                    "+habitat_baselines/rl/policy/obs_transforms@habitat_baselines.rl.policy.main_agent.obs_transforms.center_cropper=center_cropper_base",
                    "+habitat_baselines/rl/policy/obs_transforms@habitat_baselines.rl.policy.main_agent.obs_transforms.resize_shortest_edge=resize_shortest_edge_base",
                ],
            ],
            ["train", "eval"],
        )
    ),
)
def test_trainers(
    test_cfg_path, gpu2gpu, observation_transforms_overrides, mode
):
    # For testing with world_size=1
    os.environ["MAIN_PORT"] = str(find_free_port())

    test_cfg_cleaned_path = test_cfg_path.replace(
        "habitat-baselines/habitat_baselines/config/", ""
    )

    dataset_config = get_config(test_cfg_cleaned_path).habitat.dataset
    dataset = make_dataset(id_dataset=dataset_config.type)
    if not dataset.check_config_paths_exist(dataset_config):
        pytest.skip("Test skipped as dataset files are missing.")

    if gpu2gpu:
        try:
            import habitat_sim
        except ImportError:
            pytest.skip("GPU-GPU requires Habitat-Sim")

        if not habitat_sim.cuda_enabled:
            pytest.skip("GPU-GPU requires CUDA")

    try:
        baselines_config = get_config(
            test_cfg_cleaned_path,
            [
                f"habitat.simulator.habitat_sim_v0.gpu_gpu={str(gpu2gpu)}",
            ]
            + observation_transforms_overrides,
        )
        execute_exp(baselines_config, mode)
    finally:
        # Needed to destroy the trainer
        gc.collect()

        # Deinit processes group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path",
    (
        "test/config/habitat_baselines/ddppo_pointnav_test.yaml",
        "rearrange/rl_skill.yaml",
    ),
)
@pytest.mark.parametrize("variable_experience", [True, False])
@pytest.mark.parametrize("overlap_rollouts_and_learn", [True, False])
def test_ver_trainer(
    test_cfg_path,
    variable_experience,
    overlap_rollouts_and_learn,
):
    # For testing with world_size=1
    os.environ["MAIN_PORT"] = str(find_free_port())
    try:
        baselines_config = get_config(
            test_cfg_path,
            [
                "habitat_baselines.num_environments=4",
                "habitat_baselines.trainer_name=ver",
                f"habitat_baselines.rl.ver.variable_experience={str(variable_experience)}",
                f"habitat_baselines.rl.ver.overlap_rollouts_and_learn={str(overlap_rollouts_and_learn)}",
                "+habitat_baselines/rl/policy/obs_transforms@habitat_baselines.rl.policy.main_agent.obs_transforms.center_cropper=center_cropper_base",
                "+habitat_baselines/rl/policy/obs_transforms@habitat_baselines.rl.policy.main_agent.obs_transforms.resize_shorter_edge=resize_shortest_edge_base",
                "habitat_baselines.num_updates=2",
                "habitat_baselines.total_num_steps=-1",
                "habitat_baselines.rl.preemption.save_state_batch_only=True",
                "habitat_baselines.rl.ppo.num_steps=16",
            ],
        )
        execute_exp(baselines_config, "train")
    finally:
        # Needed to destroy the trainer
        gc.collect()

        # Deinit processes group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_cpca():
    cfg = get_config(
        "test/config/habitat_baselines/ppo_pointnav_test.yaml",
        ["+habitat_baselines/rl/auxiliary_losses=cpca"],
    )
    assert "cpca" in cfg.habitat_baselines.rl.auxiliary_losses

    execute_exp(cfg, "train")


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "test_cfg_path,mode",
    [
        [
            "test/config/habitat_baselines/ppo_pointnav_test.yaml",
            "train",
        ],
    ],
)
@pytest.mark.parametrize("camera", ["equirect", "fisheye", "cubemap"])
@pytest.mark.parametrize("sensor_type", ["rgb", "depth"])
def test_cubemap_stitching(
    test_cfg_path: str, mode: str, camera: str, sensor_type: str
):
    meta_config = get_config(config_path=test_cfg_path)
    with read_write(meta_config):
        config = meta_config.habitat
        CAMERA_NUM = 6
        orient: list[list[float]] = [
            [0, math.pi, 0],  # Back
            [-math.pi / 2, 0, 0],  # Down
            [0, 0, 0],  # Front
            [0, math.pi / 2, 0],  # Right
            [0, 3 / 2 * math.pi, 0],  # Left
            [math.pi / 2, 0, 0],  # Up
        ]
        sensor_uuids = []

        agent_config = get_agent_config(config.simulator)
        if sensor_type == "rgb":
            agent_config.sim_sensors = {
                "rgb_sensor": HeadRGBSensorConfig(width=256, height=256),
            }
        elif sensor_type == "depth":
            agent_config.sim_sensors = {
                "depth_sensor": HeadDepthSensorConfig(width=256, height=256),
            }
        else:
            raise ValueError(
                "Typo in the sensor type in test_cubemap_stitching"
            )

        sensor = agent_config.sim_sensors[f"{sensor_type}_sensor"]
        for camera_id in range(CAMERA_NUM):
            camera_template = f"{sensor_type}_{camera_id}"
            camera_config = deepcopy(sensor)
            camera_config.orientation = orient[camera_id]
            camera_config.uuid = camera_template.lower()
            sensor_uuids.append(camera_config.uuid)
            agent_config.sim_sensors[camera_template] = camera_config

        meta_config.habitat = config

        if camera == "equirect":
            meta_config.habitat_baselines.rl.policy.main_agent.obs_transforms = {
                "cube2eq": Cube2EqConfig(
                    sensor_uuids=sensor_uuids, width=256, height=256
                )
            }
        elif camera == "fisheye":
            meta_config.habitat_baselines.rl.policy.main_agent.obs_transforms = {
                "cube2fish": Cube2FishConfig(
                    sensor_uuids=sensor_uuids, width=256, height=256
                )
            }

    if camera in ["equirect", "fisheye"]:
        execute_exp(meta_config, mode)
        # Deinit processes group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    elif camera == "cubemap":
        # 1) Generate an equirect image from cubemap images.
        # 2) Generate cubemap images from the equirect image.
        # 3) Compare the input and output cubemap
        env_fn_args = []
        for split in ["train", "val"]:
            tmp_config = config.copy()
            with read_write(tmp_config):
                tmp_config.dataset["split"] = split
            env_fn_args.append((tmp_config,))

        with VectorEnv(
            make_env_fn=make_gym_from_config, env_fn_args=env_fn_args
        ) as envs:
            observations = envs.reset()
        batch = batch_obs(observations)
        orig_batch = deepcopy(batch)

        #  ProjectionTransformer
        obs_trans_to_eq = baseline_registry.get_obs_transformer(
            "CubeMap2Equirect"
        )
        cube2equirect = obs_trans_to_eq(sensor_uuids, (256, 512))
        obs_trans_to_cube = baseline_registry.get_obs_transformer(
            "Equirect2CubeMap"
        )
        equirect2cube = obs_trans_to_cube(
            cube2equirect.target_uuids, (256, 256)
        )

        # Cubemap to Equirect to Cubemap
        batch_eq = cube2equirect(batch)
        batch_cube = equirect2cube(batch_eq)

        # Extract input and output cubemap
        output_cube = batch_cube[cube2equirect.target_uuids[0]]
        input_cube = [orig_batch[key] for key in sensor_uuids]
        input_cube = torch.stack(input_cube, dim=1)  # type: ignore[arg-type]
        input_cube = torch.flatten(input_cube, end_dim=1)

        # Apply blur to absorb difference (blur, etc.) caused by conversion
        if sensor_type == "rgb":
            output_cube = output_cube.float() / 255
            input_cube = input_cube.float() / 255
        output_cube = output_cube.permute((0, 3, 1, 2))  # NHWC => NCHW
        input_cube = input_cube.permute((0, 3, 1, 2))  # NHWC => NCHW
        apply_blur = torch.nn.AvgPool2d(5, 3, 2)
        output_cube = apply_blur(output_cube)
        input_cube = apply_blur(input_cube)

        # Calculate the difference
        diff = torch.abs(output_cube - input_cube)
        assert diff.mean().item() < 0.01
    else:
        raise ValueError(f"Unknown camera name: {camera}")


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
def test_eval_config():
    ckpt_opts = [
        "habitat_baselines.eval.video_option=[]",
        "habitat_baselines.load_resume_state_config=True",
    ]
    eval_opts = [
        "habitat_baselines.eval.video_option=['disk']",
        "habitat_baselines.load_resume_state_config=False",
    ]

    ckpt_cfg = get_config(
        "test/config/habitat_baselines/ppo_pointnav_test.yaml", ckpt_opts
    )
    assert ckpt_cfg.habitat_baselines.eval.video_option == []

    eval_cfg = get_config(
        "test/config/habitat_baselines/ppo_pointnav_test.yaml", eval_opts
    )
    assert eval_cfg.habitat_baselines.eval.video_option == ["disk"]

    trainer = BaseRLTrainer(
        get_config("test/config/habitat_baselines/ppo_pointnav_test.yaml")
    )

    returned_config = trainer._get_resume_state_config_or_new_config(
        resume_state_config=ckpt_cfg
    )
    assert returned_config.habitat_baselines.eval.video_option == []

    trainer = BaseRLTrainer(eval_cfg)
    # Load state config is false. This means that _get_resume_state_config_or_new_config
    # should use the new (eval_cfg) config instead of the resume_state_config
    returned_config = trainer._get_resume_state_config_or_new_config(
        resume_state_config=ckpt_cfg
    )
    assert returned_config.habitat_baselines.eval.video_option == ["disk"]


def __do_pause_test(num_envs, envs_to_pause):
    class PausableShim(VectorEnv):
        def __init__(self, num_envs):
            self._running = list(range(num_envs))

        @property
        def num_envs(self):
            return len(self._running)

        def pause_at(self, idx):
            self._running.pop(idx)

    envs: PausableShim = PausableShim(num_envs)
    test_recurrent_hidden_states = (
        torch.arange(num_envs).view(num_envs, 1, 1).expand(num_envs, 4, 512)
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
    ) = pause_envs(
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

    assert envs._running == expected  # type: ignore[attr-defined]

    assert list(test_recurrent_hidden_states.size()) == [len(expected), 4, 512]
    assert test_recurrent_hidden_states[:, 0, 0].numpy().tolist() == expected

    assert not_done_masks[:, 0].numpy().tolist() == expected
    assert current_episode_reward[:, 0].numpy().tolist() == expected
    assert prev_actions[:, 0].numpy().tolist() == expected
    assert [v[0] for v in rgb_frames] == expected

    for v in batch.values():
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


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "sensor_device,batched_device",
    [("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cuda")],
)
def test_batch_obs(sensor_device, batched_device):
    if (
        "cuda" in (sensor_device, batched_device)
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA not available")

    sensor_device = torch.device(sensor_device)
    batched_device = torch.device(batched_device)

    numpy_if = lambda t: t.numpy() if sensor_device.type == "cpu" else t

    sensors = [
        {
            f"{s}": numpy_if(torch.randn(128, 128, device=sensor_device))
            for s in range(4)
        }
        for _ in range(4)
    ]

    _ = batch_obs(sensors, device=batched_device)
