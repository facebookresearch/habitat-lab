#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import itertools
import os
import random

import numpy as np
import pytest

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat_baselines.run import execute_exp

try:
    import torch
    import torch.distributed

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
            "rearrange/rl_skill.yaml",
            3,
            [
                "habitat.dataset.split=minival",
                "benchmark/rearrange/skills=place",
            ],
        ),
        (
            "rearrange/rl_skill.yaml",
            3,
            ["benchmark/rearrange/skills=open_cab"],
        ),
        (
            "rearrange/rl_skill.yaml",
            3,
            [
                "benchmark/rearrange/skills=open_fridge",
            ],
        ),
        (
            "rearrange/rl_skill.yaml",
            3,
            [
                "habitat.dataset.split=minival",
                "benchmark/rearrange/skills=pick",
            ],
        ),
        (
            "rearrange/rl_skill.yaml",
            3,
            [
                "habitat.dataset.split=minival",
                "benchmark/rearrange/skills=nav_to_obj",
            ],
        ),
        (
            "rearrange/rl_skill.yaml",
            3,
            [
                "benchmark/rearrange/skills=close_fridge",
            ],
        ),
        (
            "rearrange/rl_skill.yaml",
            3,
            ["benchmark/rearrange/skills=close_cab"],
        ),
        (
            "imagenav/ddppo_imagenav_example.yaml",
            3,
            [],
        ),
        (
            "objectnav/ddppo_objectnav_hssd-hab.yaml",
            3,
            [],
        ),
        (
            "objectnav/ddppo_objectnav_procthor-hab.yaml",
            3,
            [],
        ),
    ],
)
@pytest.mark.parametrize("trainer_name", ["ddppo", "ver"])
@pytest.mark.parametrize(
    "use_batch_renderer", [False]
)  # Batch renderer test temporarily disabled.
def test_trainers(
    config_path: str,
    num_updates: int,
    overrides: str,
    trainer_name: str,
    use_batch_renderer: bool,
):
    if use_batch_renderer:
        if config_path in [
            "imagenav/ddppo_imagenav_example.yaml",
            "objectnav/ddppo_objectnav_hssd-hab.yaml",
            "objectnav/ddppo_objectnav_procthor-hab.yaml",
        ]:
            pytest.skip(
                "Batch renderer incompatible with this config due to usage of multiple sensors."
            )
        if trainer_name == "ver":
            pytest.skip("Batch renderer incompatible with VER trainer.")

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

    if config_path in [
        "objectnav/ddppo_objectnav_hssd-hab.yaml",
        "objectnav/ddppo_objectnav_procthor-hab.yaml",
    ] and not ObjectNavDatasetV1.check_config_paths_exist(
        config.habitat.dataset
    ):
        pytest.skip("Test skipped as dataset files are missing.")

    with read_write(config):
        agent_config = get_agent_config(config.habitat.simulator)
        # Changing the visual observation size for speed
        for sim_sensor_config in agent_config.sim_sensors.values():
            sim_sensor_config.update({"height": 64, "width": 64})
        # Set config for batch renderer
        if use_batch_renderer:
            config.habitat.simulator.renderer.enable_batch_renderer = True
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
            config.habitat.simulator.create_renderer = False
            config.habitat.simulator.concur_render = False

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


@pytest.mark.parametrize(
    "config_path,policy_type,skill_type,mode",
    list(
        itertools.product(
            [
                "rearrange/rl_hierarchical_oracle_nav.yaml",
                "rearrange/rl_hierarchical.yaml",
            ],
            [
                "hl_neural",
                "hl_fixed",
            ],
            [
                "nn_skills",
                "oracle_skills",
            ],
            [
                "eval",
                "train",
            ],
        )
    ),
)
def test_hrl(config_path, policy_type, skill_type, mode):
    TRAIN_LOG_FILE = "data/test_train.log"

    if policy_type == "hl_neural" and skill_type == "nn_skills":
        return
    if policy_type == "hl_neural" and mode == "eval":
        # We don't have skill checkpoints to load right now.
        return
    if policy_type == "hl_fixed" and mode == "train":
        # Cannot train with a fixed policy
        return
    if skill_type == "oracle_skills" and "oracle" not in config_path:
        return
    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)
    if os.path.exists(TRAIN_LOG_FILE):
        os.remove(TRAIN_LOG_FILE)

    # Setup the training
    config = get_config(
        config_path,
        [
            "habitat_baselines.num_updates=1",
            "habitat_baselines.eval.split=minival",
            "habitat.dataset.split=minival",
            "habitat_baselines.total_num_steps=-1.0",
            "habitat_baselines.test_episode_count=1",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            f"habitat_baselines.log_file={TRAIN_LOG_FILE}",
            f"habitat_baselines/rl/policy@habitat_baselines.rl.policy.main_agent={policy_type}",
            f"habitat_baselines/rl/policy/hierarchical_policy/defined_skills@habitat_baselines.rl.policy.main_agent.hierarchical_policy.defined_skills={skill_type}",
        ],
    )
    with read_write(config):
        config.habitat_baselines.eval.update({"video_option": []})
        for (
            skill_name,
            skill,
        ) in (
            config.habitat_baselines.rl.policy.main_agent.hierarchical_policy.defined_skills.items()
        ):
            if skill.load_ckpt_file == "":
                continue
            skill.update(
                {
                    "force_config_file": f"benchmark/rearrange/skills={skill_name}",
                    "max_skill_steps": 1,
                    "load_ckpt_file": "",
                }
            )
        execute_exp(config, mode)


@pytest.mark.skipif(
    int(os.environ.get("TEST_BASELINE_SMALL", 0)) == 0,
    reason="Full training tests did not run. Need `export TEST_BASELINE_SMALL=1",
)
@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "config_path",
    [
        "social_rearrange/pop_play.yaml",
        "social_rearrange/plan_pop.yaml",
        "social_nav/social_nav.yaml",
    ],
)
def test_multi_agent_trainer(
    config_path: str,
):
    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)
    # Setup the training
    config = get_config(
        config_path,
        [
            "habitat_baselines.num_updates=2",
            "habitat_baselines.rl.ppo.num_mini_batch=1",
            "habitat_baselines.num_environments=1",
            "habitat_baselines.total_num_steps=-1.0",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            "habitat.dataset.data_path=data/hab3_bench_assets/episode_datasets/small_small.json.gz",
            "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/hab3_bench_assets/humanoids/female_0/female_0.urdf",
            "habitat.simulator.agents.agent_1.motion_data_path=data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl",
            "habitat.dataset.scenes_dir=data/hab3_bench_assets/hab3-hssd/",
        ],
    )

    with read_write(config):
        agent_config = get_agent_config(config.habitat.simulator)
        # Changing the visual observation size for speed. However,
        # social nav has a specific sensor that is designed for real robot,
        # which cannot change the sensor size
        if "social_nav" not in config_path:
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
