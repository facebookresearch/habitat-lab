#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type

import habitat
from habitat import Config, Env, VectorEnv, make_dataset


def make_env_fn(
    task_config: Config, rl_env_config: Config, env_class: Type, rank: int
) -> Env:
    r"""
    Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        task_config: task config file for creating env.
        rl_env_config: RL env config for creating env.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        task_config.DATASET.TYPE, config=task_config.DATASET
    )
    env = env_class(
        config_env=task_config, config_baseline=rl_env_config, dataset=dataset
    )
    env.seed(rank)
    return env


def construct_envs(config: Config, env_class: Type) -> VectorEnv:
    r"""
    Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.

    Returns:
        VectorEnv object created according to specification.
    """
    trainer_config = config.TRAINER.RL.PPO
    rl_env_config = config.TRAINER.RL
    task_config = config.TASK_CONFIG  # excluding trainer-specific configs
    env_configs, rl_env_configs = [], []
    env_classes = [env_class for _ in range(trainer_config.num_processes)]
    dataset = make_dataset(task_config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(task_config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= trainer_config.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(trainer_config.num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(trainer_config.num_processes):

        env_config = task_config.clone()
        env_config.defrost()
        if len(scenes) > 0:
            env_config.DATASET.CONTENT_SCENES = scene_splits[i]

        env_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            trainer_config.sim_gpu_id
        )

        agent_sensors = trainer_config.sensors.strip().split(",")
        env_config.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        env_config.freeze()
        env_configs.append(env_config)
        rl_env_configs.append(rl_env_config)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    env_configs,
                    rl_env_configs,
                    env_classes,
                    range(trainer_config.num_processes),
                )
            )
        ),
    )
    return envs
