#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type

import numpy as np

import habitat
from habitat import Config, Env, VectorEnv, make_dataset


def make_env_fn(config: Config, env_class: Type, rank: int) -> Env:
    r"""
    Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: config file for creating env.
        env_class: class type of the env to be created.
        rank: rank of env to be created.

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    config.defrost()
    config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config.freeze()
    env = env_class(config_env=config, config_baseline=config, dataset=dataset)
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
        vectorEnv object created according to specification.
    """
    baseline_cfg = config.BASELINE.RL.PPO
    env_configs = []
    env_classes = [env_class for _ in range(baseline_cfg.num_processes)]
    dataset = make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= baseline_cfg.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(
            np.floor(len(scenes) / baseline_cfg.num_processes)
        )

    scene_splits = [[] for _ in range(baseline_cfg.num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(baseline_cfg.num_processes):

        config_env = config.clone()
        config_env.defrost()
        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scene_splits[i]

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            baseline_cfg.sim_gpu_id
        )

        agent_sensors = baseline_cfg.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in [
                "RGB_SENSOR",
                "DEPTH_SENSOR",
            ], "currently 'RGB_SENSOR' and 'DEPTH_SENSOR' are supported "
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.freeze()
        env_configs.append(config_env)

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(
                    env_configs, env_classes, range(baseline_cfg.num_processes)
                )
            )
        ),
    )
    return envs
