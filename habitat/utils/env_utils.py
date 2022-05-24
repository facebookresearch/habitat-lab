#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import Any, List, Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, logger, make_dataset


def make_env_fn(
    config: Config, env_class: Union[Type[Env], Type[RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    if "TASK_CONFIG" in config:
        config = config.TASK_CONFIG
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    env = env_class(config=config, dataset=dataset)
    env.seed(config.SEED)
    return env


def construct_envs(
    config: Config,
    env_class: Union[Type[Env], Type[RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    :return: VectorEnv object created according to specification.
    """

    num_environments = config.NUM_ENVIRONMENTS
    configs = []
    env_classes = [env_class for _ in range(num_environments)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_environments < 1:
        raise RuntimeError("NUM_ENVIRONMENTS must be strictly positive")

    if len(scenes) == 0:
        raise RuntimeError(
            "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
        )

    random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]
    if len(scenes) < num_environments:
        logger.warn(
            f"There are less scenes ({len(scenes)}) than environments ({num_environments}). "
            "Each environment will use all the scenes instead of using a subset."
        )
        for scene in scenes:
            for split in scene_splits:
                split.append(scene)
    else:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_environments):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    vector_env_cls: Type[Any]
    if os.environ.get("HABITAT_ENV_DEBUG", 0):
        logger.warn(
            "Using the debug Vector environment interface. Expect slower performance."
        )
        vector_env_cls = habitat.ThreadedVectorEnv
    else:
        vector_env_cls = habitat.VectorEnv

    envs = vector_env_cls(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
