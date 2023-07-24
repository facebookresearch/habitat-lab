#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Type, Union

from habitat.core.env import Env, RLEnv
from habitat.datasets import make_dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_env_fn(
    config: "DictConfig",
    env_class: Union[Type[Env], Type[RLEnv]],
    dataset=None,
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        dataset: If specified, load the environment using this dataset.

    Returns:
        env object created according to specification.
    """
    if "habitat" in config:
        config = config.habitat
    if dataset is None:
        dataset = make_dataset(config.dataset.type, config=config.dataset)
    env = env_class(config=config, dataset=dataset)
    env.seed(config.seed)
    return env
