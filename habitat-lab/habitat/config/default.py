#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
import threading
from functools import partial
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from habitat.config.default_structured_configs import (
    AgentConfig,
    HabitatConfigPlugin,
    SimulatorConfig,
    register_hydra_plugin,
)
from habitat.config.read_write import read_write

_HABITAT_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# Habitat config directory inside the installed package.
# Used to access default predefined configs.
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
# in editable install, this is pwd/habitat-lab/habitat/config
CONFIG_FILE_SEPARATOR = ","


def get_full_config_path(config_path: str, configs_dir: str) -> str:
    r"""Returns absolute path to the yaml config file if exists, else raises RuntimeError.

    :param config_path: path to the yaml config file.
    :param configs_dir: path to the config files root directory.
    :return: absolute path to the yaml config file.
    """
    if osp.exists(config_path):
        return osp.abspath(config_path)

    proposed_full_path = osp.join(configs_dir, config_path)
    if osp.exists(proposed_full_path):
        return osp.abspath(proposed_full_path)

    raise RuntimeError(f"No file found for config '{config_path}'")


get_full_habitat_config_path = partial(
    get_full_config_path, configs_dir=_HABITAT_CFG_DIR
)
get_full_habitat_config_path.__doc__ = r"""
Returns absolute path to the habitat yaml config file if exists, else raises RuntimeError.

:param config_path: relative path to the habitat yaml config file.
:return: absolute config to the habitat yaml config file.
"""


def get_agent_config(
    sim_config: SimulatorConfig, agent_id: Optional[int] = None
) -> AgentConfig:
    r"""Returns agent's config node of default agent or based on index of the agent.

    :param sim_config: config of :ref:`habitat.core.simulator.Simulator`.
    :param agent_id: index of the agent config (relevant for multi-agent setup).
    :return: relevant agent's config.
    """
    if agent_id is None:
        agent_id = sim_config.default_agent_id

    agent_name = sim_config.agents_order[agent_id]
    agent_config = sim_config.agents[agent_name]

    return agent_config


lock = threading.Lock()


def patch_config(cfg: DictConfig) -> DictConfig:
    """
    Internal method only. Modifies a configuration by inferring some missing keys
    and makes sure some keys are present and compatible with each other.
    """
    # In the single-agent setup use the agent's key from `habitat.simulator.agents`.
    sim_config = cfg.habitat.simulator
    if len(sim_config.agents) == 1:
        with read_write(sim_config):
            sim_config.agents_order = list(sim_config.agents.keys())

    # Check if the `habitat.simulator.agents_order`
    # is set and matches the agents' keys in `habitat.simulator.agents`.
    assert len(sim_config.agents_order) == len(sim_config.agents) and set(
        sim_config.agents_order
    ) == set(sim_config.agents.keys()), (
        "habitat.simulator.agents_order should be set explicitly "
        "and match the agents' keys in habitat.simulator.agents.\n"
        f"habitat.simulator.agents_order: {sim_config.agents_order}\n"
        f"habitat.simulator.agents: {list(sim_config.agents.keys())}"
    )

    OmegaConf.set_readonly(cfg, True)

    return cfg


def register_configs():
    """
    This method will register the Habitat-lab benchmark configurations.
    """
    register_hydra_plugin(HabitatConfigPlugin)


def get_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    configs_dir: str = _HABITAT_CFG_DIR,
) -> DictConfig:
    r"""Returns habitat config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat.seed=1"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_HABITAT_CFG_DIR`).
    :return: composed config object.
    """
    register_configs()
    config_path = get_full_config_path(config_path, configs_dir)
    # If get_config is called from different threads, Hydra might
    # get initialized twice leading to issues. This lock fixes it.
    with lock, initialize_config_dir(
        version_base=None,
        config_dir=osp.dirname(config_path),
    ):
        cfg = compose(
            config_name=osp.basename(config_path),
            overrides=overrides if overrides is not None else [],
        )

    return patch_config(cfg)
