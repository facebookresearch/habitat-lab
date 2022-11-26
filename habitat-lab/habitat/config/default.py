#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
import threading
from functools import partial
from typing import Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat.config.read_write import read_write

_HABITAT_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
# in editable install, this is pwd/habitat-lab/habitat/config
CONFIG_FILE_SEPARATOR = ","


def get_full_config_path(config_path: str, configs_dir: str) -> str:
    if osp.exists(config_path):
        return osp.abspath(config_path)

    proposed_full_path = osp.join(configs_dir, config_path)
    if osp.exists(proposed_full_path):
        return osp.abspath(proposed_full_path)

    raise RuntimeError(f"No file found for config '{config_path}'")


get_full_habitat_config_path = partial(
    get_full_config_path, configs_dir=_HABITAT_CFG_DIR
)


def get_agent_config(sim_config, agent_id: Optional[int] = None):
    if agent_id is None:
        agent_id = sim_config.default_agent_id

    agent_name = sim_config.agents_order[agent_id]
    agent_config = sim_config.agents[agent_name]

    return agent_config


lock = threading.Lock()


def get_config(
    config_paths: str,
    overrides: Optional[list] = None,
    configs_dir: str = _HABITAT_CFG_DIR,
) -> DictConfig:
    register_hydra_plugin(HabitatConfigPlugin)

    config_path = get_full_config_path(config_paths, configs_dir)
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

    # In the single-agent setup use the agent's key from `habitat.simulator.agents`.
    sim_config = cfg.habitat.simulator
    if len(sim_config.agents) == 1:
        with read_write(sim_config):
            sim_config.agents_order = list(sim_config.agents.keys())

    # Check if the `habitat.simulator.agents_order`
    # is set and matches the agents' keys in `habitat.simulator.agents`.
    assert set(sim_config.agents_order) == set(sim_config.agents.keys()), (
        "habitat.simulator.agents_order should be set explicitly "
        "and match the agents' keys in habitat.simulator.agents"
    )

    OmegaConf.set_readonly(cfg, True)

    return cfg
