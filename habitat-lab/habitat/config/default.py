#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
import threading
from typing import Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)

_HABITAT_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
# in editable install, this is pwd/habitat-lab/habitat/config
CONFIG_FILE_SEPARATOR = ","


def get_full_config_path(
    config_path: str, default_configs_dir: str = _HABITAT_CFG_DIR
) -> str:
    if osp.exists(config_path):
        return osp.abspath(config_path)

    proposed_full_path = osp.join(default_configs_dir, config_path)
    if osp.exists(proposed_full_path):
        return osp.abspath(proposed_full_path)

    raise RuntimeError(f"No file found for config '{config_path}'")


lock = threading.Lock()


def get_config(
    config_paths: str,
    overrides: Optional[list] = None,
) -> DictConfig:
    register_hydra_plugin(HabitatConfigPlugin)

    config_path = get_full_config_path(config_paths)
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

    OmegaConf.set_readonly(cfg, True)

    return cfg
