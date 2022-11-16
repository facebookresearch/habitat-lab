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

from habitat.config.default import get_full_config_path
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

_BASELINES_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
DEFAULT_CONFIG_DIR = "habitat-lab/habitat/config/"
CONFIG_FILE_SEPARATOR = ","


lock = threading.Lock()


def get_config(
    config_paths: str,
    overrides: Optional[list] = None,
) -> DictConfig:
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)

    config_path = get_full_config_path(
        config_paths, default_configs_dir=_BASELINES_CFG_DIR
    )
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
