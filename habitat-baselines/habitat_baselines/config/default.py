#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
from typing import Optional

from omegaconf import DictConfig

from habitat.config.default import get_config as _habitat_get_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

_BASELINES_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
DEFAULT_CONFIG_DIR = "habitat-lab/habitat/config/"
CONFIG_FILE_SEPARATOR = ","


def get_config(
    config_paths: str,
    overrides: Optional[list] = None,
    configs_dir: str = _BASELINES_CFG_DIR,
) -> DictConfig:
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    cfg = _habitat_get_config(config_paths, overrides, configs_dir)

    return cfg
