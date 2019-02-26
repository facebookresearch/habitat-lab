#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# BASELINES
# -----------------------------------------------------------------------------
_C.BASELINE = CN()
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL)
# -----------------------------------------------------------------------------
_C.BASELINE.RL = CN()
_C.BASELINE.RL.SUCCESS_REWARD = 10.0
_C.BASELINE.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------


def cfg(
    config_file: Optional[str] = None, config_dir: str = DEFAULT_CONFIG_DIR
) -> CN:
    config = _C.clone()
    if config_file:
        config.merge_from_file(os.path.join(config_dir, config_file))
    return config
