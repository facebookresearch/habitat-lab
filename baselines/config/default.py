#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from typing import Optional
from habitat import get_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# BASELINE
# -----------------------------------------------------------------------------
_C.BASELINE = CN()
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL)
# -----------------------------------------------------------------------------
_C.BASELINE.RL = CN()
_C.BASELINE.RL.SUCCESS_REWARD = 10.0
_C.BASELINE.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.BASELINE.ORBSLAM2 = CN()
_C.BASELINE.ORBSLAM2.SLAM_VOCAB_PATH = "baselines/slambased/data/ORBvoc.txt"
_C.BASELINE.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.BASELINE.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.BASELINE.ORBSLAM2.MAP_SIZE = 40
_C.BASELINE.ORBSLAM2.CAMERA_HEIGHT = get_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.BASELINE.ORBSLAM2.BETA = 100
_C.BASELINE.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.BASELINE.ORBSLAM2.CAMERA_HEIGHT
_C.BASELINE.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.BASELINE.ORBSLAM2.CAMERA_HEIGHT
_C.BASELINE.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.BASELINE.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.BASELINE.ORBSLAM2.PREPROCESS_MAP = True
_C.BASELINE.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.BASELINE.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.BASELINE.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.BASELINE.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.BASELINE.ORBSLAM2.NUM_ACTIONS = 3
_C.BASELINE.ORBSLAM2.DIST_TO_STOP = 0.05
_C.BASELINE.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.BASELINE.ORBSLAM2.DEPTH_DENORM = (
    get_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
)


def cfg(
    config_file: Optional[str] = None, config_dir: str = DEFAULT_CONFIG_DIR
) -> CN:
    config = _C.clone()
    if config_file:
        config.merge_from_file(os.path.join(config_dir, config_file))
    return config
