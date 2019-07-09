#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# BASELINE
# -----------------------------------------------------------------------------
_C.BASELINE = CN()
_C.BASELINE.TRAINER_NAME = "ppo"
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL)
# -----------------------------------------------------------------------------
_C.BASELINE.RL = CN()
_C.BASELINE.RL.SUCCESS_REWARD = 10.0
_C.BASELINE.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.BASELINE.RL.PPO = CN()
_C.BASELINE.RL.PPO.clip_param = 0.2
_C.BASELINE.RL.PPO.ppo_epoch = 4
_C.BASELINE.RL.PPO.num_mini_batch = 16
_C.BASELINE.RL.PPO.value_loss_coef = 0.5
_C.BASELINE.RL.PPO.entropy_coef = 0.01
_C.BASELINE.RL.PPO.lr = 7e-4
_C.BASELINE.RL.PPO.eps = 1e-5
_C.BASELINE.RL.PPO.max_grad_norm = 0.5
_C.BASELINE.RL.PPO.num_steps = 5
_C.BASELINE.RL.PPO.hidden_size = 512
_C.BASELINE.RL.PPO.num_processes = 16
_C.BASELINE.RL.PPO.use_gae = True
_C.BASELINE.RL.PPO.use_linear_lr_decay = False
_C.BASELINE.RL.PPO.use_linear_clip_decay = False
_C.BASELINE.RL.PPO.gamma = 0.99
_C.BASELINE.RL.PPO.tau = 0.95
_C.BASELINE.RL.PPO.log_file = "train.log"
_C.BASELINE.RL.PPO.reward_window_size = 50
_C.BASELINE.RL.PPO.log_interval = 1
_C.BASELINE.RL.PPO.checkpoint_interval = 50
_C.BASELINE.RL.PPO.checkpoint_folder = "data/checkpoints"
_C.BASELINE.RL.PPO.sim_gpu_id = 0
_C.BASELINE.RL.PPO.pth_gpu_id = 0
_C.BASELINE.RL.PPO.num_updates = 10000
_C.BASELINE.RL.PPO.sensors = "RGB_SENSOR,DEPTH_SENSOR"
_C.BASELINE.RL.PPO.task_config = "configs/tasks/pointnav.yaml"
_C.BASELINE.RL.PPO.tensorboard_dir = "tb"
_C.BASELINE.RL.PPO.count_test_episodes = 2
_C.BASELINE.RL.PPO.video_option = ["disk", "tensorboard"]
_C.BASELINE.RL.PPO.video_dir = "video_Dir"
_C.BASELINE.RL.PPO.tracking_model_dir = ""
_C.BASELINE.RL.PPO.model_path = ""
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.BASELINE.ORBSLAM2 = CN()
_C.BASELINE.ORBSLAM2.SLAM_VOCAB_PATH = (
    "habitat_baselines/slambased/data/ORBvoc.txt"
)
_C.BASELINE.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
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


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
