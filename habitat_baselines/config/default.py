#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
# task config can be a list of conifgs like "A.yaml,B.yaml"
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = -1
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_ENVIRONMENTS = 16
_C.NUM_PROCESSES = -1  # depricated
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.NUM_CHECKPOINTS = 10
# Number of model updates between checkpoints
_C.CHECKPOINT_INTERVAL = -1
_C.TOTAL_NUM_STEPS = -1.0
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.FORCE_BLIND_POLICY = False
_C.VERBOSE = True
# For our use case, the CPU side things are mainly memory copies
# and nothing of substantive compute. PyTorch has been making
# more and more memory copies parallel, but that just ends up
# slowing those down dramatically and reducing our perf.
# This forces it to be single threaded.  The default
# value is left as false as it's different than how
# PyTorch normally behaves, but all configs we provide
# set it to true and yours likely should too
_C.FORCE_TORCH_SINGLE_THREADED = False
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 2.5
_C.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# preemption CONFIG
# -----------------------------------------------------------------------------
_C.RL.preemption = CN()
# Append the slurm job ID to the resume state filename if running a slurm job
# This is useful when you want to have things from a different job but same
# same checkpoint dir not resume.
_C.RL.preemption.append_slurm_job_id = False
# Number of gradient updates between saving the resume state
_C.RL.preemption.save_resume_state_interval = 100
# Save resume states only when running with slurm
# This is nice if you don't want debug jobs to resume
_C.RL.preemption.save_state_batch_only = False
# -----------------------------------------------------------------------------
# POLICY CONFIG
# -----------------------------------------------------------------------------
_C.RL.POLICY = CN()
_C.RL.POLICY.name = "PointNavResNetPolicy"
_C.RL.POLICY.action_distribution_type = "categorical"  # or 'gaussian'
# For gaussian action distribution:
_C.RL.POLICY.ACTION_DIST = CN()
_C.RL.POLICY.ACTION_DIST.use_log_std = False
_C.RL.POLICY.ACTION_DIST.use_softplus = False
_C.RL.POLICY.ACTION_DIST.min_std = 1e-6
_C.RL.POLICY.ACTION_DIST.max_std = 1
_C.RL.POLICY.ACTION_DIST.min_log_std = -5
_C.RL.POLICY.ACTION_DIST.max_log_std = 2
# For continuous action distributions (including gaussian):
_C.RL.POLICY.ACTION_DIST.action_activation = "tanh"  # ['tanh', '']
# -----------------------------------------------------------------------------
# OBS_TRANSFORMS CONFIG
# -----------------------------------------------------------------------------
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = tuple()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER.WIDTH = 256
_C.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.WIDTH = 512
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ.SENSOR_UUIDS = list()
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.WIDTH = 256
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.FOV = 180
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.PARAMS = (0.2, 0.2, 0.2)
_C.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH.SENSOR_UUIDS = list()
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE = CN()
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE.HEIGHT = 256
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE.WIDTH = 256
_C.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE.SENSOR_UUIDS = list()
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 2
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = False
_C.RL.PPO.hidden_size = 512
# Use double buffered sampling, typically helps
# when environment time is similar or large than
# policy inference time during rollout generation
# Not that this does not change the memory requirements
_C.RL.PPO.use_double_buffered_sampler = False
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"
_C.RL.DDPPO.rnn_type = "GRU"
_C.RL.DDPPO.num_recurrent_layers = 1
_C.RL.DDPPO.backbone = "resnet18"
_C.RL.DDPPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Loads just the visual encoder backbone weights
_C.RL.DDPPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.DDPPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
# Forces distributed mode for testing
_C.RL.DDPPO.force_distributed = False
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
# -----------------------------------------------------------------------------
# PROFILING
# -----------------------------------------------------------------------------
_C.PROFILING = CN()
_C.PROFILING.CAPTURE_START_STEP = -1
_C.PROFILING.NUM_STEPS_TO_CAPTURE = -1


_C.register_renamed_key


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, ``opts = ['FOO.BAR',
        0.5]``. Argument can be used for parameter sweeping or quick tests.
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
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    if config.NUM_PROCESSES != -1:
        warnings.warn(
            "NUM_PROCESSES is depricated and will be removed in a future version."
            "  Use NUM_ENVIRONMENTS instead."
            "  Overwriting NUM_ENVIRONMENTS with NUM_PROCESSES for backwards compatibility."
        )

        config.NUM_ENVIRONMENTS = config.NUM_PROCESSES

    config.freeze()
    return config
