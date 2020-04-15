#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import List, Optional, Union

import numpy as np

import habitat
from habitat.config import Config as CN
from habitat.core.utils import Singleton

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
# Use a singleton to add the habitat_baselines config into the habitat config.
# This way it won't always end up being there if you aren't going to use habitat_baselines
# but have it in your import path for some reason
class _HabitatBaselinesDefaultConfigBuilder(metaclass=Singleton):
    def __init__(self):
        habitat_baselines = CN()
        habitat_baselines.cmd_trailing_opts = []
        habitat_baselines.trainer_name = "ppo"
        habitat_baselines.env_name = "NavRLEnv"
        habitat_baselines.simulator_gpu_id = 0
        habitat_baselines.torch_gpu_id = 0
        habitat_baselines.video_option = ["disk", "tensorboard"]
        habitat_baselines.tensorboard_dir = "tb"
        habitat_baselines.video_dir = "video_dir"
        habitat_baselines.test_episode_count = 2
        habitat_baselines.eval_ckpt_path_dir = (
            "data/checkpoints"  # path to ckpt or path to ckpts dir
        )
        habitat_baselines.simulators_per_gpu = 16
        habitat_baselines.checkpoint_folder = "data/checkpoints"
        habitat_baselines.num_updates = 10000
        habitat_baselines.log_interval = 10
        habitat_baselines.log_file = "train.log"
        habitat_baselines.checkpoint_interval = 50
        # -----------------------------------------------------------------------------
        # eval CONFIG
        # -----------------------------------------------------------------------------
        habitat_baselines.eval = CN()
        # The split to evaluate on
        habitat_baselines.eval.split = "val"
        habitat_baselines.eval.use_ckpt_config = True
        # -----------------------------------------------------------------------------
        # REINFORCEMENT LEARNING (RL) environment CONFIG
        # -----------------------------------------------------------------------------
        habitat_baselines.rl = CN()
        habitat_baselines.rl.reward_measure = "distance_to_goal"
        habitat_baselines.rl.success_measure = "success"
        habitat_baselines.rl.success_reward = 10.0
        habitat_baselines.rl.slack_reward = -0.01
        # -----------------------------------------------------------------------------
        # PROXIMAL POLICY OPTIMIZATION (PPO)
        # -----------------------------------------------------------------------------
        habitat_baselines.rl.ppo = CN()
        habitat_baselines.rl.ppo.clip_param = 0.2
        habitat_baselines.rl.ppo.ppo_epoch = 4
        habitat_baselines.rl.ppo.num_mini_batch = 16
        habitat_baselines.rl.ppo.value_loss_coef = 0.5
        habitat_baselines.rl.ppo.entropy_coef = 0.01
        habitat_baselines.rl.ppo.lr = 7e-4
        habitat_baselines.rl.ppo.eps = 1e-5
        habitat_baselines.rl.ppo.max_grad_norm = 0.5
        habitat_baselines.rl.ppo.num_steps = 5
        habitat_baselines.rl.ppo.use_gae = True
        habitat_baselines.rl.ppo.use_linear_lr_decay = False
        habitat_baselines.rl.ppo.use_linear_clip_decay = False
        habitat_baselines.rl.ppo.gamma = 0.99
        habitat_baselines.rl.ppo.tau = 0.95
        habitat_baselines.rl.ppo.reward_window_size = 50
        habitat_baselines.rl.ppo.use_normalized_advantage = True
        habitat_baselines.rl.ppo.hidden_size = 512
        # -----------------------------------------------------------------------------
        # DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
        # -----------------------------------------------------------------------------
        habitat_baselines.rl.ddppo = CN()
        habitat_baselines.rl.ddppo.sync_frac = 0.6
        habitat_baselines.rl.ddppo.distrib_backend = "GLOO"
        habitat_baselines.rl.ddppo.rnn_type = "LSTM"
        habitat_baselines.rl.ddppo.num_recurrent_layers = 2
        habitat_baselines.rl.ddppo.backbone = "resnet50"
        habitat_baselines.rl.ddppo.pretrained_weights = (
            "data/ddppo-models/gibson-2plus-resnet50.pth"
        )
        # Loads pretrained weights
        habitat_baselines.rl.ddppo.pretrained = False
        # Loads just the visual encoder backbone weights
        habitat_baselines.rl.ddppo.pretrained_encoder = False
        # Whether or not the visual encoder backbone will be trained
        habitat_baselines.rl.ddppo.train_encoder = True
        # Whether or not to reset the critic linear layer
        habitat_baselines.rl.ddppo.reset_critic = True
        # -----------------------------------------------------------------------------
        # orbslam2 BASELINE
        # -----------------------------------------------------------------------------
        habitat_baselines.orbslam2 = CN()
        habitat_baselines.orbslam2.slam_vocab_path = (
            "habitat_baselines/slambased/data/ORBvoc.txt"
        )
        habitat_baselines.orbslam2.slam_settings_path = (
            "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
        )
        habitat_baselines.orbslam2.map_cell_size = 0.1
        habitat_baselines.orbslam2.map_size = 40
        habitat_baselines.orbslam2.camera_height = habitat.get_config().habitat.simulator.depth_sensor.position[
            1
        ]
        habitat_baselines.orbslam2.beta = 100
        habitat_baselines.orbslam2.h_obstacle_min = (
            0.3 * habitat_baselines.orbslam2.camera_height
        )
        habitat_baselines.orbslam2.h_obstacle_max = (
            1.0 * habitat_baselines.orbslam2.camera_height
        )
        habitat_baselines.orbslam2.d_obstacle_min = 0.1
        habitat_baselines.orbslam2.d_obstacle_max = 4.0
        habitat_baselines.orbslam2.preprocess_map = True
        habitat_baselines.orbslam2.min_pts_in_obstacle = (
            habitat.get_config().habitat.simulator.depth_sensor.width / 2.0
        )
        habitat_baselines.orbslam2.angle_th = float(np.deg2rad(15))
        habitat_baselines.orbslam2.dist_reached_th = 0.15
        habitat_baselines.orbslam2.next_waypoint_th = 0.5
        habitat_baselines.orbslam2.num_actions = 3
        habitat_baselines.orbslam2.dist_to_stop = 0.05
        habitat_baselines.orbslam2.planner_max_steps = 500
        habitat_baselines.orbslam2.depth_denorm = (
            habitat.get_config().habitat.simulator.depth_sensor.max_depth
        )

        habitat.config.extend_default_config(
            "habitat_baselines", habitat_baselines
        )


# We are also copying the docstring and annotations here as
# code completion doesn't generally do the right thing with @wraps,
# but the @wraps will make sure the docs of these always match
@functools.wraps(habitat.get_config)
def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    _HabitatBaselinesDefaultConfigBuilder()

    config = habitat.get_config(config_paths, opts)
    if opts is not None:
        config.defrost()
        config.habitat_baselines.cmd_trailing_opts = opts
        config.freeze()

    return config
