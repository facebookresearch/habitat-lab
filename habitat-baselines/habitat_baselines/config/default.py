#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN

DEFAULT_CONFIG_DIR = "habitat-lab/habitat/config/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
# task config can be a list of conifgs like "A.yaml,B.yaml"
_C.base_task_config_path = "habitat-lab/habitat/config/tasks/pointnav.yaml"
_C.cmd_trailing_opts = []  # store command line options as list of strings
_C.trainer_name = "ppo"
_C.simulator_gpu_id = 0
_C.torch_gpu_id = 0
_C.video_option = ["disk", "tensorboard"]
_C.video_render_views = []
_C.tensorboard_dir = "tb"
_C.writer_type = "tb"
_C.video_dir = "video_dir"
_C.video_fps = 10
_C.video_render_top_down = True
_C.video_render_all_info = False
_C.test_episode_count = -1
_C.eval_ckpt_path_dir = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.num_environments = 16
_C.num_processes = -1  # depricated
_C.sensors = ["rgb_sensor", "depth_sensor"]
_C.checkpoint_folder = "data/checkpoints"
_C.num_updates = 10000
_C.num_checkpoints = 10
# Number of model updates between checkpoints
_C.checkpoint_interval = -1
_C.total_num_steps = -1.0
_C.log_interval = 10
_C.log_file = "train.log"
_C.force_blind_policy = False
_C.verbose = True
_C.eval_keys_to_include_in_name = []
# For our use case, the CPU side things are mainly memory copies
# and nothing of substantive compute. PyTorch has been making
# more and more memory copies parallel, but that just ends up
# slowing those down dramatically and reducing our perf.
# This forces it to be single threaded.  The default
# value is left as false as it's different than how
# PyTorch normally behaves, but all configs we provide
# set it to true and yours likely should too
_C.force_torch_single_threaded = False
# -----------------------------------------------------------------------------
# Weights and Biases config
# -----------------------------------------------------------------------------
_C.wb = CN()
# The name of the project on W&B.
_C.wb.project_name = ""
# Logging entity (like your username or team name)
_C.wb.entity = ""
# The group ID to assign to the run. Optional to specify.
_C.wb.group = ""
# The run name to assign to the run. If not specified, W&B will randomly assign a name.
_C.wb.run_name = ""
# -----------------------------------------------------------------------------
# eval CONFIG
# -----------------------------------------------------------------------------
_C.eval = CN()
# The split to evaluate on
_C.eval.split = "val"
# Whether or not to use the config in the checkpoint. Setting this to False
# is useful if some code changes necessitate a new config but the weights
# are still valid.
_C.eval.use_ckpt_config = True
_C.eval.should_load_ckpt = True
# The number of time to run each episode through evaluation.
# Only works when evaluating on all episodes.
_C.eval.evals_per_ep = 1
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.rl = CN()
# -----------------------------------------------------------------------------
# preemption CONFIG
# -----------------------------------------------------------------------------
_C.rl.preemption = CN()
# Append the slurm job ID to the resume state filename if running a slurm job
# This is useful when you want to have things from a different job but same
# same checkpoint dir not resume.
_C.rl.preemption.append_slurm_job_id = False
# Number of gradient updates between saving the resume state
_C.rl.preemption.save_resume_state_interval = 100
# Save resume states only when running with slurm
# This is nice if you don't want debug jobs to resume
_C.rl.preemption.save_state_batch_only = False
# -----------------------------------------------------------------------------
# policy CONFIG
# -----------------------------------------------------------------------------
_C.rl.policy = CN()
_C.rl.policy.name = "PointNavResNetPolicy"
_C.rl.policy.action_distribution_type = "categorical"  # or 'gaussian'
# If the list is empty, all keys will be included.
# For gaussian action distribution:
_C.rl.policy.action_dist = CN()
_C.rl.policy.action_dist.use_log_std = True
_C.rl.policy.action_dist.use_softplus = False
_C.rl.policy.action_dist.log_std_init = 0.0
# If True, the std will be a parameter not conditioned on state
_C.rl.policy.action_dist.use_std_param = False
# If True, the std will be clamped to the specified min and max std values
_C.rl.policy.action_dist.clamp_std = True
_C.rl.policy.action_dist.min_std = 1e-6
_C.rl.policy.action_dist.max_std = 1
_C.rl.policy.action_dist.min_log_std = -5
_C.rl.policy.action_dist.max_log_std = 2
# For continuous action distributions (including gaussian):
_C.rl.policy.action_dist.action_activation = "tanh"  # ['tanh', '']
_C.rl.policy.action_dist.scheduled_std = False
# -----------------------------------------------------------------------------
# obs_transforms CONFIG
# -----------------------------------------------------------------------------
_C.rl.policy.obs_transforms = CN()
_C.rl.policy.obs_transforms.enabled_transforms = tuple()
_C.rl.policy.obs_transforms.center_cropper = CN()
_C.rl.policy.obs_transforms.center_cropper.height = 256
_C.rl.policy.obs_transforms.center_cropper.width = 256
_C.rl.policy.obs_transforms.center_cropper.channels_last = True
_C.rl.policy.obs_transforms.center_cropper.trans_keys = (
    "rgb",
    "depth",
    "semantic",
)
_C.rl.policy.obs_transforms.resize_shortest_edge = CN()
_C.rl.policy.obs_transforms.resize_shortest_edge.size = 256
_C.rl.policy.obs_transforms.resize_shortest_edge.channels_last = True
_C.rl.policy.obs_transforms.resize_shortest_edge.trans_keys = (
    "rgb",
    "depth",
    "semantic",
)
_C.rl.policy.obs_transforms.resize_shortest_edge.semantic_key = "semantic"
_C.rl.policy.obs_transforms.cube2eq = CN()
_C.rl.policy.obs_transforms.cube2eq.height = 256
_C.rl.policy.obs_transforms.cube2eq.width = 512
_C.rl.policy.obs_transforms.cube2eq.sensor_uuids = [
    "BACK",
    "DOWN",
    "FRONT",
    "LEFT",
    "RIGHT",
    "UP",
]
_C.rl.policy.obs_transforms.cube2fish = CN()
_C.rl.policy.obs_transforms.cube2fish.height = 256
_C.rl.policy.obs_transforms.cube2fish.width = 256
_C.rl.policy.obs_transforms.cube2fish.fov = 180
_C.rl.policy.obs_transforms.cube2fish.params = (0.2, 0.2, 0.2)
_C.rl.policy.obs_transforms.cube2fish.sensor_uuids = [
    "BACK",
    "DOWN",
    "FRONT",
    "LEFT",
    "RIGHT",
    "UP",
]
_C.rl.policy.obs_transforms.eq2cube = CN()
_C.rl.policy.obs_transforms.eq2cube.height = 256
_C.rl.policy.obs_transforms.eq2cube.width = 256
_C.rl.policy.obs_transforms.eq2cube.sensor_uuids = [
    "BACK",
    "DOWN",
    "FRONT",
    "LEFT",
    "RIGHT",
    "UP",
]
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.rl.ppo = CN()
_C.rl.ppo.clip_param = 0.2
_C.rl.ppo.ppo_epoch = 4
_C.rl.ppo.num_mini_batch = 2
_C.rl.ppo.value_loss_coef = 0.5
_C.rl.ppo.entropy_coef = 0.01
_C.rl.ppo.lr = 2.5e-4
_C.rl.ppo.eps = 1e-5
_C.rl.ppo.max_grad_norm = 0.5
_C.rl.ppo.num_steps = 5
_C.rl.ppo.use_gae = True
_C.rl.ppo.use_linear_lr_decay = False
_C.rl.ppo.use_linear_clip_decay = False
_C.rl.ppo.gamma = 0.99
_C.rl.ppo.tau = 0.95
_C.rl.ppo.reward_window_size = 50
_C.rl.ppo.use_normalized_advantage = False
_C.rl.ppo.hidden_size = 512
_C.rl.ppo.entropy_target_factor = 0.0
_C.rl.ppo.use_adaptive_entropy_pen = False
_C.rl.ppo.use_clipped_value_loss = True
# Use double buffered sampling, typically helps
# when environment time is similar or large than
# policy inference time during rollout generation
# Not that this does not change the memory requirements
_C.rl.ppo.use_double_buffered_sampler = False
# -----------------------------------------------------------------------------
# Variable Experience Rollout (VER)
# -----------------------------------------------------------------------------
_C.rl.ver = CN()
_C.rl.ver.variable_experience = True
_C.rl.ver.num_inference_workers = 2
_C.rl.ver.overlap_rollouts_and_learn = False
# -----------------------------------------------------------------------------
# Auxiliary Losses
# -----------------------------------------------------------------------------
_C.rl.auxiliary_losses = CN()
_C.rl.auxiliary_losses.enabled = []
# Action-Conditional Contrastive Predictive Coding Loss
_C.rl.auxiliary_losses.cpca = CN()
_C.rl.auxiliary_losses.cpca.k = 20
_C.rl.auxiliary_losses.cpca.time_subsample = 6
_C.rl.auxiliary_losses.cpca.future_subsample = 2
_C.rl.auxiliary_losses.cpca.loss_scale = 0.1
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.rl.ddppo = CN()
_C.rl.ddppo.sync_frac = 0.6
_C.rl.ddppo.distrib_backend = "GLOO"
_C.rl.ddppo.rnn_type = "GRU"
_C.rl.ddppo.num_recurrent_layers = 1
_C.rl.ddppo.backbone = "resnet18"
_C.rl.ddppo.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.rl.ddppo.pretrained = False
# Loads just the visual encoder backbone weights
_C.rl.ddppo.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.rl.ddppo.train_encoder = True
# Whether or not to reset the critic linear layer
_C.rl.ddppo.reset_critic = True
# Forces distributed mode for testing
_C.rl.ddppo.force_distributed = False
# -----------------------------------------------------------------------------
# orbslam2 BASELINE
# -----------------------------------------------------------------------------
_C.orbslam2 = CN()
_C.orbslam2.slam_vocab_path = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.orbslam2.slam_settings_path = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.orbslam2.map_cell_size = 0.1
_C.orbslam2.map_size = 40
_C.orbslam2.camera_height = (
    get_task_config().habitat.simulator.depth_sensor.position[1]
)
_C.orbslam2.beta = 100
_C.orbslam2.h_obstacle_min = 0.3 * _C.orbslam2.camera_height
_C.orbslam2.h_obstacle_max = 1.0 * _C.orbslam2.camera_height
_C.orbslam2.d_obstacle_min = 0.1
_C.orbslam2.d_obstacle_max = 4.0
_C.orbslam2.preprocess_map = True
_C.orbslam2.min_pts_in_obstacle = (
    get_task_config().habitat.simulator.depth_sensor.width / 2.0
)
_C.orbslam2.angle_th = float(np.deg2rad(15))
_C.orbslam2.dist_reached_th = 0.15
_C.orbslam2.next_waypoint_th = 0.5
_C.orbslam2.num_actions = 3
_C.orbslam2.dist_to_stop = 0.05
_C.orbslam2.planner_max_steps = 500
_C.orbslam2.depth_denorm = (
    get_task_config().habitat.simulator.depth_sensor.max_depth
)
# -----------------------------------------------------------------------------
# profiling
# -----------------------------------------------------------------------------
_C.profiling = CN()
_C.profiling.capture_start_step = -1
_C.profiling.num_steps_to_capture = -1


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
            if k == "base_task_config_path":
                config.base_task_config_path = v

    config.merge_from_other_cfg(get_task_config(config.base_task_config_path))

    # In case the config specifies overrides for the habitat config, we
    # remerge the files here
    if config_paths:
        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.cmd_trailing_opts = config.cmd_trailing_opts + opts
        config.merge_from_list(config.cmd_trailing_opts)

    if config.num_processes != -1:
        warnings.warn(
            "num_processes is depricated and will be removed in a future version."
            "  Use num_environments instead."
            "  Overwriting num_environments with num_processes for backwards compatibility."
        )

        config.num_environments = config.num_processes

    config.freeze()
    return config
