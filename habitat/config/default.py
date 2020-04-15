#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config

# from habitat.config import Config as CN # type: ignore

# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.habitat = CN()
_C.habitat.seed = 100
# -----------------------------------------------------------------------------
# environment
# -----------------------------------------------------------------------------
_C.habitat.environment = CN()
_C.habitat.environment.max_episode_steps = 1000
_C.habitat.environment.max_episode_seconds = 10000000
_C.habitat.environment.iterator_options = CN()
_C.habitat.environment.iterator_options.cycle = True
_C.habitat.environment.iterator_options.shuffle = True
_C.habitat.environment.iterator_options.group_by_scene = True
_C.habitat.environment.iterator_options.num_episode_sample = -1
_C.habitat.environment.iterator_options.max_scene_repeat_episodes = -1
_C.habitat.environment.iterator_options.max_scene_repeat_steps = int(1e4)
_C.habitat.environment.iterator_options.step_repetition_range = 0.2
# -----------------------------------------------------------------------------
# task
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # NAVIGATION task
# -----------------------------------------------------------------------------
_C.habitat.task = CN()
_C.habitat.task.type = "Nav-v0"
_C.habitat.task.sensors = []
_C.habitat.task.measurements = []
_C.habitat.task.goal_sensor_uuid = "pointgoal"
_C.habitat.task.possible_actions = [
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
]
# -----------------------------------------------------------------------------
# # actions
# -----------------------------------------------------------------------------
actions = CN()
actions.stop = CN()
actions.stop.type = "StopAction"
# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------
actions.move_forward = CN()
actions.move_forward.type = "MoveForwardAction"
actions.turn_left = CN()
actions.turn_left.type = "TurnLeftAction"
actions.turn_right = CN()
actions.turn_right.type = "TurnRightAction"
actions.look_up = CN()
actions.look_up.type = "LookUpAction"
actions.look_down = CN()
actions.look_down.type = "LookDownAction"
actions.teleport = CN()
actions.teleport.type = "TeleportAction"

_C.habitat.task.actions = actions
# -----------------------------------------------------------------------------
# # task sensors
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.pointgoal_sensor = CN()
_C.habitat.task.pointgoal_sensor.type = "PointGoalSensor"
_C.habitat.task.pointgoal_sensor.goal_format = "POLAR"
_C.habitat.task.pointgoal_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.pointgoal_with_gps_compass_sensor = (
    _C.habitat.task.pointgoal_sensor.clone()
)
_C.habitat.task.pointgoal_with_gps_compass_sensor.type = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.objectgoal_sensor = CN()
_C.habitat.task.objectgoal_sensor.type = "ObjectGoalSensor"
_C.habitat.task.objectgoal_sensor.goal_spec = "TASK_CATEGORY_ID"
_C.habitat.task.objectgoal_sensor.goal_spec_max_val = 50
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.heading_sensor = CN()
_C.habitat.task.heading_sensor.type = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.compass_sensor = CN()
_C.habitat.task.compass_sensor.type = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.gps_sensor = CN()
_C.habitat.task.gps_sensor.type = "GPSSensor"
_C.habitat.task.gps_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.proximity_sensor = CN()
_C.habitat.task.proximity_sensor.type = "ProximitySensor"
_C.habitat.task.proximity_sensor.max_detection_radius = 2.0
# -----------------------------------------------------------------------------
# success MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.success = CN()
_C.habitat.task.success.type = "Success"
_C.habitat.task.success.success_distance = 0.2
# -----------------------------------------------------------------------------
# spl MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.spl = CN()
_C.habitat.task.spl.type = "SPL"
# -----------------------------------------------------------------------------
# SOFT-SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.soft_spl = CN()
_C.habitat.task.soft_spl.type = "SoftSPL"
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.top_down_map = CN()
_C.habitat.task.top_down_map.type = "TopDownMap"
_C.habitat.task.top_down_map.max_episode_steps = (
    _C.habitat.environment.max_episode_steps
)
_C.habitat.task.top_down_map.map_padding = 3
_C.habitat.task.top_down_map.num_topdown_map_sample_points = 20000
_C.habitat.task.top_down_map.map_resolution = 1250
_C.habitat.task.top_down_map.draw_source = True
_C.habitat.task.top_down_map.draw_border = True
_C.habitat.task.top_down_map.draw_shortest_path = True
_C.habitat.task.top_down_map.fog_of_war = CN()
_C.habitat.task.top_down_map.fog_of_war.draw = True
_C.habitat.task.top_down_map.fog_of_war.visibility_dist = 5.0
_C.habitat.task.top_down_map.fog_of_war.fov = 90
_C.habitat.task.top_down_map.draw_view_points = True
_C.habitat.task.top_down_map.draw_goal_positions = True
# Axes aligned bounding boxes
_C.habitat.task.top_down_map.draw_goal_aabbs = True
# -----------------------------------------------------------------------------
# collisions MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.collisions = CN()
_C.habitat.task.collisions.type = "Collisions"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # EQA task
# -----------------------------------------------------------------------------
_C.habitat.task.actions.answer = CN()
_C.habitat.task.actions.answer.type = "AnswerAction"
# # EQA task QUESTION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.question_sensor = CN()
_C.habitat.task.question_sensor.type = "QuestionSensor"
# -----------------------------------------------------------------------------
# # EQA task correct_answer measure for training
# -----------------------------------------------------------------------------
_C.habitat.task.correct_answer = CN()
_C.habitat.task.correct_answer.type = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # EQA task answer SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.episode_info = CN()
_C.habitat.task.episode_info.type = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # VLN task INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.instruction_sensor = CN()
_C.habitat.task.instruction_sensor.type = "InstructionSensor"
_C.habitat.task.instruction_sensor_uuid = "instruction"
# -----------------------------------------------------------------------------
# # distance_to_goal MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.distance_to_goal = CN()
_C.habitat.task.distance_to_goal.type = "DistanceToGoal"
_C.habitat.task.distance_to_goal.distance_to = "POINT"
# -----------------------------------------------------------------------------
# # answer_accuracy MEASUREMENT
# -----------------------------------------------------------------------------
_C.habitat.task.answer_accuracy = CN()
_C.habitat.task.answer_accuracy.type = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# simulator
# -----------------------------------------------------------------------------
_C.habitat.simulator = CN()
_C.habitat.simulator.type = "Sim-v0"
_C.habitat.simulator.action_space_config = "v0"
_C.habitat.simulator.forward_step_size = 0.25  # in metres
_C.habitat.simulator.scene = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.habitat.simulator.seed = _C.habitat.seed
_C.habitat.simulator.turn_angle = (
    10  # angle to rotate left or right in degrees
)
_C.habitat.simulator.tilt_angle = (
    15  # angle to tilt the camera up or down in degrees
)
_C.habitat.simulator.default_agent_id = 0
# -----------------------------------------------------------------------------
# simulator sensors
# -----------------------------------------------------------------------------
SIMULATOR_SENSOR = CN()
SIMULATOR_SENSOR.height = 480
SIMULATOR_SENSOR.width = 640
SIMULATOR_SENSOR.hfov = 90  # horizontal field of view in degrees
SIMULATOR_SENSOR.position = [0, 1.25, 0]
SIMULATOR_SENSOR.orientation = [0.0, 0.0, 0.0]  # Euler's angles
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.rgb_sensor = SIMULATOR_SENSOR.clone()
_C.habitat.simulator.rgb_sensor.type = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.depth_sensor = SIMULATOR_SENSOR.clone()
_C.habitat.simulator.depth_sensor.type = "HabitatSimDepthSensor"
_C.habitat.simulator.depth_sensor.min_depth = 0.0
_C.habitat.simulator.depth_sensor.max_depth = 10.0
_C.habitat.simulator.depth_sensor.normalize_depth = True
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.habitat.simulator.semantic_sensor = SIMULATOR_SENSOR.clone()
_C.habitat.simulator.semantic_sensor.type = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.habitat.simulator.agent_0 = CN()
_C.habitat.simulator.agent_0.height = 1.5
_C.habitat.simulator.agent_0.radius = 0.1
_C.habitat.simulator.agent_0.mass = 32.0
_C.habitat.simulator.agent_0.linear_acceleration = 20.0
_C.habitat.simulator.agent_0.angular_acceleration = 4 * 3.14
_C.habitat.simulator.agent_0.linear_friction = 0.5
_C.habitat.simulator.agent_0.angular_friction = 1.0
_C.habitat.simulator.agent_0.coefficient_of_restitution = 0.0
_C.habitat.simulator.agent_0.sensors = ["rgb_sensor"]
_C.habitat.simulator.agent_0.is_set_start_state = False
_C.habitat.simulator.agent_0.start_position = [0, 0, 0]
_C.habitat.simulator.agent_0.start_rotation = [0, 0, 0, 1]
_C.habitat.simulator.agents = ["agent_0"]

_C.habitat.simulator.noise_model = CN()
_C.habitat.simulator.noise_model.robot = "LoCoBot"
_C.habitat.simulator.noise_model.controller = "Proportional"
_C.habitat.simulator.noise_model.noise_multiplier = 0.5
# -----------------------------------------------------------------------------
# simulator habitat_sim_v0
# -----------------------------------------------------------------------------
_C.habitat.simulator.habitat_sim_v0 = CN()
_C.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.habitat.simulator.habitat_sim_v0.gpu_gpu = False
# Whether or not the agent slides on collisions
_C.habitat.simulator.habitat_sim_v0.allow_sliding = True
_C.habitat.simulator.habitat_sim_v0.enable_physics = False
_C.habitat.simulator.habitat_sim_v0.physics_config_file = (
    "./data/default.phys_scene_config.json"
)
# -----------------------------------------------------------------------------
# pyrobot
# -----------------------------------------------------------------------------
_C.habitat.pyrobot = CN()
_C.habitat.pyrobot.robots = ["locobot"]  # types of robots supported
_C.habitat.pyrobot.robot = "locobot"
_C.habitat.pyrobot.sensors = ["rgb_sensor", "depth_sensor", "bump_sensor"]
_C.habitat.pyrobot.base_controller = "proportional"
_C.habitat.pyrobot.base_planner = "none"
# -----------------------------------------------------------------------------
# sensors
# -----------------------------------------------------------------------------
PYROBOT_VISUAL_SENSOR = CN()
PYROBOT_VISUAL_SENSOR.height = 480
PYROBOT_VISUAL_SENSOR.width = 640
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.rgb_sensor = PYROBOT_VISUAL_SENSOR.clone()
_C.habitat.pyrobot.rgb_sensor.type = "PyRobotRGBSensor"
_C.habitat.pyrobot.rgb_sensor.center_crop = False
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.depth_sensor = PYROBOT_VISUAL_SENSOR.clone()
_C.habitat.pyrobot.depth_sensor.type = "PyRobotDepthSensor"
_C.habitat.pyrobot.depth_sensor.min_depth = 0.0
_C.habitat.pyrobot.depth_sensor.max_depth = 5.0
_C.habitat.pyrobot.depth_sensor.normalize_depth = True
_C.habitat.pyrobot.depth_sensor.center_crop = False
# -----------------------------------------------------------------------------
# BUMP SENSOR
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.bump_sensor = CN()
_C.habitat.pyrobot.bump_sensor.type = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# actions locobot
# -----------------------------------------------------------------------------
_C.habitat.pyrobot.locobot = CN()
_C.habitat.pyrobot.locobot.actions = ["base_actions", "camera_actions"]
_C.habitat.pyrobot.locobot.base_actions = ["go_to_relative", "go_to_absolute"]
_C.habitat.pyrobot.locobot.camera_actions = [
    "set_pan",
    "set_tilt",
    "set_pan_tilt",
]
# TODO(akadian): add support for Arm actions
# -----------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------
_C.habitat.dataset = CN()
_C.habitat.dataset.type = "PointNav-v1"
_C.habitat.dataset.split = "train"
_C.habitat.dataset.scenes_dir = "data/scene_datasets"
_C.habitat.dataset.content_scenes = ["*"]
_C.habitat.dataset.data_path = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

# -----------------------------------------------------------------------------


def extend_default_config(prefix: str, new_node: Config):
    r"""Extended the default config with a new node

    :param prefix: The prefix of the new node in the config, i.e. to add
                   add a new config node under habitat.dataset.episode,
                   you would provide that!
    :param new_node: The new config node to add
    """
    prefix = prefix.split(".")
    node = _C
    for name in prefix[:-1]:
        node = getattr(node, name)

    setattr(node, prefix[-1], new_node)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> Config:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
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
