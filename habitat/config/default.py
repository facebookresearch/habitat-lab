#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config


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
_C.SEED = 100
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
_C.ENVIRONMENT.ITERATOR_OPTIONS = CN()
_C.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
_C.ENVIRONMENT.ITERATOR_OPTIONS.STEP_REPETITION_RANGE = 0.2
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # NAVIGATION TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = "Nav-v0"
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
_C.TASK.GOAL_SENSOR_UUID = "pointgoal"
_C.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
# -----------------------------------------------------------------------------
# # ACTIONS
# -----------------------------------------------------------------------------
ACTIONS = CN()
ACTIONS.STOP = CN()
ACTIONS.STOP.TYPE = "StopAction"
# -----------------------------------------------------------------------------
# # NAVIGATION ACTIONS
# -----------------------------------------------------------------------------
ACTIONS.MOVE_FORWARD = CN()
ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
ACTIONS.TURN_LEFT = CN()
ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
ACTIONS.TURN_RIGHT = CN()
ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
ACTIONS.LOOK_UP = CN()
ACTIONS.LOOK_UP.TYPE = "LookUpAction"
ACTIONS.LOOK_DOWN = CN()
ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"
ACTIONS.TELEPORT = CN()
ACTIONS.TELEPORT.TYPE = "TeleportAction"
ACTIONS.VELOCITY_CONTROL = CN()
ACTIONS.VELOCITY_CONTROL.TYPE = "VelocityAction"
ACTIONS.VELOCITY_CONTROL.LIN_VEL_RANGE = [0.0, 0.25]  # meters per sec
ACTIONS.VELOCITY_CONTROL.ANG_VEL_RANGE = [-10.0, 10.0]  # deg per sec
ACTIONS.VELOCITY_CONTROL.MIN_ABS_LIN_SPEED = 0.025  # meters per sec
ACTIONS.VELOCITY_CONTROL.MIN_ABS_ANG_SPEED = 1.0  # deg per sec
ACTIONS.VELOCITY_CONTROL.TIME_STEP = 1.0  # seconds

_C.TASK.ACTIONS = ACTIONS
# -----------------------------------------------------------------------------
# # TASK SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECTGOAL_SENSOR = CN()
_C.TASK.OBJECTGOAL_SENSOR.TYPE = "ObjectGoalSensor"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC = "TASK_CATEGORY_ID"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC_MAX_VAL = 50
# -----------------------------------------------------------------------------
# IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.IMAGEGOAL_SENSOR = CN()
_C.TASK.IMAGEGOAL_SENSOR.TYPE = "ImageGoalSensor"
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COMPASS_SENSOR = CN()
_C.TASK.COMPASS_SENSOR.TYPE = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_SENSOR = CN()
_C.TASK.GPS_SENSOR.TYPE = "GPSSensor"
_C.TASK.GPS_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUCCESS = CN()
_C.TASK.SUCCESS.TYPE = "Success"
_C.TASK.SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
# -----------------------------------------------------------------------------
# SOFT-SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SOFT_SPL = CN()
_C.TASK.SOFT_SPL.TYPE = "SoftSPL"
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1024
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True
# -----------------------------------------------------------------------------
# COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # EQA TASK
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.ANSWER = CN()
_C.TASK.ACTIONS.ANSWER.TYPE = "AnswerAction"
# # EQA TASK QUESTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.QUESTION_SENSOR = CN()
_C.TASK.QUESTION_SENSOR.TYPE = "QuestionSensor"
# -----------------------------------------------------------------------------
# # EQA TASK CORRECT_ANSWER measure for training
# -----------------------------------------------------------------------------
_C.TASK.CORRECT_ANSWER = CN()
_C.TASK.CORRECT_ANSWER.TYPE = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # EQA TASK ANSWER SENSOR
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_INFO = CN()
_C.TASK.EPISODE_INFO.TYPE = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # VLN TASK INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.INSTRUCTION_SENSOR = CN()
_C.TASK.INSTRUCTION_SENSOR.TYPE = "InstructionSensor"
_C.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_GOAL = CN()
_C.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_C.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # ANSWER_ACCURACY MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ANSWER_ACCURACY = CN()
_C.TASK.ANSWER_ACCURACY.TYPE = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.SIMULATOR.SCENE_DATASET = "default"  # the scene dataset to load in the MetaDataMediator. Should contain SIMULATOR.SCENE
_C.SIMULATOR.ADDITIONAL_OBJECT_PATHS = (
    []
)  # a list of directory or config paths to search in addition to the dataset for object configs
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
# -----------------------------------------------------------------------------
# SIMULATOR SENSORS
# -----------------------------------------------------------------------------
SIMULATOR_SENSOR = CN()
SIMULATOR_SENSOR.HEIGHT = 480
SIMULATOR_SENSOR.WIDTH = 640
SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
SIMULATOR_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles

# -----------------------------------------------------------------------------
# CAMERA SENSOR
# -----------------------------------------------------------------------------
CAMERA_SIM_SENSOR = SIMULATOR_SENSOR.clone()
CAMERA_SIM_SENSOR.HFOV = 90  # horizontal field of view in degrees
CAMERA_SIM_SENSOR.SENSOR_SUBTYPE = "PINHOLE"

SIMULATOR_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
SIMULATOR_DEPTH_SENSOR.MIN_DEPTH = 0.0
SIMULATOR_DEPTH_SENSOR.MAX_DEPTH = 10.0
SIMULATOR_DEPTH_SENSOR.NORMALIZE_DEPTH = True

# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# EQUIRECT RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_RGB_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_RGB_SENSOR.TYPE = "HabitatSimEquirectangularRGBSensor"
# -----------------------------------------------------------------------------
# EQUIRECT DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR.TYPE = (
    "HabitatSimEquirectangularDepthSensor"
)
# -----------------------------------------------------------------------------
# EQUIRECT SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_SEMANTIC_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_SEMANTIC_SENSOR.TYPE = (
    "HabitatSimEquirectangularSemanticSensor"
)
# -----------------------------------------------------------------------------
# FISHEYE SENSOR
# -----------------------------------------------------------------------------
FISHEYE_SIM_SENSOR = SIMULATOR_SENSOR.clone()
FISHEYE_SIM_SENSOR.HEIGHT = FISHEYE_SIM_SENSOR.WIDTH
# -----------------------------------------------------------------------------
# ROBOT HEAD RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.HEAD_RGB_SENSOR.UUID = "robot_head_rgb"
# -----------------------------------------------------------------------------
# ROBOT HEAD DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.HEAD_DEPTH_SENSOR.UUID = "robot_head_depth"
# -----------------------------------------------------------------------------
# ARM RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.ARM_RGB_SENSOR.UUID = "robot_arm_rgb"
# -----------------------------------------------------------------------------
# ARM DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.ARM_DEPTH_SENSOR.UUID = "robot_arm_depth"
# -----------------------------------------------------------------------------
# 3rd RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.THIRD_RGB_SENSOR.UUID = "robot_third_rgb"
# -----------------------------------------------------------------------------
# 3rd DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.THIRD_DEPTH_SENSOR.UUID = "robot_third_rgb"

# The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
# Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
# Camera Model, The International Conference on 3D Vision (3DV), 2018
# You can find the intrinsic parameters for the other lenses in the same table as well.
FISHEYE_SIM_SENSOR.XI = -0.27
FISHEYE_SIM_SENSOR.ALPHA = 0.57
FISHEYE_SIM_SENSOR.FOCAL_LENGTH = [364.84, 364.86]
# Place camera at center of screen
# Can be specified, otherwise is calculated automatically.
FISHEYE_SIM_SENSOR.PRINCIPAL_POINT_OFFSET = None  # (defaults to (h/2,w/2))
FISHEYE_SIM_SENSOR.SENSOR_MODEL_TYPE = "DOUBLE_SPHERE"
# -----------------------------------------------------------------------------
# FISHEYE RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_RGB_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_RGB_SENSOR.TYPE = "HabitatSimFisheyeRGBSensor"
# -----------------------------------------------------------------------------
# FISHEYE DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR.TYPE = "HabitatSimFisheyeDepthSensor"
# -----------------------------------------------------------------------------
# FISHEYE SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_SEMANTIC_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_SEMANTIC_SENSOR.TYPE = "HabitatSimFisheyeSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
# Whether or not the agent slides on collisions
_C.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
_C.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = False
_C.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = (
    "./data/default.physics_config.json"
)
# Possibly unstable optimization for extra performance with concurrent rendering
_C.SIMULATOR.HABITAT_SIM_V0.LEAVE_CONTEXT_WITH_BACKGROUND_RENDERER = False
# -----------------------------------------------------------------------------
# PYROBOT
# -----------------------------------------------------------------------------
_C.PYROBOT = CN()
_C.PYROBOT.ROBOTS = ["locobot"]  # types of robots supported
_C.PYROBOT.ROBOT = "locobot"
_C.PYROBOT.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "BUMP_SENSOR"]
_C.PYROBOT.BASE_CONTROLLER = "proportional"
_C.PYROBOT.BASE_PLANNER = "none"
# -----------------------------------------------------------------------------
# SENSORS
# -----------------------------------------------------------------------------
PYROBOT_VISUAL_SENSOR = CN()
PYROBOT_VISUAL_SENSOR.HEIGHT = 480
PYROBOT_VISUAL_SENSOR.WIDTH = 640
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.RGB_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.RGB_SENSOR.TYPE = "PyRobotRGBSensor"
_C.PYROBOT.RGB_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.DEPTH_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.DEPTH_SENSOR.TYPE = "PyRobotDepthSensor"
_C.PYROBOT.DEPTH_SENSOR.MIN_DEPTH = 0.0
_C.PYROBOT.DEPTH_SENSOR.MAX_DEPTH = 5.0
_C.PYROBOT.DEPTH_SENSOR.NORMALIZE_DEPTH = True
_C.PYROBOT.DEPTH_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# BUMP SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.BUMP_SENSOR = CN()
_C.PYROBOT.BUMP_SENSOR.TYPE = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# ACTIONS LOCOBOT
# -----------------------------------------------------------------------------
_C.PYROBOT.LOCOBOT = CN()
_C.PYROBOT.LOCOBOT.ACTIONS = ["BASE_ACTIONS", "CAMERA_ACTIONS"]
_C.PYROBOT.LOCOBOT.BASE_ACTIONS = ["go_to_relative", "go_to_absolute"]
_C.PYROBOT.LOCOBOT.CAMERA_ACTIONS = ["set_pan", "set_tilt", "set_pan_tilt"]
# TODO(akadian): add support for Arm actions
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "PointNav-v1"
_C.DATASET.SPLIT = "train"
_C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.CONTENT_SCENES = ["*"]
_C.DATASET.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# DEPRECATED KEYS
# -----------------------------------------------------------------------------'
_C.register_deprecated_key("TASK.SUCCESS_DISTANCE")
# -----------------------------------------------------------------------------


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
