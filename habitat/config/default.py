#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

from habitat.config import Config as CN  # type: ignore

DEFAULT_CONFIG_DIR = "configs/"

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
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = "Nav-v0"
_C.TASK.SUCCESS_DISTANCE = 0.2
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
# -----------------------------------------------------------------------------
# # POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
# -----------------------------------------------------------------------------
# # HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# # PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# # SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
_C.TASK.SPL.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# # TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
# -----------------------------------------------------------------------------
# # COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/" "van-gogh-room.glb"
)
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
# -----------------------------------------------------------------------------
# # SENSORS
# -----------------------------------------------------------------------------
SENSOR = CN()
SENSOR.HEIGHT = 480
SENSOR.WIDTH = 640
SENSOR.HFOV = 90  # horizontal field of view in degrees
SENSOR.POSITION = [0, 1.25, 0]
# -----------------------------------------------------------------------------
# # RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
_C.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0
_C.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10
_C.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.MASS = 32.0
_C.SIMULATOR.AGENT_0.LINEAR_ACCELERATION = 20.0
_C.SIMULATOR.AGENT_0.ANGULAR_ACCELERATION = 4 * 3.14
_C.SIMULATOR.AGENT_0.LINEAR_FRICTION = 0.5
_C.SIMULATOR.AGENT_0.ANGULAR_FRICTION = 1.0
_C.SIMULATOR.AGENT_0.COEFFICIENT_OF_RESTITUTION = 0.0
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
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "PointNav-v1"
_C.DATASET.SPLIT = "train"
# -----------------------------------------------------------------------------
# MP3DEQAV1 DATASET
# -----------------------------------------------------------------------------
_C.DATASET.MP3DEQAV1 = CN()
_C.DATASET.MP3DEQAV1.DATA_PATH = (
    "data/datasets/eqa/mp3d/v1/{split}/{split}.json.gz"
)
# -----------------------------------------------------------------------------
# POINTNAVV1 DATASET
# -----------------------------------------------------------------------------
_C.DATASET.POINTNAVV1 = CN()
_C.DATASET.POINTNAVV1.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)
_C.DATASET.POINTNAVV1.CONTENT_SCENES = ["*"]


# -----------------------------------------------------------------------------


def get_config(
    config_file: Optional[str] = None, config_dir: str = DEFAULT_CONFIG_DIR
) -> CN:
    config = _C.clone()
    if config_file:
        config.merge_from_file(os.path.join(config_dir, config_file))
    config.freeze()
    return config
