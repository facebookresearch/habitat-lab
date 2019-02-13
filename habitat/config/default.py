#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from habitat.config import Config as CN
from habitat.core.logging import logger

CFG_DIR = "configs/"

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
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.SCENE = "data/habitat-sim/test/test.glb"
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
_C.DATASET.TYPE = "MP3DEQA-v1"
_C.DATASET.SPLIT = "val"
# -----------------------------------------------------------------------------
# MP3DEQAV1 DATASET
# -----------------------------------------------------------------------------
_C.DATASET.MP3DEQAV1 = CN()
_C.DATASET.MP3DEQAV1.DATA_PATH = "data/datasets/eqa/mp3d/v1/{split}.json.gz"
# -----------------------------------------------------------------------------
# POINTNAVV1 DATASET
# -----------------------------------------------------------------------------
_C.DATASET.POINTNAVV1 = CN()
_C.DATASET.POINTNAVV1.DATA_PATH = (
    "data/datasets/pointnav/gibson/v1/{" "split}/{split}.json.gz"
)
_C.DATASET.POINTNAVV1.CONTENT_SCENES = ["*"]
# -----------------------------------------------------------------------------
# BASELINES
# -----------------------------------------------------------------------------
_C.BASELINES = CN()
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------


def cfg(config_file: Optional[str] = None) -> CN:
    config = _C.clone()
    if config_file:
        config.merge_from_file("{}{}".format(CFG_DIR, config_file))
        logger.info(config)
    return config
