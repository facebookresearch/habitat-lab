#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

__version__ = "0.1.0"

try:
    __HABITAT_SETUP__  # type: ignore
except NameError:
    __HABITAT_SETUP__ = False

if __HABITAT_SETUP__:
    sys.stderr.write("Partial import of habitat during the install process.\n")
else:
    from habitat.core.dataset import Dataset
    from habitat.core.embodied_task import EmbodiedTask, Measure, Measurements
    from habitat.core.simulator import (
        SensorTypes,
        Sensor,
        SensorSuite,
        Simulator,
    )
    from habitat.core.env import Env, RLEnv
    from habitat.core.vector_env import VectorEnv, ThreadedVectorEnv
    from habitat.core.logging import logger
    from habitat.config import Config
    from habitat.datasets import make_dataset

    __all__ = [
        "Dataset",
        "EmbodiedTask",
        "Measure",
        "Measurements",
        "Env",
        "RLEnv",
        "Simulator",
        "Sensor",
        "logger",
        "SensorTypes",
        "SensorSuite",
        "VectorEnv",
        "ThreadedVectorEnv",
        "make_dataset",
        "Config",
    ]
