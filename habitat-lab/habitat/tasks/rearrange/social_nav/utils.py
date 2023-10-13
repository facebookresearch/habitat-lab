#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import magnum as mn
import numpy as np

from habitat.core.logging import HabitatLogger

rearrange_logger = HabitatLogger(
    name="rearrange_task",
    level=int(os.environ.get("HABITAT_REARRANGE_LOG", logging.ERROR)),
    format_str="[%(levelname)s,%(name)s] %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)


def robot_human_vec_dot_product(robot_pos, human_pos, base_T):
    """Compute the dot product between the human_robot vector and robot forward vector"""
    vector_human_robot = human_pos[[0, 2]] - robot_pos[[0, 2]]
    vector_human_robot = vector_human_robot / np.linalg.norm(
        vector_human_robot
    )
    forward_robot = np.array(base_T.transform_vector(mn.Vector3(1, 0, 0)))[
        [0, 2]
    ]
    forward_robot = forward_robot / np.linalg.norm(forward_robot)
    return np.dot(forward_robot, vector_human_robot)
