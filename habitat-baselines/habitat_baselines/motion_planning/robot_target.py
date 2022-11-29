#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import attr
import magnum as mn
import numpy as np


@attr.s(auto_attribs=True, slots=True)
class RobotTarget:
    """
    Data class to define the target needed as input for the motion planner.
    """

    # End-effector in world coordinate frame.
    ee_target_pos: np.ndarray = None
    obj_id_target: int = None
    joints_target: np.ndarray = None
    is_guess: bool = False


@attr.s(auto_attribs=True, slots=True)
class ObjectGraspTarget:
    """
    Data class to define the target needed as input for the grasp planner.
    """

    # Bounding Box
    bb: mn.Range3D
    transformation: mn.Matrix4
