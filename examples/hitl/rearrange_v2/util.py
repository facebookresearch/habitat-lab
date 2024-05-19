#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from time import time

import magnum as mn

# TODO: Move outside of tutorial.
from habitat_hitl.environment.hitl_tutorial import (
    _lookat_bounding_box_top_down,
)

UP = mn.Vector3(0, 1, 0)
FWD = mn.Vector3(0, 0, 1)


def timestamp() -> str:
    "Generate a Unix timestamp at the current time."
    return str(int(time()))


def get_top_down_view(sim) -> mn.Matrix4:
    """
    Get a top-down view of the current scene.
    """
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_target_bb: mn.Range3D = scene_root_node.cumulative_bb
    look_at = _lookat_bounding_box_top_down(200, scene_target_bb, FWD)
    return mn.Matrix4.look_at(look_at[0], look_at[1], UP)


def get_empty_view(sim) -> mn.Matrix4:
    """
    Get a view looking into the void.
    Used to avoid displaying previously-loaded content in intermediate stages.
    """
    return mn.Matrix4.look_at(1000 * FWD, FWD, UP)
