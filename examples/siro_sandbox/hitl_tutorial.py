#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import ctypes
import math

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys
from math import radians

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

from functools import wraps
from typing import Any, List, Tuple

import magnum as mn

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim


class TutorialStage:
    _prev_lookat: Tuple[mn.Vector3, mn.Vector3]
    _next_lookat: Tuple[mn.Vector3, mn.Vector3]
    _transition_duration: float
    _stage_duration: float
    _elapsed_time: float

    def __init__(
        self,
        stage_duration: float,
        next_lookat: Tuple[mn.Vector3, mn.Vector3],
        prev_lookat: Tuple[mn.Vector3, mn.Vector3] = None,
        transition_duration: float = 0.0,
    ) -> None:
        self._transition_duration = transition_duration
        self._stage_duration = stage_duration
        self._prev_lookat = prev_lookat
        self._next_lookat = next_lookat
        self._elapsed_time = 0.0

    def update(self, dt: float) -> None:
        self._elapsed_time += dt

    def get_look_at_matrix(self) -> mn.Matrix4:
        # If there's no transition, return the next view
        assert self._next_lookat
        if not self._prev_lookat or self._transition_duration <= 0.0:
            return mn.Matrix4.look_at(
                self._next_lookat[0], self._next_lookat[1], mn.Vector3(0, 1, 0)
            )
        # Interpolate camera look-ats
        t: float = (
            _ease_fn_in_out_quat(
                min(
                    self._elapsed_time / self._transition_duration,
                    1.0,
                )
            )
            if self._transition_duration > 0.0
            else 1.0
        )
        look_at: List[mn.Vector3] = []
        for i in range(2):  # Only interpolate eye and target vectors
            look_at.append(
                mn.math.lerp(
                    self._prev_lookat[i],
                    self._next_lookat[i],
                    t,
                )
            )
        return mn.Matrix4.look_at(look_at[0], look_at[1], mn.Vector3(0, 1, 0))

    def is_stage_completed(self) -> bool:
        return self._elapsed_time >= self._stage_duration


def generate_tutorial(sim, agent_idx, final_lookat) -> List[TutorialStage]:
    assert sim is not None
    assert agent_idx is not None
    assert final_lookat is not None
    camera_fov_deg = 90  # TODO: Get the actual FOV
    tutorial_stages: List[TutorialStage] = []

    # Scene overview
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_target_bb: mn.Range3D = scene_root_node.cumulative_bb
    scene_top_down_lookat = _lookat_bounding_box_top_down(
        camera_fov_deg, scene_target_bb
    )
    tutorial_stages.append(
        TutorialStage(
            stage_duration=5.0, next_lookat=scene_top_down_lookat
        )
    )

    # Show all the targets
    pathfinder = sim.pathfinder
    idxs, goal_pos = sim.get_targets()
    scene_positions = sim.get_scene_pos()
    target_positions = scene_positions[idxs]
    for target_pos in target_positions:
        next_lookat = _lookat_point_from_closest_navmesh_pos(
            mn.Vector3(target_pos), 0.75, 1.5, pathfinder
        )
        tutorial_stages.append(
            TutorialStage(
                stage_duration=3.0,
                transition_duration=2.0,
                prev_lookat=tutorial_stages[
                    len(tutorial_stages) - 1
                ]._next_lookat,
                next_lookat=next_lookat,
            )
        )

    # Controlled agent focus
    art_obj = (
        sim.agents_mgr[agent_idx].articulated_agent.sim_obj
    )
    agent_root_node = art_obj.get_link_scene_node(
        -1
    )  # Root link always has index -1
    target_bb = mn.Range3D.from_center(
        mn.Vector3(agent_root_node.absolute_translation),
        mn.Vector3(1.0, 1.0, 1.0),
    )  # Assume 2x2x2m bounding box
    agent_lookat = _lookat_bounding_box_top_down(camera_fov_deg, target_bb)

    tutorial_stages.append(
        TutorialStage(
            stage_duration=2.0,
            transition_duration=1.0,
            prev_lookat=tutorial_stages[
                len(tutorial_stages) - 1
            ]._next_lookat,
            next_lookat=agent_lookat,
        )
    )

    # Transition from the avatar view to simulated view (e.g. first person)
    tutorial_stages.append(
        TutorialStage(
            stage_duration=1.0,
            transition_duration=1.0,
            prev_lookat=tutorial_stages[
                len(tutorial_stages) - 1
            ]._next_lookat,
            next_lookat=final_lookat,
        )
    )

    return tutorial_stages


def _ease_fn_in_out_quat(t: float):
    if t < 0.5:
        return 16 * t * pow(t, 4)
    else:
        return 1 - pow(-2 * t + 2, 4) / 2


def _lookat_bounding_box_top_down(
    camera_fov_deg: float, target_bb: mn.Range3D
) -> Tuple[mn.Vector3, mn.Vector3]:
    r"""
    Creates look-at vectors for a top-down camera such as the entire 'target_bb' bounding box is visible.
    """
    camera_fov_rad = radians(camera_fov_deg)
    target_dimension = max(target_bb.size_x(), target_bb.size_z())
    camera_position = mn.Vector3(
        target_bb.center_x(),
        target_bb.center_y()
        + abs(target_dimension / math.sin(camera_fov_rad / 2)),
        target_bb.center_z(),
    )
    epsilon = 0.0001  # Because of gimbal lock, we apply an epsilon bias to force a camera orientation
    return (
        camera_position,
        target_bb.center() + mn.Vector3(0.0, 0.0, epsilon),
    )


def _lookat_point_from_closest_navmesh_pos(
    target: mn.Vector3,
    distance_from_target: float,
    viewpoint_height: float,
    pathfinder: habitat_sim.PathFinder,
) -> Tuple[mn.Vector3, mn.Vector3]:
    r"""
    Creates look-at vectors for a viewing a point from a nearby navigable point (with a height offset).
    Helps finding a viewpoint that is not occluded by a wall or other obstacle.
    """
    # Look up for a camera position by sampling radially around the target for a navigable position.
    navigable_point = None
    sample_count = 8
    for i in range(sample_count):
        radial_angle = i * 2.0 * math.pi / float(sample_count)
        dist_x = math.sin(radial_angle) * distance_from_target
        dist_z = math.cos(radial_angle) * distance_from_target
        candidate = mn.Vector3(target.x + dist_x, target.y, target.z + dist_z)
        if pathfinder.is_navigable(candidate, 3.0):
            navigable_point = candidate

    if navigable_point == None:
        # Fallback to a default point
        navigable_point = mn.Vector3(
            target.x + distance_from_target, target.y, target.z
        )

    camera_position = mn.Vector3(
        navigable_point.x,
        navigable_point.y + viewpoint_height,
        navigable_point.z,
    )
    return (camera_position, target)