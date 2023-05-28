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

from typing import List, Tuple

import magnum as mn

import habitat_sim


class ObjectAnimation:
    _object: habitat_sim.physics.ManagedBulletRigidObject
    _view_lookat: Tuple[mn.Vector3, mn.Vector3]
    _object_starting_pos: mn.Vector3
    _object_starting_rot: mn.Quaternion
    _object_target_pos: mn.Vector3
    _object_target_rot: mn.Quaternion
    _duration: float
    _place_back_time: float
    _rotation_animation_speed: float
    _rotation_animation_axis: mn.Vector3 = mn.Vector3(0, 1, -0.5).normalized()
    _elapsed_time: float = 0.0
    _rotation_offset: mn.Quaternion = mn.Quaternion.identity_init()

    def __init__(
        self,
        obj: habitat_sim.physics.ManagedBulletRigidObject,
        view_lookat: Tuple[mn.Vector3, mn.Vector3],
        distance_from_view: float,
        duration: float,
        place_back_time: float = 0.3,
        rotation_animation_speed: float = 25.0,
    ) -> None:
        self._object = obj
        self._view_lookat = view_lookat
        self._duration = duration
        self._place_back_time = place_back_time
        self._rotation_animation_speed = rotation_animation_speed
        self._object_starting_pos = obj.translation
        self._object_starting_rot = obj.rotation

        self._object_target_pos = view_lookat[0] - mn.Vector3(
            0.0, distance_from_view, 0.0
        )

        object_lookat = mn.Matrix4.look_at(
            self._object_target_pos,
            view_lookat[0],
            mn.Vector3(-1, 0, 0),  # Object forward axis
        )
        self._object_target_rot = mn.Quaternion.from_matrix(
            object_lookat.rotation()
        )

    def update(self, dt: float):
        # Slowly rotate the object to give more perspective
        self._rotation_offset *= mn.Quaternion.rotation(
            mn.Rad(mn.Deg(dt * self._rotation_animation_speed)),
            self._rotation_animation_axis,
        )

        self._elapsed_time += dt
        t = self._elapsed_time / self._duration
        if t <= 0.0:
            self.reset()
            return
        elif t >= 1.0:
            self._place_back(dt)
            return

        t = _ease_fn_in_out_quat(t)

        # Interpolate
        pos = mn.math.lerp(
            self._object_starting_pos,
            self._object_target_pos,
            t,
        )
        rot = mn.math.slerp(
            self._object_starting_rot,
            self._object_target_rot * self._rotation_offset,
            t,
        )

        self._object.translation = pos
        self._object.rotation = rot

    def _place_back(self, dt: float):
        t = (self._elapsed_time - self._duration) / (
            self._duration + self._place_back_time
        )
        if t > 1.0:
            self.reset()
            return
        t = _ease_fn_in_out_quat(t)

        # Interpolate
        pos = mn.math.lerp(
            self._object_target_pos,
            self._object_starting_pos,
            t,
        )
        rot = mn.math.slerp(
            self._object_target_rot * self._rotation_offset,
            self._object_starting_rot,
            t,
        )

        self._object.translation = pos
        self._object.rotation = rot

    def reset(self):
        self._object.translation = self._object_starting_pos
        self._object.rotation = self._object_starting_rot


class TutorialStage:
    _prev_lookat: Tuple[mn.Vector3, mn.Vector3]
    _next_lookat: Tuple[mn.Vector3, mn.Vector3]
    _transition_duration: float
    _stage_duration: float
    _elapsed_time: float
    _display_text: str
    _object_animation: ObjectAnimation

    def __init__(
        self,
        stage_duration: float,
        next_lookat: Tuple[mn.Vector3, mn.Vector3],
        prev_lookat: Tuple[mn.Vector3, mn.Vector3] = None,
        transition_duration: float = 0.0,
        display_text: str = "",
        object_animation: ObjectAnimation = None,
    ) -> None:
        self._transition_duration = transition_duration
        self._stage_duration = stage_duration
        self._prev_lookat = prev_lookat
        self._next_lookat = next_lookat
        self._elapsed_time = 0.0
        self._display_text = display_text
        self._object_animation = object_animation

    def update(self, dt: float) -> None:
        self._elapsed_time += dt

        if self._object_animation is not None:
            self._object_animation.update(dt)

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

    def is_completed(self) -> bool:
        return self._elapsed_time >= self._stage_duration

    def get_display_text(self) -> str:
        return self._display_text


class Tutorial:
    _tutorial_stages: List[TutorialStage] = None
    _pending_object_animations: List[ObjectAnimation] = []
    _tutorial_stage_index: int = 0

    def __init__(self, tutorial_stages: List[TutorialStage]) -> None:
        self._tutorial_stages = tutorial_stages

    def update(self, dt: float) -> None:
        for object_animation in self._pending_object_animations:
            object_animation.update(
                dt
            )  # Keep updating so that objects are placed back

        if self._tutorial_stage_index >= len(self._tutorial_stages):
            return

        tutorial_stage = self._tutorial_stages[self._tutorial_stage_index]
        tutorial_stage.update(dt)

        if tutorial_stage.is_completed():
            self._tutorial_stage_index += 1

            if tutorial_stage._object_animation is not None:
                self._pending_object_animations.append(
                    tutorial_stage._object_animation
                )

    def is_completed(self) -> bool:
        return self._tutorial_stage_index >= len(self._tutorial_stages)

    def get_look_at_matrix(self) -> mn.Matrix4:
        assert not self.is_completed()
        return self._tutorial_stages[
            self._tutorial_stage_index
        ].get_look_at_matrix()

    def get_display_text(self) -> str:
        assert not self.is_completed()
        return self._tutorial_stages[
            self._tutorial_stage_index
        ].get_display_text()

    def reset(self) -> None:
        for object_animation in self._pending_object_animations:
            object_animation.reset()
        self._pending_object_animations.clear()


def generate_tutorial(
    sim, agent_idx: int, final_lookat: Tuple[mn.Vector3, mn.Vector3]
) -> Tutorial:
    assert sim is not None
    assert agent_idx is not None
    assert final_lookat is not None
    camera_fov_deg = 100  # TODO: Get the actual FOV
    tutorial_stages: List[TutorialStage] = []
    view_forward_vector = final_lookat[1] - final_lookat[0]
    rom = sim.get_rigid_object_manager()

    # Scene overview
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_target_bb: mn.Range3D = scene_root_node.cumulative_bb
    scene_top_down_lookat = _lookat_bounding_box_top_down(
        camera_fov_deg, scene_target_bb, view_forward_vector
    )
    tutorial_stages.append(
        TutorialStage(stage_duration=6.0, next_lookat=scene_top_down_lookat)
    )

    # Show all the targets
    # pathfinder = sim.pathfinder
    idxs, goal_pos = sim.get_targets()
    target_objs = [
        rom.get_object_by_id(sim._scene_obj_ids[idx]) for idx in idxs
    ]
    for target_obj in target_objs:
        prev_lookat = tutorial_stages[len(tutorial_stages) - 1]._next_lookat

        target_bb = mn.Range3D.from_center(
            mn.Vector3(target_obj.translation),
            mn.Vector3(1.0, 1.0, 1.0),
        )  # Assume 2x2x2m bounding box
        far_lookat = _lookat_bounding_box_top_down(
            camera_fov_deg / 3, target_bb, view_forward_vector
        )
        close_lookat = _lookat_bounding_box_top_down(
            camera_fov_deg, target_bb, view_forward_vector
        )

        # Top-down view over the object from far away
        tutorial_stages.append(
            TutorialStage(
                stage_duration=2.0,
                transition_duration=2.0,
                prev_lookat=prev_lookat,
                next_lookat=far_lookat,
            )
        )

        # Zoom-in on the object, and bring the object in front of the camera
        obj_anim = ObjectAnimation(
            obj=target_obj,
            view_lookat=close_lookat,
            distance_from_view=0.5,
            duration=3.0,
        )

        tutorial_stages.append(
            TutorialStage(
                stage_duration=3.0,
                transition_duration=1.5,
                prev_lookat=far_lookat,
                next_lookat=close_lookat,
                object_animation=obj_anim,
            )
        )

    # Controlled agent focus
    art_obj = sim.agents_mgr[agent_idx].articulated_agent.sim_obj
    agent_root_node = art_obj.get_link_scene_node(
        -1
    )  # Root link always has index -1
    target_bb = mn.Range3D.from_center(
        mn.Vector3(agent_root_node.absolute_translation),
        mn.Vector3(1.0, 1.0, 1.0),
    )  # Assume 2x2x2m bounding box
    agent_lookat = _lookat_bounding_box_top_down(
        camera_fov_deg / 3, target_bb, view_forward_vector
    )

    tutorial_stages.append(
        TutorialStage(
            stage_duration=2.0,
            transition_duration=2.0,
            prev_lookat=tutorial_stages[len(tutorial_stages) - 1]._next_lookat,
            next_lookat=agent_lookat,
        )
    )

    # Transition from the avatar view to simulated view (e.g. first person)
    tutorial_stages.append(
        TutorialStage(
            stage_duration=1.5,
            transition_duration=1.5,
            prev_lookat=tutorial_stages[len(tutorial_stages) - 1]._next_lookat,
            next_lookat=final_lookat,
        )
    )

    return Tutorial(tutorial_stages)


def _ease_fn_in_out_quat(t: float):
    if t < 0.5:
        return 16 * t * pow(t, 4)
    else:
        return 1 - pow(-2 * t + 2, 4) / 2


def _lookat_bounding_box_top_down(
    camera_fov_deg: float,
    target_bb: mn.Range3D,
    view_forward_vector: mn.Vector3,
) -> Tuple[mn.Vector3, mn.Vector3]:
    r"""
    Creates look-at vectors for a top-down camera such as the entire 'target_bb' bounding box is visible.
    The 'view_forward_vector' is used to control the yaw of the top-down view, which is otherwise gimbal locked.
    """
    camera_fov_rad = radians(camera_fov_deg)
    target_dimension = max(target_bb.size_x(), target_bb.size_z())
    camera_position = mn.Vector3(
        target_bb.center_x(),
        target_bb.center_y()
        + abs(target_dimension / math.sin(camera_fov_rad / 2)),
        target_bb.center_z(),
    )
    # Because of gimbal lock, we apply an epsilon bias to force a camera orientation
    epsilon = 0.0001
    view_forward_vector = view_forward_vector.normalized()
    camera_position = camera_position - (epsilon * view_forward_vector)
    return (
        camera_position,
        target_bb.center(),
    )
