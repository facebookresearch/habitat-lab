#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import math
from math import radians
from typing import List, Tuple

import magnum as mn

import habitat_sim

TEXT_SCENE_OVERVIEW: str = (
    "Take note of the following objects.\nYou need to bring them to the goals."
)
TEXT_ROBOT_FOCUS: str = (
    "This is your robot assistant.\nIt will help you accomplish your tasks.\n"
)
TEXT_AVATAR_FOCUS: str = "You will now gain control of your avatar."
TEXT_HELP: str = "Q: Skip \nSpacebar: Skip to the next stage of the tutorial"


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

    def _get_look_at_vectors(self) -> Tuple[mn.Vector3, mn.Vector3]:
        # If there's no transition, return the next view
        assert self._next_lookat
        if not self._prev_lookat or self._transition_duration <= 0.0:
            return (self._next_lookat[0], self._next_lookat[1])
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
        return (look_at[0], look_at[1])

    def get_look_at_matrix(self) -> mn.Matrix4:
        look_at = self._get_look_at_vectors()
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
        # Keep updating objects anims so that they are placed back
        for object_animation in self._pending_object_animations:
            object_animation.update(dt)

        if self._tutorial_stage_index >= len(self._tutorial_stages):
            return

        tutorial_stage = self._tutorial_stages[self._tutorial_stage_index]
        tutorial_stage.update(dt)

        if tutorial_stage.is_completed():
            self._next_stage()

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

    def get_help_text(self) -> str:
        return TEXT_HELP

    def skip_stage(self) -> None:
        prev_stage = self._tutorial_stages[self._tutorial_stage_index]
        # Skip stage
        self._next_stage()
        # Stop object animations
        self.stop_animations()
        # Change 'prev lookat' of next stage to start from transitioning viewpoint
        if not self.is_completed():
            next_stage = self._tutorial_stages[self._tutorial_stage_index]
            next_stage._prev_lookat = prev_stage._get_look_at_vectors()

    def stop_animations(self) -> None:
        for object_animation in self._pending_object_animations:
            object_animation.reset()
        self._pending_object_animations.clear()

    def _next_stage(self) -> None:
        if self.is_completed():
            return

        tutorial_stage = self._tutorial_stages[self._tutorial_stage_index]

        if tutorial_stage._object_animation is not None:
            self._pending_object_animations.append(
                tutorial_stage._object_animation
            )

        self._tutorial_stage_index += 1


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
    pathfinder = sim.pathfinder

    # Scene overview
    scene_root_node = sim.get_active_scene_graph().get_root_node()
    scene_target_bb: mn.Range3D = scene_root_node.cumulative_bb
    scene_top_down_lookat = _lookat_bounding_box_top_down(
        camera_fov_deg, scene_target_bb, view_forward_vector
    )
    tutorial_stages.append(
        TutorialStage(
            stage_duration=8.0,
            next_lookat=scene_top_down_lookat,
            display_text=TEXT_SCENE_OVERVIEW,
        )
    )

    # Show all the targets
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

    # Robot focus
    # Limitation: Only works if there's 1 robot
    if len(sim.agents_mgr) == 2:
        robot_idx = 1 if agent_idx == 0 else 0
        art_obj = sim.agents_mgr[robot_idx].articulated_agent.sim_obj
        agent_root_node = art_obj.get_link_scene_node(
            -1
        )  # Root link always has index -1
        target_bb = mn.Range3D.from_center(
            mn.Vector3(agent_root_node.absolute_translation),
            mn.Vector3(1.0, 1.0, 1.0),
        )  # Assume 2x2x2m bounding box
        robot_lookat_far = _lookat_bounding_box_top_down(
            camera_fov_deg / 3, target_bb, view_forward_vector
        )
        robot_lookat_near = _lookat_point_from_closest_navmesh_pos(
            agent_root_node.absolute_translation, 1.5, 1.0, pathfinder
        )
        tutorial_stages.append(
            TutorialStage(
                stage_duration=2.0,
                transition_duration=2.0,
                prev_lookat=tutorial_stages[
                    len(tutorial_stages) - 1
                ]._next_lookat,
                next_lookat=robot_lookat_far,
                display_text=TEXT_ROBOT_FOCUS,
            )
        )
        tutorial_stages.append(
            TutorialStage(
                stage_duration=3.0,
                transition_duration=1.5,
                prev_lookat=robot_lookat_far,
                next_lookat=robot_lookat_near,
                display_text=TEXT_ROBOT_FOCUS,
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
            display_text=TEXT_AVATAR_FOCUS,
        )
    )

    # Transition from the avatar view to simulated view (e.g. first person)
    tutorial_stages.append(
        TutorialStage(
            stage_duration=2.0,
            transition_duration=1.5,
            prev_lookat=tutorial_stages[len(tutorial_stages) - 1]._next_lookat,
            next_lookat=final_lookat,
            display_text=TEXT_AVATAR_FOCUS,
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
    max_dist_to_obstacle = 0.0
    for i in range(sample_count):
        radial_angle = i * 2.0 * math.pi / float(sample_count)
        dist_x = math.sin(radial_angle) * distance_from_target
        dist_z = math.cos(radial_angle) * distance_from_target
        candidate = mn.Vector3(target.x + dist_x, target.y, target.z + dist_z)
        if pathfinder.is_navigable(candidate, 3.0):
            dist_to_closest_obstacle = pathfinder.distance_to_closest_obstacle(
                candidate, 2.0
            )
            if dist_to_closest_obstacle > max_dist_to_obstacle:
                max_dist_to_obstacle = dist_to_closest_obstacle
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
