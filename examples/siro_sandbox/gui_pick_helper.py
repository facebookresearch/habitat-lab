#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import magnum as mn
import numpy as np

DIST_HIGHLIGHT: Final[float] = 0.15


class GuiPickHelper:
    """Helper for picking up objects from the GUI."""

    def __init__(self, gui_service, agent_idx, agent_feet_height):
        self._sandbox_service = gui_service
        self._agent_idx = agent_idx
        self._rom = self._get_sim().get_rigid_object_manager()
        self.obj_ids = self._get_sim()._scene_obj_ids
        self.agent_feet_height = agent_feet_height
        self._dist_to_highlight_obj = DIST_HIGHLIGHT

    def _get_sim(self):
        return self._sandbox_service.sim

    def _closest_point_and_dist(self, origin, direction_vector, points):
        norm_direction = direction_vector / np.linalg.norm(direction_vector)
        vectors_to_points = points - origin
        dot_products = np.dot(vectors_to_points, norm_direction)
        closest_points = origin + dot_products[:, np.newaxis] * norm_direction
        distances = np.linalg.norm(closest_points - points, axis=1)
        return np.argmin(distances), np.min(distances)

    def on_environment_reset(self, agent_feet_height=0.15):
        sim = self._get_sim()
        self._rom = sim.get_rigid_object_manager()
        self.obj_ids = sim._scene_obj_ids
        self.agent_feet_height = agent_feet_height

    def _viz_object(self, this_target_pos):
        color = mn.Color3(0, 255, 0)  # green

        this_target_pos = mn.Vector3(this_target_pos)
        box_half_size = 0.20
        box_offset = mn.Vector3(box_half_size, box_half_size, box_half_size)
        self._sandbox_service.line_render.draw_box(
            this_target_pos - box_offset,
            this_target_pos + box_offset,
            color,
        )

        # draw can grasp area
        can_grasp_position = mn.Vector3(this_target_pos)
        can_grasp_position[1] = self.agent_feet_height
        self._sandbox_service.line_render.draw_circle(
            can_grasp_position,
            self._sandbox_service.hitl_config.can_grasp_place_threshold,
            mn.Color3(255 / 255, 255 / 255, 0),
            24,
        )

    def viz_and_get_pick_object(self):
        ray = self._sandbox_service.gui_input.mouse_ray
        if (
            not ray
            or ray.direction.y >= 0
            or ray.origin.y <= self.agent_feet_height
        ):
            return None

        object_coords = [
            np.array(
                self._rom.get_object_by_id(obj_id).transformation.translation
            )[None, ...]
            for obj_id in self.obj_ids
        ]
        if len(object_coords) == 0:
            return None
        object_coords = np.concatenate(object_coords)
        obj_id, distance = self._closest_point_and_dist(
            np.array(ray.origin), np.array(ray.direction), object_coords
        )
        if distance < self._dist_to_highlight_obj:
            self._viz_object(mn.Vector3(object_coords[obj_id]))
            return self.obj_ids[obj_id]
        else:
            return None
