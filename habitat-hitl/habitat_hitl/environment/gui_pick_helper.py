#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import magnum as mn
import numpy as np

DIST_HIGHLIGHT: Final[float] = 0.15
COLOR_GRASPABLE: Final[mn.Color3] = mn.Color3(1, 0.75, 0)
COLOR_GRASP_PREVIEW: Final[mn.Color3] = mn.Color3(0.5, 1, 0)
COLOR_FOCUS_OBJECT: Final[mn.Color3] = mn.Color3(1, 1, 1)
RADIUS_GRASPABLE: Final[float] = 0.15
RADIUS_GRASP_PREVIEW: Final[float] = 0.2
RING_PULSE_SIZE: Final[float] = 0.03


class GuiPickHelper:
    """Helper for picking up objects from the GUI."""

    def __init__(self, gui_service, agent_idx):
        self._app_service = gui_service
        self._agent_idx = agent_idx
        self._rom = self._get_sim().get_rigid_object_manager()
        self._obj_ids = self._get_sim()._scene_obj_ids
        self._dist_to_highlight_obj = DIST_HIGHLIGHT

    def _get_sim(self):
        return self._app_service.sim

    def _closest_point_and_dist_to_ray(
        self, ray_origin, ray_direction_vector, points
    ):
        norm_direction = ray_direction_vector / np.linalg.norm(
            ray_direction_vector
        )
        vectors_to_points = points - ray_origin
        dot_products = np.dot(vectors_to_points, norm_direction)
        closest_points = (
            ray_origin + dot_products[:, np.newaxis] * norm_direction
        )
        distances = np.linalg.norm(closest_points - points, axis=1)
        return np.argmin(distances), np.min(distances)

    def on_environment_reset(self):
        sim = self._get_sim()
        self._rom = sim.get_rigid_object_manager()
        self._obj_ids = sim._scene_obj_ids

    def _closest_point_and_dist_to_query_position(self, points, query_pos):
        distances = np.linalg.norm(points - query_pos, axis=1)
        return np.argmin(distances), np.min(distances)

    def _get_object_positions(self):
        n = len(self._obj_ids)
        positions = np.zeros((n, 3), dtype=float)
        for i in range(n):
            obj_id = self._obj_ids[i]
            t = self._rom.get_object_by_id(obj_id).transformation.translation
            positions[i][0] = t.x
            positions[i][1] = t.y
            positions[i][2] = t.z
        return positions

    def get_pick_object_near_query_position(self, query_pos):
        obj_positions = self._get_object_positions()
        obj_index, distance = self._closest_point_and_dist_to_query_position(
            obj_positions, query_pos
        )
        if distance < self._app_service.hitl_config.can_grasp_place_threshold:
            self._pick_candidate_index = obj_index
            return self._obj_ids[obj_index]
        else:
            self._pick_candidate_index = None
            return None

    def _draw_circle(self, pos, color, radius, billboard):
        num_segments = 24
        self._app_service.line_render.draw_circle(
            pos,
            radius,
            color,
            num_segments,
        )
        if self._app_service.client_message_manager:
            self._app_service.client_message_manager.add_highlight(
                pos, radius, billboard=billboard, color=color
            )

    def _add_highlight_ring(
        self, pos, color, radius, do_pulse=False, billboard=True
    ):
        if do_pulse:
            radius += self._app_service.get_anim_fraction() * RING_PULSE_SIZE
        self._draw_circle(pos, color, radius, billboard)

    def viz_objects(self):
        obj_positions = self._get_object_positions()

        if self._pick_candidate_index is not None:
            obj_id = self._obj_ids[self._pick_candidate_index]
            pos = self._rom.get_object_by_id(obj_id).transformation.translation
            self._add_highlight_ring(
                pos,
                COLOR_GRASP_PREVIEW,
                RADIUS_GRASP_PREVIEW,
                do_pulse=False,
            )
        else:
            for i in range(len(obj_positions)):
                obj_id = self._obj_ids[i]
                pos = self._rom.get_object_by_id(
                    obj_id
                ).transformation.translation
                self._add_highlight_ring(
                    pos, COLOR_GRASPABLE, RADIUS_GRASPABLE, do_pulse=True
                )

    # Reference code
    # def viz_and_get_pick_object_mouse_ray(self):
    #     ray = self._app_service.gui_input.mouse_ray
    #     if (
    #         not ray
    #         or ray.direction.y >= 0
    #         or ray.origin.y <= self._agent_feet_height
    #     ):
    #         return None

    #     object_coords = [
    #         np.array(
    #             self._rom.get_object_by_id(obj_id).transformation.translation
    #         )[None, ...]
    #         for obj_id in self._obj_ids
    #     ]
    #     if len(object_coords) == 0:
    #         return None
    #     object_coords = np.concatenate(object_coords)
    #     obj_id, distance = self._closest_point_and_dist_to_ray(
    #         np.array(ray.origin), np.array(ray.direction), object_coords
    #     )
    #     if distance < self._dist_to_highlight_obj:
    #         self._viz_object(mn.Vector3(object_coords[obj_id]))
    #         return self._obj_ids[obj_id]
    #     else:
    #         return None
