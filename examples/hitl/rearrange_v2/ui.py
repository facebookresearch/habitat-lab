#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import magnum as mn

from habitat.sims.habitat_simulator import sim_utilities
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentManager,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_hitl.core.gui_drawer import GuiDrawer
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import MouseButton
from habitat_hitl.core.selection import Selection
from habitat_hitl.core.user_mask import Mask
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.physics import ManagedArticulatedObject

# Verticality threshold for successful placement.
MINIMUM_DROP_VERTICALITY: float = 0.9

# Maximum delay between two clicks to be registered as a double-click.
DOUBLE_CLICK_DELAY: float = 0.33

_HI = 0.8
_LO = 0.4
# Color for a valid action.
COLOR_VALID = mn.Color4(0.0, _HI, 0.0, 1.0)  # Green
# Color for an invalid action.
COLOR_INVALID = mn.Color4(_HI, 0.0, 0.0, 1.0)  # Red
# Color for goal object-receptacle pairs.
COLOR_GOALS: List[mn.Color4] = [
    mn.Color4(0.0, _HI, _HI, 1.0),  # Cyan
    mn.Color4(_HI, 0.0, _HI, 1.0),  # Magenta
    mn.Color4(_HI, _HI, 0.0, 1.0),  # Yellow
    mn.Color4(_HI, 0.0, _LO, 1.0),  # Purple
    mn.Color4(_LO, _HI, 0.0, 1.0),  # Orange
]


class UI:
    """
    User interface for the rearrange_v2 app.
    Each user has their own UI class.
    """

    def __init__(
        self,
        hitl_config,
        user_index: int,
        gui_controller: GuiController,
        sim: RearrangeSim,
        gui_input: GuiInput,
        gui_drawer: GuiDrawer,
        camera_helper: CameraHelper,
    ):
        self._user_index = user_index
        self._dest_mask = Mask.from_index(self._user_index)
        self._gui_controller = gui_controller
        self._sim = sim
        self._gui_input = gui_input
        self._gui_drawer = gui_drawer
        self._camera_helper = camera_helper

        self._can_grasp_place_threshold = hitl_config.can_grasp_place_threshold

        # ID of the object being held. None if no object is held.
        self._held_object_id: Optional[int] = None
        # Cache of all link IDs and their parent articulated objects.
        self._link_id_to_ao_map: Dict[int, int] = {}
        # Cache of all opened articulated object links.
        self._opened_link_set: Set = set()
        # Cache of pickable objects IDs.
        self._pickable_object_ids: Set[int] = set()
        # Cache of interactable objects IDs.
        self._interactable_object_ids: Set[int] = set()
        # Last time a click was done. Used to track double-clicking.
        self._last_click_time: datetime = datetime.now()
        # Cache of goal object-receptacle pairs.
        self._object_receptacle_pairs: List[Tuple[List[int], List[int]]] = []

        # Selection trackers.
        self._selections: List[Selection] = []
        # Track hovered object.
        self._hover_selection = Selection(
            self._sim, self._gui_input, Selection.hover_fn
        )
        self._selections.append(self._hover_selection)
        # Track left-clicked object.
        self._click_selection = Selection(
            self._sim,
            self._gui_input,
            Selection.left_click_fn,
        )
        self._selections.append(self._click_selection)

        # Track drop placement.
        def place_selection_fn(gui_input: GuiInput) -> bool:
            return gui_input.get_mouse_button(
                MouseButton.RIGHT
            ) or gui_input.get_mouse_button_up(MouseButton.RIGHT)

        self._place_selection = Selection(
            self._sim,
            self._gui_input,
            place_selection_fn,
        )
        self._selections.append(self._place_selection)

    def reset(
        self, object_receptacle_pairs: List[Tuple[List[int], List[int]]]
    ) -> None:
        """
        Reset the UI. Call on simulator reset.
        """
        sim = self._sim

        self._held_object_id = None
        self._link_id_to_ao_map = sim_utilities.get_ao_link_id_map(sim)
        self._opened_link_set = set()
        self._object_receptacle_pairs = object_receptacle_pairs
        self._last_click_time = datetime.now()
        for selection in self._selections:
            selection.deselect()

        self._pickable_object_ids = set(sim._scene_obj_ids)
        for pickable_obj_id in self._pickable_object_ids:
            rigid_obj = self._get_rigid_object(pickable_obj_id)
            # Ensure that rigid objects are collidable.
            rigid_obj.collidable = True

        # Get set of interactable articulated object links.
        # Exclude all agents.
        agent_ao_object_ids: Set[int] = set()
        agent_manager: ArticulatedAgentManager = sim.agents_mgr
        for agent_index in range(len(agent_manager)):
            agent = agent_manager[agent_index]
            agent_ao = agent.articulated_agent.sim_obj
            agent_ao_object_ids.add(agent_ao.object_id)
        self._interactable_object_ids = set()
        aom = sim.get_articulated_object_manager()
        all_ao: List[
            ManagedArticulatedObject
        ] = aom.get_objects_by_handle_substring().values()
        # All add non-root links that are not agents.
        for ao in all_ao:
            if ao.object_id not in agent_ao_object_ids:
                for link_object_id in ao.link_object_ids:
                    if link_object_id != ao.object_id:
                        self._interactable_object_ids.add(link_object_id)

    def update(self) -> None:
        """
        Handle user actions and update the UI.
        """

        def _handle_double_click() -> bool:
            time_since_last_click = datetime.now() - self._last_click_time
            double_clicking = time_since_last_click < timedelta(
                seconds=DOUBLE_CLICK_DELAY
            )
            if not double_clicking:
                self._last_click_time = datetime.now()
            return double_clicking

        for selection in self._selections:
            selection.update()

        if self._gui_input.get_mouse_button_down(MouseButton.LEFT):
            clicked_object_id = self._click_selection.object_id
            if _handle_double_click():
                # Double-click to select pickable.
                if self._is_object_pickable(clicked_object_id):
                    self._pick_object(clicked_object_id)
                # Double-click to interact.
                elif self._is_object_interactable(clicked_object_id):
                    self._interact_with_object(clicked_object_id)

        # Drop when releasing right click.
        if self._gui_input.get_mouse_button_up(MouseButton.RIGHT):
            self._place_object()
            self._place_selection.deselect()

    def draw_ui(self) -> None:
        """
        Draw the UI.
        """
        self._update_held_object_placement()
        self._draw_place_selection()
        self._draw_hovered_interactable()
        self._draw_hovered_pickable()
        self._draw_goals()

    def _pick_object(self, object_id: int) -> None:
        """Pick the specified object_id. The object must be pickable."""
        if not self._is_holding_object() and self._is_object_pickable(
            object_id
        ):
            rigid_object = self._get_rigid_object(object_id)
            if rigid_object is not None:
                rigid_pos = rigid_object.translation
                if self._is_within_reach(rigid_pos):
                    # Pick the object.
                    self._held_object_id = object_id
                    self._place_selection.deselect()

    def _update_held_object_placement(self) -> None:
        """Update the location of the held object."""
        object_id = self._held_object_id
        if not object_id:
            return

        eye_position = self._camera_helper.get_eye_pos()
        forward_vector = (
            self._camera_helper.get_lookat_pos()
            - self._camera_helper.get_eye_pos()
        ).normalized()

        rigid_object = self._sim.get_rigid_object_manager().get_object_by_id(
            object_id
        )
        rigid_object.translation = eye_position + forward_vector

    def _place_object(self) -> None:
        """Place the currently held object."""
        if not self._place_selection.selected:
            return

        object_id = self._held_object_id
        point = self._place_selection.point
        normal = self._place_selection.normal
        if (
            object_id is not None
            and object_id != self._place_selection.object_id
            and self._is_location_suitable_for_placement(point, normal)
        ):
            # Drop the object.
            rigid_object = self._get_rigid_object(object_id)
            rigid_object.translation = point + mn.Vector3(
                0.0, rigid_object.collision_shape_aabb.size_y() / 2, 0.0
            )
            self._held_object_id = None
            self._place_selection.deselect()

    def _interact_with_object(self, object_id: int) -> None:
        """Open/close the selected object. Must be interactable."""
        if self._is_object_interactable(object_id):
            link_id = object_id
            link_index = self._get_link_index(link_id)
            if link_index:
                ao_id = self._link_id_to_ao_map[link_id]
                ao = self._get_articulated_object(ao_id)
                link_node = ao.get_link_scene_node(link_index)
                link_pos = link_node.translation
                if self._is_within_reach(link_pos):
                    # Open/close object.
                    if link_id in self._opened_link_set:
                        sim_utilities.close_link(ao, link_index)
                        self._opened_link_set.remove(link_id)
                    else:
                        sim_utilities.open_link(ao, link_index)
                        self._opened_link_set.add(link_id)

    def _user_pos(self) -> mn.Vector3:
        """Get the translation of the agent controlled by the user."""
        return get_agent_art_obj_transform(
            self._sim, self._gui_controller._agent_idx
        ).translation

    def _get_rigid_object(self, object_id: int) -> Optional[Any]:
        """Get the rigid object with the specified ID. Returns None if unsuccessful."""
        rom = self._sim.get_rigid_object_manager()
        return rom.get_object_by_id(object_id)

    def _get_articulated_object(self, object_id: int) -> Optional[Any]:
        """Get the articulated object with the specified ID. Returns None if unsuccessful."""
        aom = self._sim.get_articulated_object_manager()
        return aom.get_object_by_id(object_id)

    def _get_link_index(self, object_id: int) -> int:
        """Get the index of a link. Returns None if unsuccessful."""
        link_id = object_id
        if link_id in self._link_id_to_ao_map:
            ao_id = self._link_id_to_ao_map[link_id]
            ao = self._get_articulated_object(ao_id)
            link_id_to_index: Dict[int, int] = ao.link_object_ids
            if link_id in link_id_to_index:
                return link_id_to_index[link_id]
        return None

    def _horizontal_distance(self, a: mn.Vector3, b: mn.Vector3) -> float:
        """Compute the distance between two points on the horizontal plane."""
        displacement = a - b
        displacement.y = 0.0
        return mn.Vector3(displacement.x, 0.0, displacement.z).length()

    def _is_object_pickable(self, object_id: int) -> bool:
        """Returns true if the object can be picked."""
        return object_id is not None and object_id in self._pickable_object_ids

    def _is_object_interactable(self, object_id: int) -> bool:
        """Returns true if the object can be opened or closed."""
        return (
            object_id is not None
            and object_id in self._interactable_object_ids
            and object_id in self._link_id_to_ao_map
        )

    def _is_holding_object(self) -> bool:
        """Returns true if the user is holding an object."""
        return self._held_object_id is not None

    def _is_within_reach(self, target_pos: mn.Vector3) -> bool:
        """Returns true if the target can be reached by the user."""
        return (
            self._horizontal_distance(self._user_pos(), target_pos)
            < self._can_grasp_place_threshold
        )

    def _is_location_suitable_for_placement(
        self, point: mn.Vector3, normal: mn.Vector3
    ) -> bool:
        """Returns true if the target location is suitable for placement."""
        placement_verticality = mn.math.dot(normal, mn.Vector3(0, 1, 0))
        placement_valid = placement_verticality > MINIMUM_DROP_VERTICALITY
        return placement_valid and self._is_within_reach(point)

    def _draw_aabb(
        self, aabb: mn.Range3D, transform: mn.Matrix4, color: mn.Color3
    ) -> None:
        """Draw an AABB."""
        self._gui_drawer.push_transform(
            transform, destination_mask=self._dest_mask
        )
        self._gui_drawer.draw_box(
            min_extent=aabb.back_bottom_left,
            max_extent=aabb.front_top_right,
            color=color,
            destination_mask=self._dest_mask,
        )
        self._gui_drawer.pop_transform(destination_mask=self._dest_mask)

    def _draw_place_selection(self) -> None:
        """Draw the object placement selection."""
        if not self._place_selection.selected or self._held_object_id is None:
            return

        point = self._place_selection.point
        normal = self._place_selection.normal
        placement_valid = self._is_location_suitable_for_placement(
            point, normal
        )
        color = COLOR_VALID if placement_valid else COLOR_INVALID
        radius = 0.15 if placement_valid else 0.05
        self._gui_drawer.draw_circle(
            translation=point,
            radius=radius,
            color=color,
            normal=normal,
            billboard=False,
            destination_mask=self._dest_mask,
        )

    def _draw_hovered_interactable(self) -> None:
        """Highlight the hovered interactable object."""
        if not self._hover_selection.selected:
            return

        object_id = self._hover_selection.object_id
        if not self._is_object_interactable(object_id):
            return

        link_index = self._get_link_index(object_id)
        if link_index:
            ao = sim_utilities.get_obj_from_id(
                self._sim, object_id, self._link_id_to_ao_map
            )
            link_node = ao.get_link_scene_node(link_index)
            aabb = link_node.cumulative_bb
            reachable = self._is_within_reach(link_node.translation)
            color = COLOR_VALID if reachable else COLOR_INVALID
            self._draw_aabb(aabb, link_node.transformation, color)

    def _draw_hovered_pickable(self) -> None:
        """Highlight the hovered pickable object."""
        if not self._hover_selection.selected or self._is_holding_object():
            return

        object_id = self._hover_selection.object_id
        if not self._is_object_pickable(object_id):
            return

        managed_object = sim_utilities.get_obj_from_id(
            self._sim, object_id, self._link_id_to_ao_map
        )
        translation = managed_object.translation
        reachable = self._is_within_reach(translation)
        color = COLOR_VALID if reachable else COLOR_INVALID
        aabb = managed_object.collision_shape_aabb
        self._draw_aabb(aabb, managed_object.transformation, color)

    def _draw_goals(self) -> None:
        """Draw goal object-receptacle pairs."""
        # TODO: Cache
        sim = self._sim
        obj_receptacle_pairs = self._object_receptacle_pairs
        link_id_to_ao_map = self._link_id_to_ao_map
        dest_mask = self._dest_mask
        get_obj_from_id = sim_utilities.get_obj_from_id
        draw_gui_circle = self._gui_drawer.draw_circle
        draw_gui_aabb = self._draw_aabb

        for i in range(len(obj_receptacle_pairs)):
            rigid_ids = obj_receptacle_pairs[i][0]
            receptacle_ids = obj_receptacle_pairs[i][1]
            goal_pair_color = COLOR_GOALS[i % len(COLOR_GOALS)]
            for rigid_id in rigid_ids:
                managed_object = get_obj_from_id(
                    sim, rigid_id, link_id_to_ao_map
                )
                translation = managed_object.translation
                draw_gui_circle(
                    translation=translation,
                    radius=0.25,
                    color=goal_pair_color,
                    billboard=True,
                    destination_mask=dest_mask,
                )
            for receptacle_id in receptacle_ids:
                managed_object = get_obj_from_id(
                    sim, receptacle_id, link_id_to_ao_map
                )
                aabb, matrix = sim_utilities.get_bb_for_object_id(
                    sim, receptacle_id, link_id_to_ao_map
                )
                if aabb is not None:
                    draw_gui_aabb(aabb, matrix, goal_pair_color)