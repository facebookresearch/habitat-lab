#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Set,
    Tuple,
    cast,
)

import magnum as mn
from ui_overlay import ObjectStateControl, UIOverlay
from world import World

import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.sims.habitat_simulator.object_state_machine import (
    BooleanObjectState,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.event import Event
from habitat_hitl.core.gui_drawer import GuiDrawer
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.selection import Selection
from habitat_hitl.core.user_mask import Mask
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.controller_abc import GuiController
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim import stage_id
from habitat_sim.geo import Ray
from habitat_sim.physics import RayHitInfo

if TYPE_CHECKING:
    from habitat.tasks.rearrange.rearrange_grasp_manager import (
        RearrangeGraspManager,
    )

# Verticality threshold for successful placement.
MINIMUM_DROP_VERTICALITY: float = 0.9

# Maximum delay between two clicks to be registered as a double-click.
DOUBLE_CLICK_DELAY: float = 0.33

_HI = 0.8
_LO = 0.1
# Color for a valid action.
COLOR_VALID = mn.Color4(_LO, _HI, _LO, 1.0)  # Green
# Color for an invalid action.
COLOR_INVALID = mn.Color4(_HI, _LO, _LO, 1.0)  # Red
# Color for goal object-receptacle pairs.
COLOR_HIGHLIGHT = mn.Color4(_HI, _HI, _LO, _HI)  # Yellow
# Color for selected objects.
COLOR_SELECTION = mn.Color4(_LO, _HI, _HI, 1.0)  # Cyan


@dataclass
class UISettings:
    """
    Configuration of the UI.
    """

    can_change_object_states: bool
    highlight_default_receptacles: bool


class UI:
    """
    User interface for the rearrange_v2 app.
    Each user has their own UI class.
    """

    def __init__(
        self,
        hitl_config,
        user_index: int,
        app_service: AppService,
        world: World,
        gui_controller: GuiController,
        sim: RearrangeSim,
        gui_input: GuiInput,
        gui_drawer: GuiDrawer,
        camera_helper: CameraHelper,
        ui_settings: UISettings,
    ):
        self._user_index = user_index
        self._dest_mask = Mask.from_index(self._user_index)
        self._world = world
        self._gui_controller = gui_controller
        self._sim = sim
        self._gui_input = gui_input
        self._gui_drawer = gui_drawer
        self._camera_helper = camera_helper
        self._ui_settings = ui_settings

        self._can_grasp_place_threshold = hitl_config.can_grasp_place_threshold

        # ID of the object being held. None if no object is held.
        self._held_object_id: Optional[int] = None
        # Last time a click was done. Used to track double-clicking.
        self._last_click_time: datetime = datetime.now()
        # Cache of goal object-receptacle pairs.
        self._object_receptacle_pairs: List[Tuple[List[int], List[int]]] = []

        # Selection trackers.
        self._selections: List[Selection] = []
        # Track hovered object.
        self._hover_selection = Selection(
            self._sim,
            self._gui_input,
            Selection.hover_fn,
            self.selection_discriminator_ignore_agents,
        )
        self._selections.append(self._hover_selection)
        # Track left-clicked object.
        self._click_selection = Selection(
            self._sim,
            self._gui_input,
            Selection.left_click_fn,
            self.selection_discriminator_ignore_agents,
        )
        self._selections.append(self._click_selection)

        # Track drop placement.
        def place_selection_fn(gui_input: GuiInput) -> bool:
            # TODO: Temp keyboard equivalent
            return (
                gui_input.get_mouse_button(MouseButton.RIGHT)
                or gui_input.get_mouse_button_up(MouseButton.RIGHT)
                or gui_input.get_key(KeyCode.SPACE)
                or gui_input.get_key_up(KeyCode.SPACE)
            )

        self._place_selection = Selection(
            self._sim,
            self._gui_input,
            place_selection_fn,
            self.selection_discriminator_ignore_agents,
        )
        self._selections.append(self._place_selection)

        # Set up UI overlay
        self._ui_overlay = UIOverlay(app_service, user_index)
        self._is_help_shown = True
        self._last_changed_state_timestamp: Optional[
            Tuple[str, datetime]
        ] = None

        # Set up user events
        self._on_pick = Event()
        self._on_place = Event()
        self._on_open = Event()
        self._on_close = Event()
        self._on_object_state_change = Event()

        # Disable the snap manager automatic object positioning so that object placement is controlled here.
        self._get_grasp_manager()._automatically_update_snapped_object = False

        # Set up object state manipulation.
        self._object_state_manipulator: Optional[
            "ObjectStateManipulator"
        ] = None
        try:
            from object_state_manipulator import ObjectStateManipulator

            self._object_state_manipulator = ObjectStateManipulator(
                sim=sim,
                agent_index=gui_controller._agent_idx,
                world=world,
                maximum_distance=self._can_grasp_place_threshold,
            )
        except Exception as e:
            print(f"Cannot load object state manipulator. {e}")

    @dataclass
    class PickEventData:
        object_id: int
        object_handle: str

    @property
    def on_pick(self) -> Event:
        return self._on_pick

    @dataclass
    class PlaceEventData:
        object_id: int
        object_handle: str
        receptacle_id: int

    @property
    def on_place(self) -> Event:
        return self._on_place

    @dataclass
    class OpenEventData:
        object_id: int
        object_handle: str

    @property
    def on_open(self) -> Event:
        return self._on_open

    @dataclass
    class CloseEventData:
        object_id: int
        object_handle: str

    @property
    def on_close(self) -> Event:
        return self._on_close

    @dataclass
    class StateChangeEventData:
        object_handle: str
        state_name: str
        new_value: Any

    @property
    def on_state_change(self) -> Event:
        return self._on_object_state_change

    def selection_discriminator_ignore_agents(self, object_id: int) -> bool:
        """Allow selection through agents."""
        return object_id not in self._world._agent_object_ids

    def reset(self) -> None:
        """
        Reset the UI. Call on simulator reset.
        """
        self._held_object_id = None
        self._last_click_time = datetime.now()
        for selection in self._selections:
            selection.deselect()
        self._ui_overlay.reset()

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

        # Deselect on right click or escape.
        if self._gui_input.get_mouse_button_down(
            MouseButton.RIGHT
        ) or self._gui_input.get_key_down(KeyCode.ESC):
            self._click_selection.deselect()
            # Select held object.
            if self._held_object_id is not None:
                self._click_selection._object_id = self._held_object_id

        # Handle double-click.
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
        # TODO: Temp keyboard equivalent
        if self._gui_input.get_mouse_button_up(
            MouseButton.RIGHT
        ) or self._gui_input.get_key_up(KeyCode.SPACE):
            self._place_object()
            self._place_selection.deselect()

        # Toggle help text.
        if self._gui_input.get_key_down(KeyCode.H):
            self._is_help_shown = not self._is_help_shown

        # Clear drop selection if not holding right click.
        if not self._gui_input.get_mouse_button(MouseButton.RIGHT):
            self._place_selection.deselect()

        # Update the UI overlay.
        # This activates callbacks on UI events.
        self._ui_overlay.update()

    def draw_ui(self) -> None:
        """
        Draw the UI.
        """
        self._update_overlay_help_text()

        self._update_held_object_placement()
        self._update_hovered_object_ui()
        self._update_selected_object_ui()

        self._draw_place_selection()
        self._draw_pickable_object_highlights()
        self._draw_hovered_object_highlights()
        self._draw_selected_object_highlights()

    def _get_grasp_manager(self) -> "RearrangeGraspManager":
        agent_mgr = self._sim.agents_mgr
        agent_data = agent_mgr._all_agent_data[self._gui_controller._agent_idx]
        return agent_data.grasp_mgr

    def _pick_object(self, object_id: int) -> None:
        """Pick the specified object_id. The object must be pickable and held by nobody else."""
        if (
            not self._is_holding_object()
            and self._is_object_pickable(object_id)
            and not self._world.is_any_agent_holding_object(object_id)
        ):
            rigid_object = self._world.get_rigid_object(object_id)
            if rigid_object is not None:
                rigid_pos = rigid_object.translation
                if self._is_within_reach(rigid_pos):
                    # Pick the object.
                    self._held_object_id = object_id
                    self._place_selection.deselect()
                    self._world._all_held_object_ids.add(object_id)

                    grasp_mgr = self._get_grasp_manager()
                    grasp_mgr.snap_to_obj(
                        snap_obj_id=object_id,
                        force=True,
                    )
                    sim = self._sim
                    if sim._kinematic_mode:
                        krm = sim.kinematic_relationship_manager
                        krm.relationship_graph.remove_obj_relations(
                            object_id, parents_only=True
                        )
                        krm.update_snapshots()

                    self._on_pick.invoke(
                        UI.PickEventData(
                            object_id=object_id,
                            object_handle=rigid_object.handle,
                        )
                    )

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
        receptacle_object_id = self._place_selection.object_id
        if (
            object_id is not None
            and object_id != self._place_selection.object_id
            and self._is_location_suitable_for_placement(
                point, normal, receptacle_object_id
            )
        ):
            # Drop the object.
            rigid_object = self._world.get_rigid_object(object_id)
            rigid_object.translation = point + mn.Vector3(
                0.0, rigid_object.collision_shape_aabb.size_y() / 2, 0.0
            )
            self._held_object_id = None
            self._place_selection.deselect()
            self._world._all_held_object_ids.remove(object_id)

            # Force the grasp manager to release the object.
            grasp_mgr = self._get_grasp_manager()
            grasp_mgr.desnap(force=True)
            grasp_mgr._snapped_obj_id = None
            grasp_mgr._snapped_marker_id = None
            grasp_mgr._managed_articulated_agent.close_gripper()

            # Update the kinematic relationships.
            sim = self._sim
            if sim._kinematic_mode and receptacle_object_id != stage_id:
                krm = sim.kinematic_relationship_manager
                krm.relationship_graph.add_relation(
                    receptacle_object_id, object_id, "ontop"
                )
                krm.update_snapshots()

            self._on_place.invoke(
                UI.PlaceEventData(
                    object_id=object_id,
                    object_handle=rigid_object.handle,
                    receptacle_id=receptacle_object_id,
                )
            )

    def update_overlay_instructions(
        self, instructions: Optional[str], warning_text: Optional[str]
    ):
        overlay = self._ui_overlay
        overlay.update_instructions_panel(
            instructions, warning_text, self._is_help_shown
        )

    def _update_overlay_help_text(self):
        """
        Update the UI overlay.
        """
        overlay = self._ui_overlay

        controls: Optional[List[Tuple[str, str]]] = (
            [
                ("H", "Hide Help"),
                ("WASD", "Move"),
                ("R/Middle-Click", "Look Around"),
                ("I/K", "Look Up/Down"),
                ("Left-Click", "Select"),
                ("Double-Click", "Pick-up"),
                ("Double-Click", "Open/Close"),
                ("Right-Click", "Drop"),
                ("0", "Finish Task"),
            ]
            if self._is_help_shown
            else None
        )

        overlay.update_controls_panel(controls)

    def _can_open_close_receptacle(self, link_pos: mn.Vector3) -> bool:
        return self._is_within_reach(link_pos) and self._held_object_id is None

    def _interact_with_object(self, object_id: int) -> None:
        """Open/close the selected object. Must be interactable."""
        if self._is_object_interactable(object_id):
            link_id = object_id
            link_index = self._world.get_link_index(link_id)
            if link_index:
                ao_id = self._world._link_id_to_ao_map[link_id]
                ao = self._world.get_articulated_object(ao_id)
                link_node = ao.get_link_scene_node(link_index)
                link_pos = link_node.translation

                if self._can_open_close_receptacle(link_pos):
                    # Open/close receptacle.
                    if link_id in self._world._opened_link_set:
                        sutils.close_link(ao, link_index)
                        self._world._opened_link_set.remove(link_id)
                        if self._sim._kinematic_mode:
                            # first apply the current relative transform
                            self._sim.kinematic_relationship_manager.apply_relations()
                            # then update the root transform of the closed link (links are always root parents)
                            self._sim.kinematic_relationship_manager.update_snapshots(
                                root_parent_subset=[link_id]
                            )
                        self._on_close.invoke(
                            UI.OpenEventData(
                                object_id=object_id,
                                object_handle=ao.handle,
                            )
                        )
                    else:
                        sutils.open_link(ao, link_index)
                        self._world._opened_link_set.add(link_id)
                        if self._sim._kinematic_mode:
                            # first apply the current relative transform
                            self._sim.kinematic_relationship_manager.apply_relations()
                            # then update the root transform of the opened link (links are always root parents)
                            self._sim.kinematic_relationship_manager.update_snapshots(
                                root_parent_subset=[link_id]
                            )
                        self._on_open.invoke(
                            UI.CloseEventData(
                                object_id=object_id,
                                object_handle=ao.handle,
                            )
                        )

    def _user_pos(self) -> mn.Vector3:
        """Get the translation of the agent controlled by the user."""
        return get_agent_art_obj_transform(
            self._sim, self._gui_controller._agent_idx
        ).translation

    def _horizontal_distance(self, a: mn.Vector3, b: mn.Vector3) -> float:
        """Compute the distance between two points on the horizontal plane."""
        displacement = a - b
        displacement.y = 0.0
        return mn.Vector3(displacement.x, 0.0, displacement.z).length()

    def _is_object_pickable(self, object_id: int) -> bool:
        """Returns true if the object can be picked."""
        return (
            object_id is not None
            and object_id in self._world._pickable_object_ids
        )

    def _is_object_interactable(self, object_id: int) -> bool:
        """Returns true if the object can be opened or closed."""
        world = self._world
        return (
            object_id is not None
            and object_id in world._interactable_object_ids
            and object_id in world._link_id_to_ao_map
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
        self,
        point: mn.Vector3,
        normal: mn.Vector3,
        receptacle_object_id: int,
    ) -> bool:
        """Returns true if the target location is suitable for placement."""
        # Cannot place on agents.
        if receptacle_object_id in self._world._agent_object_ids:
            return False
        # Cannot place on non-horizontal surfaces.
        placement_verticality = mn.math.dot(normal, mn.Vector3(0, 1, 0))
        if placement_verticality < MINIMUM_DROP_VERTICALITY:
            return False
        # Cannot place further than reach.
        if not self._is_within_reach(point):
            return False
        # Cannot place on objects held by agents.
        if self._world.is_any_agent_holding_object(receptacle_object_id):
            return False
        # NOTE: _info_text contains the reason for the failure
        (
            matching_rec_names,
            _conf,
            _info_text,
        ) = sutils.get_obj_receptacle_and_confidence(
            sim=self._sim,
            obj=sutils.get_obj_from_id(self._sim, self._held_object_id),
            obj_bottom_location=point,
            support_surface_id=receptacle_object_id,
            candidate_receptacles=self._sim.receptacles,
            island_index=self._sim._largest_indoor_island_idx,
        )
        if len(matching_rec_names) == 0 or (
            "floor" not in matching_rec_names[0]
            and matching_rec_names[0] not in self._sim.receptacles
        ):
            # no matches with receptacle or floor. Region match isn't enough for placement.
            return False
        return True

    def _raycast(
        self, ray: Ray, discriminator: Callable[[int], bool]
    ) -> Optional[RayHitInfo]:
        """
        Raycast the scene using the specified ray.
        Objects rejected by the discriminator function are transparent to selection.
        """
        raycast_results = self._sim.cast_ray(ray=ray)
        if not raycast_results.has_hits():
            return None
        # Results are sorted by distance. [0] is the nearest one.
        hits = raycast_results.hits
        for hit in hits:
            object_id: int = hit.object_id
            if not discriminator(object_id):
                continue
            else:
                return hit

        return None

    def _dot_object(self, object_id: int) -> float:
        """
        Dot product between the camera forward vector and the specified object.
        """
        sim = self._sim
        cam_direction = self._camera_helper.get_cam_forward_vector()
        cam_translation = self._camera_helper.get_eye_pos()
        obj_translation = sutils.get_obj_from_id(sim, object_id).translation
        ray_direction = obj_translation - cam_translation
        return mn.math.dot(
            cam_direction.normalized(), ray_direction.normalized()
        )

    def _is_object_visible(self, object_id: int) -> bool:
        """
        Returns true if the camera can see the object.
        """
        world = self._world
        sim = self._sim

        cam_translation = self._camera_helper.get_eye_pos()
        obj_translation = sutils.get_obj_from_id(sim, object_id).translation
        ray_direction = obj_translation - cam_translation

        # Check if object is in front of camera before raycasting.
        if self._dot_object(object_id) <= 0:
            return False

        ray = Ray(origin=cam_translation, direction=ray_direction)

        def discriminator(object_id: int) -> bool:
            return (
                object_id not in world._agent_object_ids
                and object_id not in world._all_held_object_ids
            )

        hit_info = self._raycast(ray, discriminator)
        if hit_info is not None:
            return hit_info.object_id == object_id
        else:
            return False

    def _update_hovered_object_ui(self):
        """Draw a UI when hovering an object with the cursor."""
        object_id = self._hover_selection.object_id

        object_category: Optional[str] = None
        object_states: List[Tuple[str, str]] = []
        primary_region_name: Optional[str] = None

        if object_id is not None:
            world = self._world
            sim = self._sim
            obj = sutils.get_obj_from_id(
                sim, object_id, world._link_id_to_ao_map
            )
            if obj is not None:
                object_category = world.get_category_from_handle(obj.handle)

                obj_states = world.get_states_for_object_handle(obj.handle)
                for state in obj_states:
                    spec = state.state_spec
                    if isinstance(spec, BooleanObjectState):
                        val = cast(bool, state.value)
                        object_states.append(
                            (
                                spec.display_name,
                                "True" if val else "False",
                            )
                        )
                    else:
                        # Unsupported type.
                        pass

                primary_region = world.get_primary_object_region(obj)
                if primary_region is not None:
                    primary_region_name = primary_region.category.name()

        self._ui_overlay.update_hovered_object_info_panel(
            object_category, object_states, primary_region_name
        )

    def _update_selected_object_ui(self):
        """Draw a UI for the currently selected object."""
        object_id = self._click_selection.object_id

        object_category_name: Optional[str] = None
        object_state_controls: List[ObjectStateControl] = []
        primary_region_name: Optional[str] = None
        contextual_info: List[Tuple[str, Optional[List[float]]]] = []

        if object_id is None:
            return

        if object_id is not None:
            world = self._world
            sim = self._sim
            obj = sutils.get_obj_from_id(
                sim, object_id, world._link_id_to_ao_map
            )
            if obj is not None:
                # Get object category name.
                object_category_name = world.get_category_from_handle(
                    obj.handle
                )
                if object_category_name is None:
                    object_category_name = "Object"

                # Get object state controls.
                object_state_controls = self._get_object_state_controls(
                    obj.handle
                )

                # Get the primary region name.
                primary_region = world.get_primary_object_region(obj)
                if primary_region is not None:
                    primary_region_name = primary_region.category.name()

                # Get the contextual information.
                color_ui_valid = [0.2, 1.0, 0.2, 1.0]
                color_ui_invalid = [1.0, 0.2, 0.2, 1.0]
                color_ui_info = None
                color_ui_object_info = [0.8, 0.8, 0.2, 1.0]
                if self._is_object_pickable(object_id):
                    if self._held_object_id == None:
                        if self._world.is_any_agent_holding_object(object_id):
                            contextual_info.append(
                                (
                                    "The object is held by another agent.",
                                    color_ui_invalid,
                                )
                            )
                        else:
                            if self._is_within_reach(obj.translation):
                                contextual_info.append(
                                    (
                                        "Double-click to pick up.",
                                        color_ui_valid,
                                    )
                                )
                            else:
                                contextual_info.append(
                                    ("Too far to pick up.", color_ui_invalid)
                                )
                    elif self._held_object_id == object_id:
                        if self._place_selection.point is not None:
                            point = self._place_selection.point
                            normal = self._place_selection.normal
                            receptacle_object_id = (
                                self._place_selection.object_id
                            )
                            placement_valid = (
                                self._is_location_suitable_for_placement(
                                    point, normal, receptacle_object_id
                                )
                            )
                            if placement_valid:
                                contextual_info.append(
                                    ("Release to place.", color_ui_valid)
                                )
                            else:
                                contextual_info.append(
                                    (
                                        "Cannot place object here.",
                                        color_ui_invalid,
                                    )
                                )
                        else:
                            contextual_info.append(
                                ("Hold right-click to place.", color_ui_info)
                            )
                elif self._is_object_interactable(object_id):
                    link_id = object_id
                    link_index = self._world.get_link_index(link_id)
                    if link_index:
                        action_name = (
                            "close"
                            if object_id in world._opened_link_set
                            else "open"
                        )

                        # Get parent articulated object.
                        ao_id = self._world._link_id_to_ao_map[link_id]
                        ao = self._world.get_articulated_object(ao_id)

                        if self._held_object_id is not None:
                            contextual_info.append(
                                (
                                    f"Cannot {action_name} while holding an object.",
                                    color_ui_invalid,
                                )
                            )
                        else:
                            link_node = ao.get_link_scene_node(link_index)
                            link_pos = link_node.translation

                            if self._is_within_reach(link_pos):
                                contextual_info.append(
                                    (
                                        f"Double-click to {action_name}.",
                                        color_ui_valid,
                                    )
                                )
                            else:
                                contextual_info.append(
                                    (
                                        f"Too far to {action_name}.",
                                        color_ui_invalid,
                                    )
                                )

                    if self._ui_settings.highlight_default_receptacles:
                        contextual_info.append(
                            (
                                "Yellow container\nmay have objects.",
                                color_ui_object_info,
                            )
                        )

        # Update the UI.
        self._ui_overlay.update_selected_object_panel(
            object_category_name=object_category_name,
            object_state_controls=object_state_controls,
            primary_region_name=primary_region_name,
            contextual_info=contextual_info,
        )

    def _get_object_state_controls(
        self, object_handle: str
    ) -> List[ObjectStateControl]:
        """
        Create a list of object state manipulation controls for the specified object.
        """
        # Requires the object state
        osm = self._object_state_manipulator
        if osm is None:
            return []

        object_state_controls: List[ObjectStateControl] = []

        # Get all possible actions for this object.
        all_possible_actions = osm.get_all_available_boolean_actions(
            object_handle
        )
        for action in all_possible_actions:
            spec = action.state_spec
            recently_changed = False

            # If this user can manipulate object states...
            if self._ui_settings.can_change_object_states:
                enabled = action.enabled
                available = action.available
                tooltip = (
                    action.error if action.available else "Action unavailable."
                )
                callback = partial(
                    self._state_change_callback,
                    spec.name,
                    action.target_value,
                    object_handle,
                )

                if (
                    self._last_changed_state_timestamp is not None
                    and self._last_changed_state_timestamp[0] == spec.name
                ):
                    time_since_last_state_change = (
                        datetime.now() - self._last_changed_state_timestamp[1]
                    )
                    recently_changed = (
                        time_since_last_state_change < timedelta(seconds=2.0)
                    )
                    if recently_changed:
                        tooltip = "Action executed."

            # If this user cannot manipulate object states...
            else:
                enabled = False
                available = False
                tooltip = "The robot cannot do this action."
                callback = None

            object_state_controls.append(
                ObjectStateControl(
                    spec=spec,
                    value=action.current_value,
                    enabled=enabled,
                    available=available,
                    callback=callback,
                    tooltip=tooltip,
                    recently_changed=recently_changed,
                )
            )
        return object_state_controls

    def _state_change_callback(
        self, state_name: str, target_value: Any, object_handle: str
    ):
        # Requires object state manipulator.
        osm = self._object_state_manipulator
        if osm is None or not self._ui_settings.can_change_object_states:
            return

        result = osm.try_execute_action(
            state_name, target_value, object_handle
        )
        if result.success:
            self._on_object_state_change.invoke(
                UI.StateChangeEventData(
                    object_handle=object_handle,
                    state_name=state_name,
                    new_value=target_value,
                )
            )

            self._last_changed_state_timestamp = (state_name, datetime.now())

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
        receptacle_object_id = self._place_selection.object_id
        placement_valid = self._is_location_suitable_for_placement(
            point, normal, receptacle_object_id
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

    def _draw_pickable_object_highlights(self):
        """Draw a highlight circle around visible pickable objects."""
        if self._is_holding_object():
            return

        sim = self._sim
        world = self._world
        draw_gui_circle = self._gui_drawer.draw_circle
        dest_mask = self._dest_mask

        for object_id in world._pickable_object_ids:
            if not world.is_any_agent_holding_object(
                object_id
            ) and self._is_object_visible(object_id):
                obj = sutils.get_obj_from_id(sim, object_id)

                # Make highlights near the edge of the screen less opaque.
                dot_object = self._dot_object(object_id)
                opacity = max(min(dot_object, 1.0), 0.0)
                opacity *= opacity

                # Calculate radius
                aabb = obj.collision_shape_aabb
                diameter = max(aabb.size_x(), aabb.size_y(), aabb.size_z())
                radius = diameter * 0.65

                # Calculate color
                if (
                    self._hover_selection.object_id == object_id
                    or self._click_selection.object_id == object_id
                ):
                    reachable = self._is_within_reach(obj.translation)
                    color = COLOR_VALID if reachable else COLOR_INVALID
                    color[3] = (
                        0.3
                        if self._click_selection.object_id != object_id
                        else 1.0
                    )
                else:
                    color = COLOR_HIGHLIGHT

                draw_gui_circle(
                    translation=obj.translation,
                    radius=radius,
                    color=color,
                    billboard=True,
                    destination_mask=dest_mask,
                )

    def _draw_hovered_object_highlights(self) -> None:
        """Highlight the hovered object."""
        object_id = self._hover_selection.object_id
        if object_id is None:
            return

        sim = self._sim
        obj = sutils.get_obj_from_id(
            sim, object_id, self._world._link_id_to_ao_map
        )
        if obj is None:
            return

        # Draw the outline of the highlighted interactable link (e.g. drawer in a cabinet).
        if self._is_object_interactable(object_id):
            link_index = self._world.get_link_index(object_id)
            if link_index:
                link_node = obj.get_link_scene_node(link_index)
                reachable = self._can_open_close_receptacle(
                    link_node.translation
                )
                color = COLOR_VALID if reachable else COLOR_INVALID
                color = self._to_color_array(color)
                color[3] = 0.3  # Make hover color dimmer than selection.
                self._gui_drawer._client_message_manager.draw_object_outline(
                    priority=1,
                    color=color,
                    line_width=8.0,
                    object_ids=[object_id],
                    destination_mask=Mask.from_index(self._user_index),
                )

    def _draw_selected_object_highlights(self) -> None:
        """Highlight the selected object."""
        object_id = self._click_selection.object_id
        if object_id is None:
            return

        sim = self._sim
        obj = sutils.get_obj_from_id(
            sim, object_id, self._world._link_id_to_ao_map
        )
        if obj is None:
            return

        world = self._world

        # Draw the outline of the highlighted interactable link (e.g. drawer in a cabinet).
        if self._is_object_interactable(object_id):
            link_index = self._world.get_link_index(object_id)
            if link_index:
                link_node = obj.get_link_scene_node(link_index)
                reachable = self._can_open_close_receptacle(
                    link_node.translation
                )
                color = COLOR_VALID if reachable else COLOR_INVALID
                color = self._to_color_array(color)
                self._gui_drawer._client_message_manager.draw_object_outline(
                    priority=2,
                    color=color,
                    line_width=8.0,
                    object_ids=[object_id],
                    destination_mask=Mask.from_index(self._user_index),
                )

        # Draw outline of selected object.
        # Articulated objects are always fully contoured.
        object_ids: Set[int] = set()
        object_ids.add(object_id)
        if object_id in world._link_id_to_ao_map:
            ao_id = world._link_id_to_ao_map[object_id]
            ao = sutils.get_obj_from_id(sim, ao_id, world._link_id_to_ao_map)
            link_object_ids = ao.link_object_ids
            for link_id in link_object_ids.keys():
                object_ids.add(link_id)

            # Draw an outline around the default receptacle.
            # TODO: Cache the default receptacle.
            if self._ui_settings.highlight_default_receptacles:
                link_index = self._world.get_link_index(object_id)
                default_link_index = sutils.get_ao_default_link(
                    ao, compute_if_not_found=True
                )
                if (
                    default_link_index is not None
                    and default_link_index != link_index
                ):
                    default_link_node = obj.get_link_scene_node(
                        default_link_index
                    )
                    if (
                        default_link_node.object_semantic_id
                        not in self._world._opened_link_set
                    ):
                        color_default_link = self._to_color_array(
                            COLOR_HIGHLIGHT
                        )
                        color_default_link[3] = 0.75
                        self._gui_drawer._client_message_manager.draw_object_outline(
                            priority=-10,
                            color=color_default_link,
                            line_width=4.0,
                            object_ids=[default_link_node.object_semantic_id],
                            destination_mask=Mask.from_index(self._user_index),
                        )

        self._gui_drawer._client_message_manager.draw_object_outline(
            priority=0,
            color=self._to_color_array(COLOR_SELECTION),
            line_width=4.0,
            object_ids=list(object_ids),
            destination_mask=Mask.from_index(self._user_index),
        )

    @staticmethod
    def _to_color_array(color: mn.Color4) -> List[float]:
        return [color.r, color.g, color.b, color.a]
