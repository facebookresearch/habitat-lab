#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Tuple, cast

import magnum as mn
import numpy as np
from scipy import spatial
from ui_overlay import UIOverlay
from world import World

import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.datasets.rearrange.samplers.receptacle import (
    Receptacle,
    TriangleMeshReceptacle,
)
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
    ):
        self._user_index = user_index
        self._dest_mask = Mask.from_index(self._user_index)
        self._world = world
        self._gui_controller = gui_controller
        self._sim = sim
        self._gui_input = gui_input
        self._gui_drawer = gui_drawer
        self._camera_helper = camera_helper

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

        # Set up user events
        self._on_pick = Event()
        self._on_place = Event()
        self._on_open = Event()
        self._on_close = Event()

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

    def point_to_tri_dist(
        self, point: np.ndarray, triangles: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the minimum distance between a 3D point and a set of triangles (e.g. a triangle mesh) and return both the minimum distance and that closest point.
        Uses vectorized numpy operations for high performance with a large number of triangles.
        Implementation adapted from https://stackoverflow.com/questions/32342620/closest-point-projection-of-a-3d-point-to-3d-triangles-with-numpy-scipy
        Algorithm is vectorized form of e.g. https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

        :param point: A 3D point.
        :param triangles: An nx3x3 numpy array of triangles. Each entry of the first axis is a triangle with three 3D vectors, the vertices of the triangle.
        :return: The minimum distance from point to triangle set and the closest point on the surface of any triangle.
        """

        with np.errstate(all="ignore"):
            # Unpack triangle points
            p0, p1, p2 = np.asarray(triangles).swapaxes(0, 1)

            # Calculate triangle edges
            e0 = p1 - p0
            e1 = p2 - p0
            a = np.einsum("...i,...i", e0, e0)
            b = np.einsum("...i,...i", e0, e1)
            c = np.einsum("...i,...i", e1, e1)

            # Calculate determinant and denominator
            det = a * c - b * b
            invDet = 1.0 / det
            denom = a - 2 * b + c

            # Project to the edges
            p = p0 - point
            d = np.einsum("...i,...i", e0, p)
            e = np.einsum("...i,...i", e1, p)
            u = b * e - c * d
            v = b * d - a * e

            # Calculate numerators
            bd = b + d
            ce = c + e
            numer0 = (ce - bd) / denom
            numer1 = (c + e - b - d) / denom
            da = -d / a
            ec = -e / c

            # Vectorize test conditions
            m0 = u + v < det
            m1 = u < 0
            m2 = v < 0
            m3 = d < 0
            m4 = a + d > b + e
            m5 = ce > bd

            t0 = m0 & m1 & m2 & m3
            t1 = m0 & m1 & m2 & ~m3
            t2 = m0 & m1 & ~m2
            t3 = m0 & ~m1 & m2
            t4 = m0 & ~m1 & ~m2
            t5 = ~m0 & m1 & m5
            t6 = ~m0 & m1 & ~m5
            t7 = ~m0 & m2 & m4
            t8 = ~m0 & m2 & ~m4
            t9 = ~m0 & ~m1 & ~m2

            u = np.where(t0, np.clip(da, 0, 1), u)
            v = np.where(t0, 0, v)
            u = np.where(t1, 0, u)
            v = np.where(t1, 0, v)
            u = np.where(t2, 0, u)
            v = np.where(t2, np.clip(ec, 0, 1), v)
            u = np.where(t3, np.clip(da, 0, 1), u)
            v = np.where(t3, 0, v)
            u *= np.where(t4, invDet, 1)
            v *= np.where(t4, invDet, 1)
            u = np.where(t5, np.clip(numer0, 0, 1), u)
            v = np.where(t5, 1 - u, v)
            u = np.where(t6, 0, u)
            v = np.where(t6, 1, v)
            u = np.where(t7, np.clip(numer1, 0, 1), u)
            v = np.where(t7, 1 - u, v)
            u = np.where(t8, 1, u)
            v = np.where(t8, 0, v)
            u = np.where(t9, np.clip(numer1, 0, 1), u)
            v = np.where(t9, 1 - u, v)
            u = u[:, None]
            v = v[:, None]

            # this array contains a list of points, the closest on each triangle
            closest_points_each_tri = p0 + u * e0 + v * e1

            # now extract the closest point on the mesh and minimum distance for return
            closest_point_index = np.argmin(
                spatial.distance.cdist(
                    np.array([point]), closest_points_each_tri
                ),
                axis=1,
            )
            closest_point: np.ndarray = closest_points_each_tri[
                closest_point_index
            ]
            min_dist = float(np.linalg.norm(point - closest_point))

            # Return the minimum distance
            return min_dist, closest_point

    def compute_dist_to_recs(
        self, point: np.ndarray, candidate_recs: List[Receptacle]
    ) -> List[float]:
        """
        For each receptacle in the input list, compute a distance from point to receptacle and return the list of distances.

        :param point: A 3D point in global space. Typically the bottom center point of a placed object.
        :param candidate_recs: A list of candidate Receptacles which could be matched to the point. Typically a subset of all Receptacles.
        :return: A list of point to Receptacle distances, one for each input in candidate_recs .
        """

        dist_to_recs = []
        for rec in candidate_recs:
            if isinstance(rec, TriangleMeshReceptacle):
                t_form = rec.get_global_transform(self._sim)
                # optimization: transform the point into local space instead of transforming the mesh into global space
                local_point = t_form.inverted().transform_point(point)
                # iterate over the triangles, getting point to edge distances
                # NOTE: list of lists, each with 3 numpy arrays, one for each vertex
                # TODO: these could be cached since it doesn't require local->global transform
                triangles = []
                for f_ix in range(int(len(rec.mesh_data.indices) / 3)):
                    v = rec.get_face_verts(f_ix)
                    triangles.append(v)
                np_tri = np.array(triangles)
                np_point = np.array(local_point)
                # compute the minimum point to mesh distance
                p_to_t_dist = self.point_to_tri_dist(np_point, np_tri)[0]
                dist_to_recs.append(p_to_t_dist)
            else:
                raise NotImplementedError(
                    "TODO: add handling for other Receptacle types."
                )

        return dist_to_recs

    def get_place_obj_receptacle_and_confidence(
        self,
        bottom_point: np.ndarray,
        support_surface_id: int,
        max_dist_to_rec: float = 0.25,
    ) -> Tuple[Optional[str], float, str]:
        """
        Heuristic to match a potential placement point with a Receptacle and provide some confidence.

        :param bottom_point: The bottom center point of the object or equivalent (e.g the candidate raycast point for placement)
        :param support_surface_id: The object_id of the intended support surface (rigid object, articulated link, or stage_id)
        :param max_dist_to_rec: The threshold point to mesh distance for an object to be matched with a Receptacle.
        :return: Tuple containing: (1): "floor,region", Receptacle.unique_name, or None (2): a floating point confidence score [0,1] (3): a message string describing the results for use in a UI tooltip
        """
        info_text = ""
        try_floor = False
        if support_surface_id == stage_id:
            # support_surface on stage could be the floor
            try_floor = True
        else:
            support_object = sutils.get_obj_from_id(
                self._sim, support_surface_id
            )
            matching_recs = [
                rec
                for u_name, rec in self._sim.receptacles.items()
                if support_object.handle in u_name
            ]
            if support_object.object_id != support_surface_id:
                # support object is a link
                link_index = support_object.link_object_ids[
                    self._place_selection.object_id
                ]
                # further cull the list to this link's recs
                matching_recs = [
                    rec
                    for rec in matching_recs
                    if rec.parent_link == link_index
                ]
            if len(matching_recs) == 0:
                # there are no Receptacles for this support surface
                try_floor = True
            else:
                # select a Receptacle which most likely contains the point
                dist_to_recs = self.compute_dist_to_recs(
                    bottom_point, matching_recs
                )
                index_min = min(
                    range(len(dist_to_recs)), key=dist_to_recs.__getitem__
                )
                min_dist = dist_to_recs[index_min]
                if min_dist < max_dist_to_rec:
                    # return the closest receptacle within distance threshold
                    return (
                        matching_recs[index_min].unique_name,
                        1.0 - (min_dist / max_dist_to_rec),
                        "successful match",
                    )
                else:
                    info_text = "Point is too far from a valid Receptacle on the support surface."

        # check if the point is navigable and if so, try matching it to a region
        if try_floor:
            if self._sim.pathfinder.is_navigable(bottom_point):
                # this point is on the floor and should be mapped to a region
                point_regions = (
                    self._sim.semantic_scene.get_weighted_regions_for_point(
                        bottom_point
                    )
                )
                if len(point_regions) > 0:
                    # found matching regions, pick the primary (most precise) one
                    region_name = self._sim.semantic_scene.regions[
                        point_regions[0][0]
                    ].id
                else:
                    # point is not matched to a region
                    region_name = "unknown_region"
                return f"floor,{region_name}", 1.0, "successful match"
            else:
                info_text = (
                    "Point does not match any Receptacle and is not navigable."
                )

        # all receptacles are too far away or there are no matches
        return None, 1.0, info_text

    def _place_object(self) -> None:
        """Place the currently held object."""
        if not self._place_selection.selected:
            return

        object_id = self._held_object_id
        point = self._place_selection.point
        normal = self._place_selection.normal
        receptacle_object_id = self._place_selection.object_id

        # check for a valid Receptacle mapping for the place point
        # TODO: cache this ground truth mapping in the trajectory?
        (
            _placement_receptacle,
            _confidence,
            _info_text,
        ) = self.get_place_obj_receptacle_and_confidence(
            point, receptacle_object_id
        )
        print(
            f"Placed object on Receptacle '{_placement_receptacle}', confidence[0,1]={_confidence}. Info text: {_info_text}"
        )
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
        # check if the placement matches a Receptacle
        (
            recepacle_name,
            _confidence,
            _info_text,
        ) = self.get_place_obj_receptacle_and_confidence(
            point, receptacle_object_id
        )
        if recepacle_name is None:
            # TODO: display the _info_text with failure message
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
        return hit_info.object_id == object_id

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

        if primary_region_name is not None and object_category is not None:
            # TODO: Draw UI
            # print(f"Region: {primary_region_name}")
            # print(f"Category: {object_category}")
            # print(f"States: {object_states}")
            pass

    def _update_selected_object_ui(self):
        """Draw a UI for the currently selected object."""

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
