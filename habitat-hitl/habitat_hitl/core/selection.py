#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import magnum as mn

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import MouseButton
from habitat_sim.geo import Ray
from habitat_sim.physics import RayHitInfo


class Selection:
    """
    Class that handles selection by tracking a given GuiInput.
    """

    def hover_fn(_gui_input: GuiInput) -> bool:  # type: ignore
        """Select the object under the cursor every frame."""
        return True

    def left_click_fn(_gui_input: GuiInput) -> bool:  # type: ignore
        """Select the object under the cursor when left clicking."""
        return _gui_input.get_mouse_button_down(MouseButton.LEFT)

    def right_click_fn(_gui_input: GuiInput) -> bool:  # type: ignore
        """Select the object under the cursor when right clicking."""
        return _gui_input.get_mouse_button_down(MouseButton.RIGHT)

    def default_discriminator(_object_id: int) -> bool:  # type: ignore
        """Pick any object ID."""
        return True

    def __init__(
        self,
        simulator: HabitatSim,
        gui_input: GuiInput,
        selection_fn: Callable[[GuiInput], bool],
        object_id_discriminator: Callable[[int], bool] = default_discriminator,
    ):
        """
        :param simulator: Simulator that is raycast upon.
        :param gui_input: GuiInput to track.
        :param selection_fn: Function that returns true if gui_input is attempting selection.
        :param object_id_discriminator: Function that determines whether an object ID is selectable.
                                        Rejected objects are transparent to selection.
                                        By default, all objects are selectable.
        """
        self._sim = simulator
        self._gui_input = gui_input
        self._discriminator = object_id_discriminator
        self._selection_fn = selection_fn

        self._selected = False
        self._object_id: Optional[int] = None
        self._point: Optional[mn.Vector3] = None
        self._normal: Optional[mn.Vector3] = None

    @property
    def selected(self) -> bool:
        """Returns true if something is selected."""
        return self._selected

    @property
    def object_id(self) -> Optional[int]:
        """Currently selected object ID."""
        return self._object_id

    @property
    def point(self) -> Optional[mn.Vector3]:
        """Point of the currently selected location."""
        return self._point

    @property
    def normal(self) -> Optional[mn.Vector3]:
        """Normal at the currently selected location."""
        return self._normal

    def deselect(self) -> None:
        """Clear selection."""
        self._selected = False
        self._object_id = None
        self._point = None
        self._normal = None

    def update(self) -> None:
        """Update selection."""
        if self._selection_fn(self._gui_input):
            ray = self._gui_input.mouse_ray
            if ray is not None:
                hit_info = self._raycast(ray)
                if hit_info is None:
                    self.deselect()
                    return

                self._selected = True
                self._object_id = hit_info.object_id
                self._point = hit_info.point
                self._normal = hit_info.normal
            else:
                self.deselect()

    def _raycast(self, ray: Ray) -> Optional[RayHitInfo]:
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
            if not self._discriminator(object_id):
                continue
            else:
                return hit

        return None
