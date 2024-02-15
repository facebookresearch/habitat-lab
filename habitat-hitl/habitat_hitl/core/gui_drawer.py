#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, List, Optional

import magnum as mn

from habitat_hitl.core import ClientMessageManager
from habitat_sim.gfx import DebugLineRender


class GuiDrawer:
    """
    Renders UI elements.
    """

    ID_BROADCAST: Final[int] = -1
    DEFAULT_SEGMENT_COUNT: Final[int] = 24
    DEFAULT_NORMAL: mn.Vector3 = mn.Vector3(0.0, 1.0, 0.0)

    def __init__(
        self,
        sim_debug_line_render: Optional[DebugLineRender],
        client_message_manager: Optional[ClientMessageManager],
    ) -> None:
        """
        Construct the UI drawer.
        If sim_debug_line_render is defined, uses the habitat-sim DebugLineRender to render on the server viewport.
        If client_message_manager is defined, sends render messages to the clients.
        """
        self._sim_debug_line_render = sim_debug_line_render
        self._client_message_manager = client_message_manager

    def set_line_width(
        self,
        line_width: float,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        Set global line width for all lines rendered by GuiDrawer.
        """
        if self._sim_debug_line_render:
            self._sim_debug_line_render.set_line_width(line_width)

        if self._client_message_manager:
            # Networking not implemented
            pass

    def push_transform(
        self,
        transform: mn.Matrix4,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        Push (multiply) a transform onto the transform stack, affecting all line-drawing until popped.
        Must be paired with popTransform().
        """
        if self._sim_debug_line_render:
            self._sim_debug_line_render.push_transform(transform)

        if self._client_message_manager:
            # Networking not implemented
            pass

    def pop_transform(
        self,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        See push_transform.
        """
        if self._sim_debug_line_render:
            self._sim_debug_line_render.pop_transform()

        if self._client_message_manager:
            # Networking not implemented
            pass

    def draw_box(
        self,
        min_extent: mn.Vector3,
        max_extent: mn.Vector3,
        color: mn.Color4,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        Draw a box in world-space or local-space (see pushTransform).
        """
        if self._sim_debug_line_render:
            self._sim_debug_line_render.draw_box(min, max, color)

        if self._client_message_manager:
            # Networking not implemented
            pass

    def draw_circle(
        self,
        translation: mn.Vector3,
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        Draw a circle in world-space or local-space (see pushTransform).
        The circle is an approximation; see numSegments.
        """
        if self._sim_debug_line_render:
            self._sim_debug_line_render.draw_circle(
                translation, radius, color, num_segments, normal
            )

        if self._client_message_manager:
            # Networking not implemented
            pass

    def draw_transformed_line(
        self,
        from_pos: mn.Vector3,
        to_pos: mn.Vector3,
        from_color: mn.Color4,
        to_color: mn.Color4 = None,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        Draw a line segment in world-space or local-space (see pushTransform) with interpolated color.
        Specify two colors to interpolate the line color.
        """
        if self._sim_debug_line_render:
            if to_color is None:
                self._sim_debug_line_render.draw_transformed_line(
                    from_pos, to_pos, from_color
                )
            else:
                self._sim_debug_line_render.draw_transformed_line(
                    from_pos, to_pos, from_color, to_color
                )

        if self._client_message_manager:
            # Networking not implemented
            pass

    def draw_path_with_endpoint_circles(
        self,
        points: List[mn.Vector3],
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
        destination_id: int = ID_BROADCAST,
    ) -> None:
        """
        Draw a sequence of line segments with circles at the two endpoints.
        In world-space or local-space (see pushTransform).
        """
        if self._sim_debug_line_render:
            self._sim_debug_line_render.draw_path_with_endpoint_circles(
                points, radius, color, num_segments, normal
            )

        if self._client_message_manager:
            # Networking not implemented
            pass
