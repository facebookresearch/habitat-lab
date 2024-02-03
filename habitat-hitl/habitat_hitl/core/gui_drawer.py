#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import Final, List

import magnum as mn

from habitat_sim.gfx import DebugLineRender

DEFAULT_SEGMENT_COUNT: Final[int] = 24
DEFAULT_NORMAL: mn.Vector3 = mn.Vector3(0.0, 1.0, 0.0)


class AbstractGuiDrawer(ABC):
    def set_line_width(
        self,
        line_width: float,
    ) -> None:
        """Set global line width for all lines rendered by GuiDrawer."""

    def push_transform(
        self,
        transform: mn.Matrix4,
    ) -> None:
        """
        Push (multiply) a transform onto the transform stack, affecting all line-drawing until popped. Must be paired with popTransform().
        """

    def pop_transform(
        self,
    ) -> None:
        """See push_transform."""

    def draw_box(
        self,
        min_extent: mn.Vector3,
        max_extent: mn.Vector3,
        color: mn.Color4,
    ) -> None:
        """Draw a box in world-space or local-space (see pushTransform)."""

    def draw_circle(
        self,
        translation: mn.Vector3,
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
    ) -> None:
        """Draw a circle in world-space or local-space (see pushTransform). The circle is an approximation; see numSegments."""

    def draw_transformed_line(
        self,
        from_pos: mn.Vector3,
        to_pos: mn.Vector3,
        from_color: mn.Color4,
        to_color: mn.Color4,
    ) -> None:
        """Draw a line segment in world-space or local-space (see pushTransform) with interpolated color. Specify two colors to interpolate the line color."""

    def draw_path_with_endpoint_circles(
        self,
        points: List[mn.Vector3],
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
    ) -> None:
        """Draw a sequence of line segments with circles at the two endpoints. In world-space or local-space (see pushTransform)."""


class StubGuiDrawer(AbstractGuiDrawer):
    """
    Stub GuiDrawer class. Has no effect but allows user code to run without error.

    This is intended for use with habitat_hitl.headless. See also GuiDrawer.
    """


class GuiDrawer(AbstractGuiDrawer):
    """
    Renders UI elements using the specified habitat-sim DebugLineRender.
    """

    def __init__(self, sim_debug_line_render: DebugLineRender) -> None:
        self._sim_debug_line_render: DebugLineRender = sim_debug_line_render

    def set_line_width(
        self,
        line_width: float,
    ) -> None:
        self._sim_debug_line_render.set_line_width(line_width)

    def push_transform(
        self,
        transform: mn.Matrix4,
    ) -> None:
        self._sim_debug_line_render.push_transform(transform)

    def pop_transform(
        self,
    ) -> None:
        self._sim_debug_line_render.pop_transform()

    def draw_box(
        self,
        min_extent: mn.Vector3,
        max_extent: mn.Vector3,
        color: mn.Color4,
    ) -> None:
        self._sim_debug_line_render.draw_box(min, max, color)

    def draw_circle(
        self,
        translation: mn.Vector3,
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
    ) -> None:
        self._sim_debug_line_render.draw_circle(
            translation, radius, color, num_segments, normal
        )

    def draw_transformed_line(
        self,
        from_pos: mn.Vector3,
        to_pos: mn.Vector3,
        from_color: mn.Color4,
        to_color: mn.Color4 = None,
    ) -> None:
        if to_color is None:
            self._sim_debug_line_render.draw_transformed_line(
                from_pos, to_pos, from_color
            )
        else:
            self._sim_debug_line_render.draw_transformed_line(
                from_pos, to_pos, from_color, to_color
            )

    def draw_path_with_endpoint_circles(
        self,
        points: List[mn.Vector3],
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
    ) -> None:
        self._sim_debug_line_render.draw_path_with_endpoint_circles(
            points, radius, color, num_segments, normal
        )
