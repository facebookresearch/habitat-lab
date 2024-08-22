#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final, List, Optional

import magnum as mn

from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.user_mask import Mask
from habitat_sim.gfx import DebugLineRender


class GuiDrawer:
    """
    Renders UI elements.
    """

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
        If client_message_manager is defined, sends render messages to the client.
        """
        self._sim_debug_line_render = sim_debug_line_render
        self._client_message_manager = client_message_manager

        # One local transform stack per user.
        self._local_transforms: List[List[mn.Matrix4]] = []
        if self._client_message_manager:
            users = client_message_manager._users
            for _ in range(users.max_user_count):
                self._local_transforms.append([])

    def get_sim_debug_line_render(self) -> Optional[DebugLineRender]:
        """
        Set the internal 'sim_debug_line_render' object, used for rendering lines onto the server.
        Returns None if server rendering is disabled.
        """
        return self._sim_debug_line_render

    def set_line_width(
        self,
        line_width: float,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        Set global line width for all lines rendered by GuiDrawer.
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            self._sim_debug_line_render.set_line_width(line_width)

        # If remote rendering is enabled:
        if self._client_message_manager:
            # Networking not implemented
            pass

    def push_transform(
        self,
        transform: mn.Matrix4,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        Push (multiply) a transform onto the transform stack, affecting all line-drawing until popped.
        Must be paired with popTransform().
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            self._sim_debug_line_render.push_transform(transform)

        # If remote rendering is enabled:
        if self._client_message_manager:
            for user_index in self._client_message_manager._users.indices(
                destination_mask
            ):
                self._local_transforms[user_index].append(transform)

    def pop_transform(
        self,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        See push_transform.
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            self._sim_debug_line_render.pop_transform()

        # If remote rendering is enabled:
        if self._client_message_manager:
            for user_index in self._client_message_manager._users.indices(
                destination_mask
            ):
                self._local_transforms[user_index].pop()

    def draw_box(
        self,
        min_extent: mn.Vector3,
        max_extent: mn.Vector3,
        color: mn.Color4,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        Draw a box in world-space or local-space (see pushTransform).
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            self._sim_debug_line_render.draw_box(min_extent, max_extent, color)

        # If remote rendering is enabled:
        if self._client_message_manager:

            def vec(x, y, z) -> mn.Vector3:
                return mn.Vector3(x, y, z)

            def draw_line(a: mn.Vector3, b: mn.Vector3) -> None:
                self.draw_transformed_line(
                    a, b, from_color=color, destination_mask=destination_mask
                )

            e0 = min_extent
            e1 = max_extent

            # 4 lines along x axis
            draw_line(vec(e0.x, e0.y, e0.z), vec(e1.x, e0.y, e0.z))
            draw_line(vec(e0.x, e0.y, e1.z), vec(e1.x, e0.y, e1.z))
            draw_line(vec(e0.x, e1.y, e0.z), vec(e1.x, e1.y, e0.z))
            draw_line(vec(e0.x, e1.y, e1.z), vec(e1.x, e1.y, e1.z))

            # 4 lines along y axis
            draw_line(vec(e0.x, e0.y, e0.z), vec(e0.x, e1.y, e0.z))
            draw_line(vec(e1.x, e0.y, e0.z), vec(e1.x, e1.y, e0.z))
            draw_line(vec(e0.x, e0.y, e1.z), vec(e0.x, e1.y, e1.z))
            draw_line(vec(e1.x, e0.y, e1.z), vec(e1.x, e1.y, e1.z))

            # 4 lines along z axis
            draw_line(vec(e0.x, e0.y, e0.z), vec(e0.x, e0.y, e1.z))
            draw_line(vec(e1.x, e0.y, e0.z), vec(e1.x, e0.y, e1.z))
            draw_line(vec(e0.x, e1.y, e0.z), vec(e0.x, e1.y, e1.z))
            draw_line(vec(e1.x, e1.y, e0.z), vec(e1.x, e1.y, e1.z))

    def draw_circle(
        self,
        translation: mn.Vector3,
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
        billboard: bool = False,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        Draw a circle in world-space or local-space (see pushTransform).
        The circle is an approximation; see numSegments.

        The normal is always in world-space.
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            self._sim_debug_line_render.draw_circle(
                translation, radius, color, num_segments, normal
            )

        # If remote rendering is enabled:
        if self._client_message_manager:
            # TODO: Move to client message manager.
            for user_index in self._client_message_manager._users.indices(
                destination_mask
            ):
                parent_transform = self._compute_parent_transform(user_index)
                global_translation = parent_transform.transform_point(
                    translation
                )

                self._client_message_manager.add_highlight(
                    pos=_vec_to_list(global_translation),
                    radius=radius,
                    normal=_vec_to_list(normal),
                    billboard=billboard,
                    color=color,
                    destination_mask=Mask.from_index(user_index),
                )

    def draw_transformed_line(
        self,
        from_pos: mn.Vector3,
        to_pos: mn.Vector3,
        from_color: mn.Color4,
        to_color: mn.Color4 = None,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        Draw a line segment in world-space or local-space (see pushTransform) with interpolated color.
        Specify two colors to interpolate the line color.
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            if to_color is None:
                self._sim_debug_line_render.draw_transformed_line(
                    from_pos, to_pos, from_color
                )
            else:
                self._sim_debug_line_render.draw_transformed_line(
                    from_pos, to_pos, from_color, to_color
                )

        # If remote rendering is enabled:
        if self._client_message_manager:
            # TODO: Move to client message manager.
            for user_index in self._client_message_manager._users.indices(
                destination_mask
            ):
                parent_transform = self._compute_parent_transform(user_index)
                global_from_pos = parent_transform.transform_point(from_pos)
                global_to_pos = parent_transform.transform_point(to_pos)

                self._client_message_manager.add_line(
                    _vec_to_list(global_from_pos),
                    _vec_to_list(global_to_pos),
                    from_color=from_color,
                    to_color=to_color,
                    destination_mask=Mask.from_index(user_index),
                )

    def draw_path_with_endpoint_circles(
        self,
        points: List[mn.Vector3],
        radius: float,
        color: mn.Color4,
        num_segments: int = DEFAULT_SEGMENT_COUNT,
        normal: mn.Vector3 = DEFAULT_NORMAL,
        destination_mask: Mask = Mask.ALL,
    ) -> None:
        """
        Draw a sequence of line segments with circles at the two endpoints.
        In world-space or local-space (see pushTransform).
        """
        # If server rendering is enabled:
        if self._sim_debug_line_render:
            self._sim_debug_line_render.draw_path_with_endpoint_circles(
                points, radius, color, num_segments, normal
            )

        # If remote rendering is enabled:
        if self._client_message_manager:
            # Networking not implemented
            pass

    def _compute_parent_transform(self, user_index: int) -> mn.Matrix4:
        """
        Resolve the transform resulting from the push/pop_transform calls.
        To apply to a point, use {ret_val}.transform_point(from_pos).
        """
        assert user_index < len(self._local_transforms)
        parent_transform = mn.Matrix4.identity_init()
        for local_transform in self._local_transforms[user_index]:
            parent_transform = parent_transform @ local_transform
        return parent_transform


def _vec_to_list(vec: mn.Vector3) -> List[float]:
    return [vec.x, vec.y, vec.z]
