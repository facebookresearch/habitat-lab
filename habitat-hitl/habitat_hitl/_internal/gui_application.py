#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import time

import magnum as mn
from magnum.platform.glfw import Application

from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import MagnumKeyConverter


class GuiAppRenderer:
    @abc.abstractmethod
    def post_sim_update(self, keyframe):
        pass

    @abc.abstractmethod
    def render_update(self, dt):
        return False

    @abc.abstractmethod
    def unproject(self, viewport_pos):
        pass


class InputHandlerApplication(Application):
    def __init__(self, config):
        super().__init__(config)
        self._gui_input = GuiInput()
        self._mouse_ray = None

    def key_press_event(self, event: Application.KeyEvent) -> None:
        key = MagnumKeyConverter.convert_key(event.key)
        if key is not None:
            gui_input = self._gui_input
            if key not in self._gui_input._key_held:
                gui_input._key_held.add(key)
                gui_input._key_down.add(key)

    def key_release_event(self, event: Application.KeyEvent) -> None:
        key = MagnumKeyConverter.convert_key(event.key)
        if key is not None:
            gui_input = self._gui_input
            if key in gui_input._key_held:
                gui_input._key_held.remove(key)
            gui_input._key_up.add(key)

    def pointer_press_event(self, event: Application.PointerEvent) -> None:
        key = MagnumKeyConverter.convert_mouse_button(event.pointer)
        if key is not None:
            gui_input = self._gui_input
            # If the key is already held, this is a repeat press event and we should
            # ignore it.
            if key not in gui_input._mouse_button_held:
                gui_input._mouse_button_held.add(key)
                gui_input._mouse_button_down.add(key)

    def pointer_release_event(self, event: Application.PointerEvent) -> None:
        key = MagnumKeyConverter.convert_mouse_button(event.pointer)
        if key is not None:
            gui_input = self._gui_input
            if key in gui_input._mouse_button_held:
                gui_input._mouse_button_held.remove(key)
            gui_input._mouse_button_up.add(key)

    def scroll_event(self, event: Application.PointerEvent) -> None:
        # shift+scroll is forced into x direction on mac, seemingly at OS level,
        # so use both x and y offsets.
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )

        # accumulate
        self._gui_input._mouse_scroll_offset += scroll_mod_val

    def get_mouse_position(
        self, mouse_event_position: mn.Vector2i
    ) -> mn.Vector2i:
        """
        This function will get a screen-space mouse position appropriately
        scaled based on framebuffer size and window size.  Generally these would be
        the same value, but on certain HiDPI displays (Retina displays) they may be
        different.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(
            self.window_size
        )
        return mouse_event_position * scaling

    def pointer_move_event(self, event: Application.PointerMoveEvent) -> None:
        mouse_pos = self.get_mouse_position(event.position)
        relative_mouse_position = self.get_mouse_position(
            event.relative_position
        )
        gui_input = self._gui_input
        gui_input._mouse_position = mouse_pos
        gui_input._relative_mouse_position[0] += relative_mouse_position[0]
        gui_input._relative_mouse_position[1] += relative_mouse_position[1]
        if self._mouse_ray:
            gui_input._mouse_ray = self._mouse_ray

    def update_mouse_ray(self, unproject_fn):
        self._mouse_ray = unproject_fn(self._gui_input._mouse_position)


class GuiApplication(InputHandlerApplication):
    def __init__(self, glfw_config, target_sps):
        super().__init__(glfw_config)

        self._driver = None
        self._app_renderer = None
        self._sim_time = None
        self._target_sps = target_sps
        self._debug_sps = 0.0

    def get_sim_input(self):
        return self._gui_input

    def set_driver_and_renderer(self, driver, app_renderer):
        assert isinstance(app_renderer, GuiAppRenderer)
        self._driver = driver
        self._app_renderer = app_renderer

    def get_framebuffer_size(self):
        return self.framebuffer_size

    def _post_sim_update(self, post_sim_update_dict):
        if "application_cursor" in post_sim_update_dict:
            self.cursor = post_sim_update_dict["application_cursor"]

        if "application_exit" in post_sim_update_dict:
            self.exit(0)

    def draw_event(self):
        # tradeoff between responsiveness and simulation speed
        max_sim_updates_per_render = 1

        sim_dt = 1 / self._target_sps

        curr_time = time.time()
        if self._sim_time is None:
            num_sim_updates = 1
            self._sim_time = curr_time
            self._last_draw_event_time = curr_time
            self._debug_counter = 0
            self._debug_timer = curr_time
        else:
            elapsed_since_last_sim_update = curr_time - self._sim_time
            num_sim_updates = int(
                math.floor(elapsed_since_last_sim_update / sim_dt)
            )
            num_sim_updates = min(num_sim_updates, max_sim_updates_per_render)
            self._debug_counter += num_sim_updates
            self._sim_time += sim_dt * num_sim_updates

            # don't let sim time fall too far behind
            if (
                curr_time - self._sim_time
                > sim_dt * max_sim_updates_per_render
            ):
                self._sim_time = (
                    curr_time - sim_dt * max_sim_updates_per_render
                )

            self._last_draw_event_time = curr_time
            if self._debug_counter >= 10:
                elapsed = curr_time - self._debug_timer
                self._debug_sps = self._debug_counter / elapsed
                self._debug_timer = curr_time
                self._debug_counter = 0

        for _ in range(num_sim_updates):
            post_sim_update_dict = self._driver.sim_update(sim_dt)
            self._gui_input.reset()
            self._post_sim_update(post_sim_update_dict)
            if "application_exit" in post_sim_update_dict:
                return
            self._app_renderer.post_sim_update(post_sim_update_dict)

        render_dt = 1 / 60.0  # todo: drive correctly
        did_render = self._app_renderer.render_update(render_dt)

        # todo: also update when mouse moves
        self.update_mouse_ray(self._app_renderer.unproject)

        # app_renderer should have rendered to mn.gl.default_framebuffer
        if did_render:
            self.swap_buffers()
        else:
            # Nothing was rendered, which suggests we have some time to kill. Sleeping
            # here may lower the app CPU usage.
            time.sleep(0)

        # request redraw continuously
        self.redraw()
