#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from magnum.platform.glfw import Application

from habitat_hitl._internal.config_helper import update_config
from habitat_hitl._internal.gui_application import GuiApplication
from habitat_hitl._internal.hitl_driver import HitlDriver
from habitat_hitl._internal.replay_gui_app_renderer import ReplayGuiAppRenderer


def _parse_debug_third_person(hitl_config, framebuffer_size):
    viewport_multiplier = mn.Vector2(
        framebuffer_size.x / hitl_config.window.width,
        framebuffer_size.y / hitl_config.window.height,
    )

    do_show = hitl_config.debug_third_person_viewport.width is not None

    if do_show:
        width = hitl_config.debug_third_person_viewport.width
        # default to square aspect ratio
        height = (
            hitl_config.debug_third_person_viewport.height
            if hitl_config.debug_third_person_viewport.height is not None
            else width
        )

        width = int(width * viewport_multiplier.x)
        height = int(height * viewport_multiplier.y)
    else:
        width = 0
        height = 0

    return do_show, width, height


def hitl_main(config, create_app_state_lambda=None):
    hitl_config = config.habitat_hitl
    glfw_config = Application.Configuration()
    glfw_config.title = hitl_config.window.title
    glfw_config.size = (hitl_config.window.width, hitl_config.window.height)
    gui_app_wrapper = GuiApplication(glfw_config, hitl_config.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(hitl_config, framebuffer_size)

    viewport_rect = None
    if show_debug_third_person:
        # adjust main viewport to leave room for the debug third-person camera on the right
        assert framebuffer_size.x > debug_third_person_width
        viewport_rect = mn.Range2Di(
            mn.Vector2i(0, 0),
            mn.Vector2i(
                framebuffer_size.x - debug_third_person_width,
                framebuffer_size.y,
            ),
        )

    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size,
        viewport_rect,
        hitl_config.experimental.use_batch_renderer,
    )

    update_config(
        config,
        show_debug_third_person=show_debug_third_person,
        debug_third_person_width=debug_third_person_width,
        debug_third_person_height=debug_third_person_height,
    )

    driver = HitlDriver(
        config,
        gui_app_wrapper.get_sim_input(),
        app_renderer._replay_renderer.debug_line_render(0),
        app_renderer._text_drawer,
        create_app_state_lambda,
    )

    # sanity check if there are no agents with camera sensors
    if (
        len(config.habitat.simulator.agents) == 1
        and config.habitat_hitl.gui_controlled_agent.agent_index is not None
    ):
        assert driver.get_sim().renderer is None

    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    gui_app_wrapper.exec()

    driver.close()


if __name__ == "__main__":
    raise RuntimeError("not implemented")
    # hitl_main(None, None)
