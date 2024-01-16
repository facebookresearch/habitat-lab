#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
from hb_config_helper import update_config_and_args
from magnum.platform.glfw import Application
from sandbox_app import SandboxDriver
from utils.gui.gui_application import GuiApplication
from utils.gui.replay_gui_app_renderer import ReplayGuiAppRenderer


def _parse_debug_third_person(args, framebuffer_size):
    viewport_multiplier = mn.Vector2(
        framebuffer_size.x / args.width, framebuffer_size.y / args.height
    )

    do_show = args.debug_third_person_width != 0

    width = args.debug_third_person_width
    # default to square aspect ratio
    height = (
        args.debug_third_person_height
        if args.debug_third_person_height != 0
        else width
    )

    width = int(width * viewport_multiplier.x)
    height = int(height * viewport_multiplier.y)

    return do_show, width, height


def hitl_main(args, config, create_app_state_lambda=None):
    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"  # todo: get from hydra config
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, args.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(args, framebuffer_size)

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
        args.use_batch_renderer,
    )

    update_config_and_args(
        config,
        args,
        show_debug_third_person=show_debug_third_person,
        debug_third_person_width=debug_third_person_width,
        debug_third_person_height=debug_third_person_height,
    )

    driver = SandboxDriver(
        args,
        config,
        gui_app_wrapper.get_sim_input(),
        app_renderer._replay_renderer.debug_line_render(0),
        app_renderer._text_drawer,
        create_app_state_lambda,
    )

    # sanity check if there are no agents with camera sensors
    if (
        len(config.habitat.simulator.agents) == 1
        and args.gui_controlled_agent_index is not None
    ):
        assert driver.get_sim().renderer is None

    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    gui_app_wrapper.exec()

    driver.close()


if __name__ == "__main__":
    raise RuntimeError("not implemented")
    # hitl_main(None, None)
