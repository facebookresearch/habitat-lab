#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import magnum as mn

from habitat.config.default import patch_config
from habitat_hitl._internal.config_helper import update_config
from habitat_hitl._internal.hitl_driver import HitlDriver
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl._internal.networking.frequency_limiter import (
    FrequencyLimiter,
)
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hydra_utils import omegaconf_to_object


def _parse_debug_third_person(hitl_config, viewport_multiplier=(1, 1)):
    assert viewport_multiplier[0] > 0 and viewport_multiplier[1] > 0

    do_show = hitl_config.debug_third_person_viewport.width is not None

    if do_show:
        width = hitl_config.debug_third_person_viewport.width
        # default to square aspect ratio
        height = (
            hitl_config.debug_third_person_viewport.height
            if hitl_config.debug_third_person_viewport.height is not None
            else width
        )

        width = int(width * viewport_multiplier[0])
        height = int(height * viewport_multiplier[1])
    else:
        width = 0
        height = 0

    return do_show, width, height


def hitl_main(app_config, create_app_state_lambda=None):
    # Add check for data/
    if not os.path.exists("data/"):
        raise FileNotFoundError(
            "Could not find /data in current directory. "
            "HITL apps expect 'data/' directory to exist. "
            "Either run from habitat-lab directory or symlink data/ folder to your HITL app working directory"
        )

    app_config = patch_config(app_config)

    hitl_config = omegaconf_to_object(app_config.habitat_hitl)

    if hitl_config.experimental.headless.do_headless:
        hitl_headless_main(hitl_config, app_config, create_app_state_lambda)
    else:
        hitl_headed_main(hitl_config, app_config, create_app_state_lambda)


def hitl_headed_main(hitl_config, app_config, create_app_state_lambda):
    from magnum.platform.glfw import Application

    from habitat_hitl._internal.gui_application import GuiApplication
    from habitat_hitl._internal.replay_gui_app_renderer import (
        ReplayGuiAppRenderer,
    )

    assert hitl_config.window.width > 0
    assert hitl_config.window.height > 0

    glfw_config = Application.Configuration()
    glfw_config.title = hitl_config.window.title
    glfw_config.size = (hitl_config.window.width, hitl_config.window.height)
    gui_app_wrapper = GuiApplication(glfw_config, hitl_config.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    viewport_multiplier = (
        framebuffer_size.x / hitl_config.window.width,
        framebuffer_size.y / hitl_config.window.height,
    )

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(hitl_config, viewport_multiplier)

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

    text_drawer_kwargs = {
        "display_font_size": hitl_config.window.display_font_size,
        "relative_path_to_font": hitl_config.window.display_font_path,
    }

    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size,
        viewport_rect,
        hitl_config.experimental.use_batch_renderer,
        text_drawer_kwargs=text_drawer_kwargs,
    )

    update_config(
        app_config,
        show_debug_third_person=show_debug_third_person,
        debug_third_person_width=debug_third_person_width,
        debug_third_person_height=debug_third_person_height,
    )

    debug_line_drawer = app_renderer._replay_renderer.debug_line_render(0)

    driver = HitlDriver(
        config=app_config,
        gui_input=gui_app_wrapper.get_sim_input(),
        debug_line_drawer=debug_line_drawer,
        text_drawer=app_renderer._text_drawer,
        create_app_state_lambda=create_app_state_lambda,
    )

    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    gui_app_wrapper.exec()

    driver.close()


def _headless_app_loop(hitl_config, driver):
    headless_config = hitl_config.experimental.headless
    frequency_limiter = FrequencyLimiter(hitl_config.target_sps)
    sps_rate_tracker = AverageRateTracker(1.0)
    dt = 1.0 / hitl_config.target_sps

    step_count = 0
    next_step_count_to_debug_print = 100

    video_config = headless_config.debug_video_writer
    video_helper = None
    if video_config.num_frames_to_save > 0:
        from habitat_hitl.core.debug_video_writer import DebugVideoWriter

        video_helper = DebugVideoWriter()

    while True:
        # Print step_count periodically. print less often as time goes by, to cut down on log spam.
        if step_count == next_step_count_to_debug_print:
            print(f"step {step_count}")
            next_step_count_to_debug_print = next_step_count_to_debug_print * 2

        post_sim_update_dict = driver.sim_update(dt)

        if "application_exit" in post_sim_update_dict:
            break

        if step_count < video_config.num_frames_to_save:
            video_helper.add_frame(post_sim_update_dict["debug_images"])
            if step_count == video_config.num_frames_to_save - 1:
                video_helper.write(video_config.filepath_base)
                video_helper = None

        frequency_limiter.limit_frequency()

        new_rate = sps_rate_tracker.increment()
        if new_rate is not None:
            low_sps_warning_threshold = hitl_config.target_sps * 0.9
            if new_rate < low_sps_warning_threshold:
                print(f"low SPS: {new_rate:.1f}")

        step_count += 1
        if (
            headless_config.exit_after is not None
            and step_count == headless_config.exit_after
        ):
            break

    if video_helper:
        video_helper.write(video_config.filepath_base)

    driver.close()


def hitl_headless_main(hitl_config, config, create_app_state_lambda=None):
    from habitat_hitl.core.text_drawer import HeadlessTextDrawer

    if hitl_config.window is not None:
        raise ValueError(
            "For habitat_hitl.headless.do_headless=True, habitat_hitl.window should be None."
        )

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = _parse_debug_third_person(hitl_config)

    update_config(
        config,
        show_debug_third_person=show_debug_third_person,
        debug_third_person_width=debug_third_person_width,
        debug_third_person_height=debug_third_person_height,
    )

    driver = HitlDriver(
        config=config,
        gui_input=GuiInput(),
        debug_line_drawer=None,
        text_drawer=HeadlessTextDrawer(),
        create_app_state_lambda=create_app_state_lambda,
    )

    _headless_app_loop(hitl_config, driver)

    driver.close()
