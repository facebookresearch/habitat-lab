from collections import namedtuple
from os import path as osp

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import pytest
from magnum.platform.glfw import Application

import habitat
import habitat_sim
from examples.siro_sandbox.sandbox_app import (
    SandboxDriver,
    update_habitat_config,
)
from habitat.gui.gui_application import GuiApplication
from habitat.gui.gui_input import GuiInput
from habitat.gui.replay_gui_app_renderer import ReplayGuiAppRenderer

# import os
# import sys
# cwd = os.getcwd()
# sys.path.insert(0, cwd)
# print()
# print(sys.path)
# print()


DEFAULT_APP_WINDOW_WIDTH = 1280
DEFAULT_APP_WINDOW_HEIGHT = 720
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"
DEFAULT_CFG = "benchmark/rearrange/rearrange_easy_human_and_fetch.yaml"
DEFAULT_CFG_OPTS = ["habitat.dataset.split=minival"]

SandboxDriverArgs = namedtuple(
    "SandboxDriverArgs",
    [
        "cfg",
        "opts",
        "width",
        "height",
        "walk_pose_path",
        "never_end",
        "disable_inverse_kinematics",
        "humanoid_user_agent",
        "use_batch_renderer",
    ],
)


@pytest.mark.skipif(
    not habitat_sim.built_with_bullet,
    reason="Bullet physics used for validation.",
)
@pytest.mark.skipif(
    not osp.exists("data/replica_cad"),
    reason="Requires the replica_cad",
)
@pytest.mark.skipif(
    not osp.exists("data/datasets/rearrange_pick/replica_cad/v0"),
    reason="Requires the replica_cad v0 rearrange pick dataset",
)
@pytest.mark.parametrize(
    "args",
    [
        SandboxDriverArgs(
            cfg=DEFAULT_CFG,
            opts=DEFAULT_CFG_OPTS,
            width=DEFAULT_APP_WINDOW_WIDTH,
            height=DEFAULT_APP_WINDOW_HEIGHT,
            walk_pose_path=DEFAULT_POSE_PATH,
            never_end=True,
            disable_inverse_kinematics=True,
            humanoid_user_agent=True,
            use_batch_renderer=False,
        )
    ],
)
def test_sandbox_driver(args):
    # get config
    config = habitat.get_config(args.cfg, args.opts)
    config = update_habitat_config(config, args)

    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, 30)
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    # init SandboxDriver
    sim_input = GuiInput()
    driver = SandboxDriver(args, config, gui_input=sim_input)
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size.x, framebuffer_size.y, args.use_batch_renderer
    )
    driver.set_debug_line_render(
        app_renderer._replay_renderer.debug_line_render(0)
    )

    # assert post_sim_update_dict match expected result
    post_sim_update_dict = driver.sim_update(
        dt=None
    )  # dt argument is not used within sim_update
    # print(post_sim_update_dict)

    # todo: add asserts here

    # post_sim_update
    app_renderer.post_sim_update(post_sim_update_dict)

    # render keyframe to the MutableImageView2D
    # 1) doesn't work:
    buffer = np.empty(
        (
            framebuffer_size.y,
            framebuffer_size.x,
            3,
        ),
        dtype=np.uint8,
    )
    mutable_image_view = mn.MutableImageView2D(
        mn.PixelFormat.RGB8_UNORM, framebuffer_size, buffer
    )
    # app_renderer._replay_renderer.set_sensor_transform(
    #     0, app_renderer._sensor_uuid, app_renderer.cam_transform
    # )
    # AssertionError: ESP_CHECK failed: Renderer was not created with a background render thread, cannot do async drawing
    # app_renderer._replay_renderer.render([mutable_image_view])

    # 2) but reading the default_framebuffer to MutableImageView2D works
    def flip_vertical(obs):
        converted_obs = np.empty_like(obs)
        for row in range(obs.shape[0]):
            converted_obs[row, :] = obs[obs.shape[0] - row - 1, :]
        return converted_obs

    app_renderer.render_update(
        dt=None
    )  # dt argument is not used within render_update
    mn.gl.default_framebuffer.read(
        mn.Range2Di(framebuffer_size - framebuffer_size, framebuffer_size),
        mutable_image_view,
    )
    plt.imshow(flip_vertical(buffer))
    plt.show()


# test_sandbox_driver()
