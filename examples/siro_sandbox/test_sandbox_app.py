import os
from unittest.mock import patch

import magnum as mn
import pytest
from sandbox_app import (
    DEFAULT_CFG_PATH,
    SandboxDriver,
    get_sandbox_app_args_parser,
    override_config,
)
from utils.gui.gui_input import GuiInput
from utils.gui.replay_gui_app_renderer import ReplayGuiAppRenderer

from habitat_baselines.config.default import get_config as get_baselines_config


@pytest.mark.skipif(
    not os.path.exists("data/hab3_bench_assets"),
    reason="This test requires hab3_bench_assets assets.",
)
@patch("utils.gui.replay_gui_app_renderer.ReplayRenderer", spec=True)
@patch("utils.gui.replay_gui_app_renderer.ImageFramebufferDrawer", spec=True)
@patch("utils.gui.replay_gui_app_renderer.TextDrawer", spec=True)
def test_sandbox_driver(
    mock_replay_renderer, mock_image_framebuffer_drawer, mock_text_drawer
):
    # get default args
    sandbox_app_args_parser = get_sandbox_app_args_parser()
    default_args = sandbox_app_args_parser.parse_args(
        [
            "--disable-inverse-kinematics",
            "--gui-controlled-agent-index",
            "1",
            "--never-end",
        ]
    )
    default_args._gui_controlled_agent_index = 1

    # get default config
    config_path = DEFAULT_CFG_PATH
    config_opts = [
        "habitat_baselines.evaluate=True",
        "habitat_baselines.num_environments=1",
        "habitat_baselines.eval.should_load_ckpt=False",
        "habitat_baselines.rl.agent.num_pool_agents_per_type=[1,1]",
        "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        # use hab3_bench_assets assets
        "habitat.simulator.agents.agent_1.articulated_agent_urdf=data/hab3_bench_assets/humanoids/female_0/female_0.urdf",
        "habitat.dataset.scenes_dir=data/hab3_bench_assets/hab3-hssd/scenes",
        "habitat.dataset.data_path=data/hab3_bench_assets/episode_datasets/small_small.json.gz",
    ]
    default_config = get_baselines_config(config_path, config_opts)
    override_config(config=default_config, args=default_args)

    app_renderer = ReplayGuiAppRenderer(
        window_size=mn.Vector2i(default_args.width, default_args.height)
    )

    gui_input = GuiInput()

    sandbox_driver = SandboxDriver(
        args=default_args,
        config=default_config,
        gui_input=gui_input,
        line_render=app_renderer._replay_renderer.debug_line_render(0),
        text_drawer=app_renderer._text_drawer,
    )
    sim_dt = 1 / default_args.target_sps
    post_sim_update_dict = sandbox_driver.sim_update(dt=sim_dt)  # noqa: F841

    # TODO: assert post_sim_update_dict
