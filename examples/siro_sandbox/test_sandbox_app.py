import json
import os

import pytest
from sandbox_app import (
    DEFAULT_CFG_PATH,
    SandboxDriver,
    get_sandbox_app_args_parser,
    override_config,
)
from utils.gui.gui_input import GuiInput

from habitat_baselines.config.default import get_config as get_baselines_config


class DummyClass:
    def __getattr__(self, attr):
        # Define a default method that does nothing
        def dummy_method(*args, **kwargs):
            pass

        return dummy_method


@pytest.mark.skipif(
    not os.path.exists("data/hab3_bench_assets"),
    reason="This test requires hab3_bench_assets assets.",
)
def test_sandbox_driver():
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
        # fix agents position
        "habitat.seed=1856",
        "habitat.simulator.seed=1856",
        "+habitat.simulator.fixed_agent_starting_pose=True",
    ]
    default_config = get_baselines_config(config_path, config_opts)
    override_config(config=default_config, args=default_args)

    gui_input = GuiInput()

    # mock text drawer and line renderer
    dummy_text_drawer = DummyClass()
    dummy_line_renderer = DummyClass()

    sandbox_driver = SandboxDriver(
        args=default_args,
        config=default_config,
        gui_input=gui_input,
        line_render=dummy_line_renderer,
        text_drawer=dummy_text_drawer,
    )

    sim_dt = 1 / default_args.target_sps
    post_sim_update_dict = sandbox_driver.sim_update(dt=sim_dt)

    post_sim_update_keyframe_0 = post_sim_update_dict["keyframes"][0]

    # the first keyframe should create and pose at least one render asset instance
    post_sim_update_keyframe_0_obj = json.loads(post_sim_update_keyframe_0)
    assert (
        post_sim_update_keyframe_0_obj["keyframe"]
        and post_sim_update_keyframe_0_obj["keyframe"]["loads"]
        and post_sim_update_keyframe_0_obj["keyframe"]["creations"]
        and post_sim_update_keyframe_0_obj["keyframe"]["stateUpdates"]
    )

    with open("examples/siro_sandbox/test_sandbox_app_keyframe.txt", "r") as f:
        expected_keyframe = f.read()

    # if for some reason you need to update the expected keyframe, uncomment the following lines
    # with open("examples/siro_sandbox/test_sandbox_app_keyframe.txt", "w") as f:
    #     f.write(post_sim_update_keyframe_0)

    assert post_sim_update_keyframe_0 == expected_keyframe
