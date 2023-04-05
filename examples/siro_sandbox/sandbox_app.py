#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
See README.md in this directory.
"""

import ctypes

# must call this before importing habitat or magnum! avoids EGL_BAD_ACCESS error on some platforms
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import argparse

import magnum as mn
import numpy as np
from controllers import ControllerHelper
from magnum.platform.glfw import Application

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    OracleNavActionConfig,
    PddlApplyActionConfig,
)
from habitat.gui.gui_application import GuiAppDriver, GuiApplication
from habitat.gui.gui_input import GuiInput
from habitat.gui.replay_gui_app_renderer import ReplayGuiAppRenderer

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = "data/humanoids/humanoid_data/walking_motion_processed.pkl"

DEFAULT_CFG = "habitat-lab/habitat/config/benchmark/rearrange/rearrange_easy_human_and_fetch.yaml"


class SandboxDriver(GuiAppDriver):
    def __init__(self, args, config, gui_input):
        with habitat.config.read_write(config):
            # needed so we can provide keyframes to GuiApplication
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
        self.env = habitat.Env(config=config)
        self.obs = self.env.reset()

        self.ctrl_helper = ControllerHelper(self.env, args, gui_input)

        self.ctrl_helper.on_environment_reset()

        self.cam_zoom_dist = 1.0
        self.gui_input = gui_input

        self._debug_line_render = (
            None  # will be set later via a hack (see bottom of this file)
        )

    def do_raycast_and_get_walk_dir(self):
        walk_dir = None
        ray = self.gui_input.mouse_ray

        target_y = 0.15  # hard-coded to match ReplicaCAD stage floor y

        if not ray or ray.direction.y >= 0 or ray.origin.y <= target_y:
            return walk_dir

        dist_to_target_y = -ray.origin.y / ray.direction.y

        target = ray.origin + ray.direction * dist_to_target_y

        # reference code for casting a ray into the scene
        # raycast_results = self.env._sim.cast_ray(ray=ray)
        # if raycast_results.has_hits():
        #     hit_info = raycast_results.hits[0]
        #     self._debug_line_render.draw_circle(hit_info.point + mn.Vector3(0, 0.05, 0), 0.03, mn.Color3(0, 1, 0))

        agent_idx = 0
        art_obj = self.env._sim.agents_mgr[agent_idx].articulated_agent.sim_obj
        robot_root = art_obj.transformation

        pathfinder = self.env._sim.pathfinder
        snapped_pos = pathfinder.snap_point(target)
        snapped_start_pos = robot_root.translation
        snapped_start_pos.y = snapped_pos.y

        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start_pos
        path.requested_end = snapped_pos
        found_path = pathfinder.find_path(path)

        if found_path:
            path_color = mn.Color3(0, 0, 1)
            # skip rendering first point. It is at the object root, at the wrong height
            for path_i in range(0, len(path.points) - 1):
                a = mn.Vector3(path.points[path_i])
                b = mn.Vector3(path.points[path_i + 1])

                self._debug_line_render.draw_transformed_line(a, b, path_color)
                # env.sim.viz_ids[f"next_loc_{path_i}"] = env.sim.visualize_position(
                #     path.points[path_i], env.sim.viz_ids[f"next_loc_{path_i}"]
                # )

            end_pos = mn.Vector3(path.points[-1])
            self._debug_line_render.draw_circle(end_pos, 0.16, path_color)

            # if self.gui_input.get_key(GuiInput.KeyNS.B):
            if (
                self.gui_input.get_mouse_button(GuiInput.MouseNS.RIGHT)
                or self.gui_input.get_key(GuiInput.KeyNS.SPACE)
            ) and len(path.points) >= 2:
                walk_dir = mn.Vector3(path.points[1]) - mn.Vector3(
                    path.points[0]
                )

        color = mn.Color3(0, 0.5, 0) if found_path else mn.Color3(0.5, 0, 0)
        self._debug_line_render.draw_circle(target, 0.08, color)

        return walk_dir

    def sim_update(self, dt):
        # todo: pipe end_play somewhere

        walk_dir = self.do_raycast_and_get_walk_dir()
        # temp hack: inject walk_dir
        self.ctrl_helper.controllers[0]._walk_dir = walk_dir

        action, end_play, reset_ep = self.ctrl_helper.update(self.obs)

        self.obs = self.env.step(action)

        if reset_ep:
            self.obs = self.env.reset()
            self.ctrl_helper.on_environment_reset()

        post_sim_update_dict = {}

        if self.gui_input.mouse_scroll_offset != 0:
            zoom_sensitivity = 0.07
            if self.gui_input.mouse_scroll_offset < 0:
                self.cam_zoom_dist *= (
                    1.0
                    + -self.gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            else:
                self.cam_zoom_dist /= (
                    1.0 + self.gui_input.mouse_scroll_offset * zoom_sensitivity
                )
            max_zoom_dist = 50.0
            min_zoom_dist = 0.1
            self.cam_zoom_dist = mn.math.clamp(
                self.cam_zoom_dist, min_zoom_dist, max_zoom_dist
            )

        agent_idx = 0
        art_obj = self.env._sim.agents_mgr[agent_idx].articulated_agent.sim_obj
        robot_root = art_obj.transformation
        lookat = robot_root.translation + mn.Vector3(0, 1, 0)
        cam_transform = mn.Matrix4.look_at(
            lookat + mn.Vector3(0.5, 1, 0.5).normalized() * self.cam_zoom_dist,
            lookat,
            mn.Vector3(0, 1, 0),
        )
        post_sim_update_dict["cam_transform"] = cam_transform

        post_sim_update_dict[
            "keyframes"
        ] = (
            self.env._sim.gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()
        )

        def flip_vertical(obs):
            converted_obs = np.empty_like(obs)
            for row in range(obs.shape[0]):
                converted_obs[row, :] = obs[obs.shape[0] - row - 1, :]
            return converted_obs

        def depth_to_rgb(obs):
            converted_obs = np.concatenate(
                [obs * 255.0 for _ in range(3)], axis=2
            ).astype(np.uint8)
            return converted_obs

        # reference code for visualizing a camera sensor in the app GUI
        # post_sim_update_dict["debug_images"] = [
        #     flip_vertical(depth_to_rgb(self.obs["agent_1_robot_head_depth"]))
        # ]

        return post_sim_update_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-sps",
        type=int,
        default=30,
        help="Target rate to step the environment (steps per second); actual SPS may be lower depending on your hardware",
    )
    parser.add_argument(
        "--width",
        default=1280,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=720,
        type=int,
        help="Vertical resolution of the window.",
    )
    parser.add_argument(
        "--humanoid-user-agent",
        action="store_true",
        default=False,
        help="Set to true if the user-controlled agent is a humanoid. Set to false if the user-controlled agent is a robot.",
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control. Only relevant for a user-controlled *robot* agent.",
    )
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--walk-pose-path", type=str, default=DEFAULT_POSE_PATH
    )
    parser.add_argument(
        "--never-end",
        action="store_true",
        default=False,
        help="If true, make the task never end due to reaching max number of steps",
    )
    parser.add_argument(
        "--use-batch-renderer",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )

    args = parser.parse_args()

    config = habitat.get_config(args.cfg, args.opts)
    with habitat.config.read_write(config):
        env_config = config.habitat.environment
        sim_config = config.habitat.simulator
        task_config = config.habitat.task
        task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
        task_config.actions[
            "agent_1_oracle_nav_action"
        ] = OracleNavActionConfig(agent_index=1)

        if True:
            # Code below is ported from interactive_play.py. I'm not sure what it is for.
            agent_config = get_agent_config(sim_config=sim_config)
            if "composite_success" in task_config.measurements:
                task_config.measurements.composite_success.must_call_stop = (
                    False
                )
            if "rearrange_nav_to_obj_success" in task_config.measurements:
                task_config.measurements.rearrange_nav_to_obj_success.must_call_stop = (
                    False
                )
            if "force_terminate" in task_config.measurements:
                task_config.measurements.force_terminate.max_accum_force = -1.0
                task_config.measurements.force_terminate.max_instant_force = (
                    -1.0
                )

        if args.never_end:
            env_config.max_episode_steps = 0

        if not args.disable_inverse_kinematics:
            if "arm_action" not in task_config.actions:
                raise ValueError(
                    "Action space does not have any arm control so cannot add inverse kinematics. Specify the `--disable-inverse-kinematics` option"
                )
            sim_config.agents.main_agent.ik_arm_urdf = (
                "./data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
            )
            task_config.actions.arm_action.arm_controller = "ArmEEAction"

    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, args.target_sps)
    driver = SandboxDriver(args, config, gui_app_wrapper.get_sim_input())
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()
    # note this must be created after GuiApplication due to OpenGL stuff
    app_renderer = ReplayGuiAppRenderer(
        framebuffer_size.x, framebuffer_size.y, args.use_batch_renderer
    )
    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    # sloppy: provide replay app renderer's debug_line_render to our driver
    driver._debug_line_render = (
        app_renderer._replay_renderer.debug_line_render(0)
    )

    gui_app_wrapper.exec()
