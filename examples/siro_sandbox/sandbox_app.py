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
from functools import wraps
from typing import Any, Dict, List

import magnum as mn
import numpy as np
from controllers import ControllerHelper, GuiHumanoidController
from magnum.platform.glfw import Application

import habitat
import habitat.tasks.rearrange.rearrange_task
import habitat_sim
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    PddlApplyActionConfig,
    ThirdRGBSensorConfig,
)
from habitat.gui.gui_application import GuiAppDriver, GuiApplication
from habitat.gui.gui_input import GuiInput
from habitat.gui.replay_gui_app_renderer import ReplayGuiAppRenderer
from habitat.gui.text_drawer import TextOnScreenAlignment
from habitat_baselines.config.default import get_config as get_baselines_config

# Please reach out to the paper authors to obtain this file
DEFAULT_POSE_PATH = (
    "data/humanoids/humanoid_data/walking_motion_processed_smplx.pkl"
)

DEFAULT_CFG = "benchmark/rearrange/rearrange_easy_human_and_fetch.yaml"


def requires_habitat_sim_with_bullet(callable_):
    @wraps(callable_)
    def wrapper(*args, **kwds):
        import habitat_sim

        assert (
            habitat_sim.built_with_bullet
        ), f"Habitat-sim is built without bullet, but {callable_.__name__} requires Habitat-sim with bullet."
        return callable_(*args, **kwds)

    return wrapper


def get_pretty_object_name_from_handle(obj_handle_str):
    handle_lower = obj_handle_str.lower()
    filtered_str = "".join(filter(lambda c: c.isalpha(), handle_lower))
    return filtered_str


@requires_habitat_sim_with_bullet
class SandboxDriver(GuiAppDriver):
    def __init__(self, args, config, gui_input):
        with habitat.config.read_write(config):
            # needed so we can provide keyframes to GuiApplication
            config.habitat.simulator.habitat_sim_v0.enable_gfx_replay_save = (
                True
            )
            config.habitat.simulator.concur_render = False

        self.env = habitat.Env(config=config)
        if args.gui_controlled_agent_index is not None:
            sim_config = config.habitat.simulator
            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            oracle_nav_sensor_key = f"{gui_agent_key}_has_finished_oracle_nav"
            if oracle_nav_sensor_key in self.env.task.sensor_suite.sensors:
                del self.env.task.sensor_suite.sensors[oracle_nav_sensor_key]

        self.ctrl_helper = ControllerHelper(self.env, config, args, gui_input)

        self.gui_agent_ctrl = self.ctrl_helper.get_gui_agent_controller()

        self.cam_transform = None
        self.cam_zoom_dist = 1.0
        self._max_zoom_dist = 50.0
        self._min_zoom_dist = 0.02

        self.gui_input = gui_input

        self._debug_line_render = None
        self._debug_images = args.debug_images

        self._viz_anim_fraction = 0.0

        self.lookat = None

        if self.is_free_camera_mode() and args.first_person_mode:
            raise RuntimeError(
                "--first-person-mode must be used with --gui-controlled-agent-index"
            )

        # lookat offset yaw (spin left/right) and pitch (up/down)
        # to enable camera rotation and pitch control
        self._first_person_mode = args.first_person_mode
        if self._first_person_mode:
            self._lookat_offset_yaw = 0.0
            self._lookat_offset_pitch = float(
                mn.Rad(mn.Deg(20.0))
            )  # look slightly down
            self._min_lookat_offset_pitch = (
                -max(min(np.radians(args.max_look_up_angle), np.pi / 2), 0)
                + 1e-5
            )
            self._max_lookat_offset_pitch = (
                -min(max(np.radians(args.min_look_down_angle), -np.pi / 2), 0)
                - 1e-5
            )
        else:
            # (computed from previously hardcoded mn.Vector3(0.5, 1, 0.5).normalized())
            self._lookat_offset_yaw = 0.785
            self._lookat_offset_pitch = 0.955
            self._min_lookat_offset_pitch = -np.pi / 2 + 1e-5
            self._max_lookat_offset_pitch = np.pi / 2 - 1e-5

        self._enable_gfx_replay_save: bool = args.enable_gfx_replay_save
        self._gfx_replay_save_path: str = args.gfx_replay_save_path
        self._recording_keyframes: List[str] = []

        self._cursor_style = None
        self._can_grasp_place_threshold = args.can_grasp_place_threshold

        # not sure if these are needed here for linting
        # self._held_target_obj_idx = None
        # self._num_remaining_objects = None
        # self._num_busy_objects = None
        # self._target_obj_ids = None
        # self._goal_positions = None

        self._env_reset()

    def _env_reset(self):
        self._obs = self.env.reset()
        self._metrics = self.env.get_metrics()
        self._on_environment_reset()

    def _env_step(self, action):
        self._obs = self.env.step(action)
        self._metrics = self.env.get_metrics()

    def _env_episode_active(self) -> bool:
        """
        Returns True if current episode is active:
        1) not self.env.episode_over - none of the constraints is violated, or
        2) not self._env_task_complete - success measure value is not True
        """
        return not (self.env.episode_over or self._env_task_complete)

    def _compute_action_and_step_env(self):
        # step env if episode is active
        # otherwise pause simulation (don't do anything)
        if self._env_episode_active():
            action = self.ctrl_helper.update(self._obs)
            self._env_step(action)

    def _on_environment_reset(self):
        self.ctrl_helper.on_environment_reset()
        self._held_target_obj_idx = None
        self._num_remaining_objects = None  # resting, not at goal location yet
        self._num_busy_objects = None  # currently held by non-gui agents

        sim = self.get_sim()
        temp_ids, goal_positions_np = sim.get_targets()
        self._target_obj_ids = [
            sim._scene_obj_ids[temp_id] for temp_id in temp_ids
        ]
        self._goal_positions = [mn.Vector3(pos) for pos in goal_positions_np]

    @property
    def _env_task_complete(self):
        config = self.env._config
        success_measure_name = config.task.success_measure
        episode_success = self._metrics[success_measure_name]
        return config.task.end_on_success and episode_success

    @property
    def lookat_offset_yaw(self):
        return self._to_zero_2pi_range(self._lookat_offset_yaw)

    @property
    def lookat_offset_pitch(self):
        return self._lookat_offset_pitch

    def set_debug_line_render(self, debug_line_render):
        self._debug_line_render = debug_line_render
        self._debug_line_render.set_line_width(3)

    def set_text_drawer(self, text_drawer):
        self._text_drawer = text_drawer

    def is_free_camera_mode(self):
        return self.ctrl_helper.get_gui_controlled_agent_index() is None

    # trying to get around mypy complaints about missing sim attributes
    def get_sim(self) -> Any:
        return self.env.task._sim

    def _draw_nav_hint_from_agent(self, end_pos, end_radius, color):
        agent_idx = self.ctrl_helper.get_gui_controlled_agent_index()
        assert agent_idx is not None
        art_obj = (
            self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        agent_pos = art_obj.transformation.translation
        # get forward_dir from FPS camera yaw, not art_obj.transformation
        # (the problem with art_obj.transformation is that it includes a "wobble"
        # introduced by the walk animation)
        transformation = self.cam_transform or art_obj.transformation
        forward_dir = transformation.transform_vector(-mn.Vector3(0, 0, 1))
        forward_dir[1] = 0
        forward_dir = forward_dir.normalized()

        self._draw_nav_hint(
            agent_pos,
            forward_dir,
            end_pos,
            end_radius,
            color,
            self._viz_anim_fraction,
        )

    def _get_target_object_position(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id).translation

    def _update_grasping_and_set_act_hints(self):
        if self.is_free_camera_mode():
            return None

        end_radius = self.env._config.task.obj_succ_thresh

        drop_pos = None
        grasp_object_id = None

        if self._held_target_obj_idx is not None:
            color = mn.Color3(0, 255 / 255, 0)  # green
            goal_position = self._goal_positions[self._held_target_obj_idx]
            self._debug_line_render.draw_circle(
                goal_position, end_radius, color, 24
            )

            self._draw_nav_hint_from_agent(
                mn.Vector3(goal_position), end_radius, color
            )
            # draw can place area
            can_place_position = mn.Vector3(goal_position)
            can_place_position[1] = self._get_agent_feet_height()
            self._debug_line_render.draw_circle(
                can_place_position,
                self._can_grasp_place_threshold,
                mn.Color3(255 / 255, 255 / 255, 0),
                24,
            )

            if self.gui_input.get_key_down(GuiInput.KeyNS.SPACE):
                translation = self._get_agent_translation()
                dist_to_obj = np.linalg.norm(goal_position - translation)
                if dist_to_obj < self._can_grasp_place_threshold:
                    self._held_target_obj_idx = None
                    drop_pos = goal_position
        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:
                assert not self.gui_agent_ctrl.is_grasped
                # pick up an object
                if self.gui_input.get_key_down(GuiInput.KeyNS.SPACE):
                    translation = self._get_agent_translation()

                    min_dist = self._can_grasp_place_threshold
                    min_i = None
                    for i in range(len(self._target_obj_ids)):
                        if self._is_target_object_at_goal_position(i):
                            continue

                        this_target_pos = self._get_target_object_position(i)
                        # compute distance in xz plane
                        offset = this_target_pos - translation
                        offset.y = 0
                        dist_xz = offset.length()
                        if dist_xz < min_dist:
                            min_dist = dist_xz
                            min_i = i

                    if min_i is not None:
                        self._held_target_obj_idx = min_i
                        grasp_object_id = self._target_obj_ids[
                            self._held_target_obj_idx
                        ]

        walk_dir = (
            self._viz_and_get_humanoid_walk_dir()
            if not self._first_person_mode
            else None
        )

        self.gui_agent_ctrl.set_act_hints(
            walk_dir, grasp_object_id, drop_pos, self.lookat_offset_yaw
        )

        return drop_pos

    def _is_target_object_at_goal_position(self, target_obj_idx):
        this_target_pos = self._get_target_object_position(target_obj_idx)
        end_radius = self.env._config.task.obj_succ_thresh
        return (
            this_target_pos - self._goal_positions[target_obj_idx]
        ).length() < end_radius

    def _update_task(self):
        end_radius = self.env._config.task.obj_succ_thresh

        grasped_objects_idxs = self._get_grasped_objects_idxs()
        self._num_remaining_objects = 0
        self._num_busy_objects = len(grasped_objects_idxs)

        # draw nav_hint and target box
        for i in range(len(self._target_obj_ids)):
            # object is grasped
            if i in grasped_objects_idxs:
                continue

            color = mn.Color3(255 / 255, 128 / 255, 0)  # orange
            if self._is_target_object_at_goal_position(i):
                continue

            self._num_remaining_objects += 1

            if self._held_target_obj_idx is None:
                this_target_pos = self._get_target_object_position(i)
                box_half_size = 0.15
                box_offset = mn.Vector3(
                    box_half_size, box_half_size, box_half_size
                )
                self._debug_line_render.draw_box(
                    this_target_pos - box_offset,
                    this_target_pos + box_offset,
                    color,
                )

                if not self.is_free_camera_mode():
                    self._draw_nav_hint_from_agent(
                        mn.Vector3(this_target_pos), end_radius, color
                    )
                    # draw can grasp area
                    can_grasp_position = mn.Vector3(this_target_pos)
                    can_grasp_position[1] = self._get_agent_feet_height()
                    self._debug_line_render.draw_circle(
                        can_grasp_position,
                        self._can_grasp_place_threshold,
                        mn.Color3(255 / 255, 255 / 255, 0),
                        24,
                    )

    def _get_grasped_objects_idxs(self):
        sim = self.get_sim()
        agents_mgr = sim.agents_mgr

        grasped_objects_idxs = []
        for agent_idx in range(self.ctrl_helper.n_robots):
            if agent_idx == self.ctrl_helper.get_gui_controlled_agent_index():
                continue
            grasp_mgr = agents_mgr._all_agent_data[agent_idx].grasp_mgr
            if grasp_mgr.is_grasped:
                grasped_objects_idxs.append(
                    sim.scene_obj_ids.index(grasp_mgr.snap_idx)
                )

        return grasped_objects_idxs

    def _get_agent_translation(self):
        assert isinstance(self.gui_agent_ctrl, GuiHumanoidController)
        return (
            self.gui_agent_ctrl._humanoid_controller.obj_transform_base.translation
        )

    def _get_agent_feet_height(self):
        assert isinstance(self.gui_agent_ctrl, GuiHumanoidController)
        base_offset = (
            self.gui_agent_ctrl.get_articulated_agent().params.base_offset
        )
        agent_feet_translation = self._get_agent_translation() + base_offset
        return agent_feet_translation[1]

    def _viz_and_get_humanoid_walk_dir(self):
        path_color = mn.Color3(0, 153 / 255, 255 / 255)
        path_endpoint_radius = 0.12

        ray = self.gui_input.mouse_ray

        floor_y = 0.15  # hardcoded to ReplicaCAD

        if not ray or ray.direction.y >= 0 or ray.origin.y <= floor_y:
            return None

        dist_to_floor_y = (ray.origin.y - floor_y) / -ray.direction.y
        target_on_floor = ray.origin + ray.direction * dist_to_floor_y

        agent_idx = self.ctrl_helper.get_gui_controlled_agent_index()
        art_obj = (
            self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
        )
        robot_root = art_obj.transformation

        pathfinder = self.get_sim().pathfinder
        snapped_pos = pathfinder.snap_point(target_on_floor)
        snapped_start_pos = robot_root.translation
        snapped_start_pos.y = snapped_pos.y

        path = habitat_sim.ShortestPath()
        path.requested_start = snapped_start_pos
        path.requested_end = snapped_pos
        found_path = pathfinder.find_path(path)

        if not found_path or len(path.points) < 2:
            return None

        path_points = []
        for path_i in range(0, len(path.points)):
            adjusted_point = mn.Vector3(path.points[path_i])
            # first point in path is at wrong height
            if path_i == 0:
                adjusted_point.y = mn.Vector3(path.points[path_i + 1]).y
            path_points.append(adjusted_point)

        self._debug_line_render.draw_path_with_endpoint_circles(
            path_points, path_endpoint_radius, path_color
        )

        if (self.gui_input.get_mouse_button(GuiInput.MouseNS.RIGHT)) and len(
            path.points
        ) >= 2:
            walk_dir = mn.Vector3(path.points[1]) - mn.Vector3(path.points[0])
            return walk_dir

        return None

    def _camera_pitch_and_yaw_wasd_control(self):
        # update yaw and pitch using ADIK keys
        cam_rot_angle = 0.1

        if self.gui_input.get_key(GuiInput.KeyNS.I):
            self._lookat_offset_pitch -= cam_rot_angle
        if self.gui_input.get_key(GuiInput.KeyNS.K):
            self._lookat_offset_pitch += cam_rot_angle
        self._lookat_offset_pitch = np.clip(
            self._lookat_offset_pitch,
            self._min_lookat_offset_pitch,
            self._max_lookat_offset_pitch,
        )
        if self.gui_input.get_key(GuiInput.KeyNS.A):
            self._lookat_offset_yaw -= cam_rot_angle
        if self.gui_input.get_key(GuiInput.KeyNS.D):
            self._lookat_offset_yaw += cam_rot_angle

    def _camera_pitch_and_yaw_mouse_control(self):
        enable_mouse_control = (
            self._first_person_mode
            and self._cursor_style == Application.Cursor.HIDDEN_LOCKED
        ) or (
            not self._first_person_mode
            and self.gui_input.get_key(GuiInput.KeyNS.R)
        )

        if enable_mouse_control:
            # update yaw and pitch by scale * mouse relative position delta
            scale = 1 / 50
            self._lookat_offset_yaw += (
                scale * self.gui_input.relative_mouse_position[0]
            )
            self._lookat_offset_pitch += (
                scale * self.gui_input.relative_mouse_position[1]
            )
            self._lookat_offset_pitch = np.clip(
                self._lookat_offset_pitch,
                self._min_lookat_offset_pitch,
                self._max_lookat_offset_pitch,
            )

    def _draw_nav_hint(
        self, start_pos, start_dir, end_pos, end_radius, color, anim_fraction
    ):
        assert isinstance(start_pos, mn.Vector3)
        assert isinstance(start_dir, mn.Vector3)
        assert isinstance(end_pos, mn.Vector3)

        bias_weight = 0.5
        biased_dir = (
            start_dir + (end_pos - start_pos).normalized() * bias_weight
        ).normalized()

        start_dir_weight = min(4.0, (end_pos - start_pos).length() / 2)
        ctrl_pts = [
            start_pos,
            start_pos + biased_dir * start_dir_weight,
            end_pos,
            end_pos,
        ]

        steps_per_meter = 10
        pad_meters = 1.0
        alpha_ramp_dist = 1.0
        num_steps = max(
            2,
            int(
                ((end_pos - start_pos).length() + pad_meters) * steps_per_meter
            ),
        )

        prev_pos = None
        for step_idx in range(num_steps):
            t = step_idx / (num_steps - 1) + anim_fraction * (
                1 / (num_steps - 1)
            )
            pos = evaluate_cubic_bezier(ctrl_pts, t)

            if (pos - end_pos).length() < end_radius:
                break

            if step_idx > 0:
                alpha = min(1.0, (pos - start_pos).length() / alpha_ramp_dist)

                radius = 0.05
                num_segments = 12
                # todo: use safe_normalize
                normal = (pos - prev_pos).normalized()
                color_with_alpha = mn.Color4(color)
                color_with_alpha[3] *= alpha
                self._debug_line_render.draw_circle(
                    pos, radius, color_with_alpha, num_segments, normal
                )
            prev_pos = pos

    def _free_camera_lookat_control(self):
        if self.lookat is None:
            # init lookat
            self.lookat = np.array(
                self.get_sim().sample_navigable_point()
            ) + np.array([0, 1, 0])
        else:
            # update lookat
            move_delta = 0.1
            move = np.zeros(3)
            if self.gui_input.get_key(GuiInput.KeyNS.W):
                move[0] -= move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.S):
                move[0] += move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.O):
                move[1] += move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.P):
                move[1] -= move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.J):
                move[2] += move_delta
            if self.gui_input.get_key(GuiInput.KeyNS.L):
                move[2] -= move_delta

            # align move forward direction with lookat direction
            rotation_rad = -self.lookat_offset_yaw
            rot_matrix = np.array(
                [
                    [np.cos(rotation_rad), 0, np.sin(rotation_rad)],
                    [0, 1, 0],
                    [-np.sin(rotation_rad), 0, np.cos(rotation_rad)],
                ]
            )

            self.lookat += mn.Vector3(rot_matrix @ move)

        # highlight the lookat translation as a red circle
        self._debug_line_render.draw_circle(
            self.lookat, 0.03, mn.Color3(1, 0, 0)
        )

    def _save_recorded_keyframes_to_file(self):
        # Consolidate recorded keyframes into a single json string
        # self._recording_keyframes format:
        #     List['{"keyframe": {...}', '{"keyframe": {...}', ...]
        # Output format:
        #     '{"keyframes": [{...}, {...}, ...]}'
        json_keyframes = "".join(
            keyframe[12:-1] + ","
            for keyframe in self._recording_keyframes[:-1]
        )
        json_keyframes += self._recording_keyframes[-1:][0][
            12:-1
        ]  # Last element without trailing comma
        json_content = '{{"keyframes": [{}]}}'.format(json_keyframes)

        # Save keyframes to file
        with open(self._gfx_replay_save_path, "w") as json_file:
            json_file.write(json_content)

    def _update_cursor_style(self):
        do_update_cursor = False
        if self._cursor_style is None:
            self._cursor_style = Application.Cursor.ARROW
            do_update_cursor = True
        else:
            if (
                self._first_person_mode
                and self.gui_input.get_mouse_button_down(GuiInput.MouseNS.LEFT)
            ):
                # toggle cursor mode
                self._cursor_style = (
                    Application.Cursor.HIDDEN_LOCKED
                    if self._cursor_style == Application.Cursor.ARROW
                    else Application.Cursor.ARROW
                )
                do_update_cursor = True

        return do_update_cursor

    def _update_help_text(self):
        def get_grasp_release_controls_text():
            if self._held_target_obj_idx is not None:
                return "Spacebar: put down\n"
            else:
                return "Spacebar: pick up\n"

        controls_str = ""
        controls_str += "ESC: exit\n"
        controls_str += "M: next episode\n"

        if self._env_episode_active():
            if self._first_person_mode:
                # controls_str += "Left-click: toggle cursor\n"  # make this "unofficial" for now
                controls_str += "I, K: look up, down\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += get_grasp_release_controls_text()
            # third-person mode
            elif not self.is_free_camera_mode():
                controls_str += "R + drag: rotate camera\n"
                controls_str += "Right-click: walk\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += "Scroll: zoom\n"
                controls_str += get_grasp_release_controls_text()
            else:
                controls_str += "Left-click + drag: rotate camera\n"
                controls_str += "A, D: turn camera\n"
                controls_str += "W, S: pan camera\n"
                controls_str += "O, P: raise or lower camera\n"
                controls_str += "Scroll: zoom\n"

        self._text_drawer.add_text(
            controls_str, TextOnScreenAlignment.TOP_LEFT
        )

        status_str = ""
        if not self.env.episode_over:
            if not self._env_task_complete:
                assert self._num_remaining_objects is not None
                assert self._num_busy_objects is not None
                if self._held_target_obj_idx is not None:
                    sim = self.get_sim()
                    grasp_object_id = sim.scene_obj_ids[
                        self._held_target_obj_idx
                    ]
                    obj_handle = (
                        sim.get_rigid_object_manager().get_object_handle_by_id(
                            grasp_object_id
                        )
                    )
                    status_str += (
                        "Place the "
                        + get_pretty_object_name_from_handle(obj_handle)
                        + " at its goal location!\n"
                    )
                elif self._num_remaining_objects > 0:
                    status_str += "Move the remaining {} object{}!".format(
                        self._num_remaining_objects,
                        "s" if self._num_remaining_objects > 1 else "",
                    )
                elif self._num_busy_objects > 0:
                    status_str += (
                        "Just wait! The robot is moving the last object.\n"
                    )
            else:
                status_str += "Task complete!"
        else:
            status_str += "Episode over!"

        self._text_drawer.add_text(
            status_str,
            TextOnScreenAlignment.TOP_CENTER,
            text_delta_x=-150,
            text_delta_y=-50,
        )

    def sim_update(self, dt):
        # todo: pipe end_play somewhere
        post_sim_update_dict: Dict[str, Any] = {}

        if self.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            post_sim_update_dict["application_exit"] = True

        # Capture gfx-replay file
        if self.gui_input.get_key_down(GuiInput.KeyNS.PERIOD):
            self._save_recorded_keyframes_to_file()

        if self.gui_input.get_key_down(GuiInput.KeyNS.M):
            self._env_reset()

        # _viz_anim_fraction goes from 0 to 1 over time and then resets to 0
        viz_anim_speed = 2.0
        self._viz_anim_fraction = (
            self._viz_anim_fraction + dt * viz_anim_speed
        ) % 1.0

        self._update_task()
        self._update_grasping_and_set_act_hints()

        # Navmesh visualization only works in the debub third-person view
        # (--debug-third-person-width), not the main sandbox viewport. Navmesh
        # visualization is only implemented for simulator-rendering, not replay-
        # rendering.
        if self.gui_input.get_key_down(GuiInput.KeyNS.N):
            self.env._sim.navmesh_visualization = (  # type: ignore
                not self.env._sim.navmesh_visualization  # type: ignore
            )

        self._compute_action_and_step_env()

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
            self.cam_zoom_dist = mn.math.clamp(
                self.cam_zoom_dist, self._min_zoom_dist, self._max_zoom_dist
            )

        # two ways for camera pitch and yaw control for UX comparison:
        # 1) press/hold ADIK keys
        self._camera_pitch_and_yaw_wasd_control()
        # 2) press left mouse button and move mouse
        self._camera_pitch_and_yaw_mouse_control()

        agent_idx = self.ctrl_helper.get_gui_controlled_agent_index()
        if agent_idx is None:
            self._free_camera_lookat_control()
            lookat = self.lookat
        else:
            art_obj = (
                self.get_sim().agents_mgr[agent_idx].articulated_agent.sim_obj
            )
            robot_root = art_obj.transformation
            lookat = robot_root.translation + mn.Vector3(0, 1, 0)

        if self._first_person_mode:
            self.cam_zoom_dist = self._min_zoom_dist
            lookat += 0.075 * robot_root.backward
            lookat -= mn.Vector3(0, 0.2, 0)

        offset = mn.Vector3(
            np.cos(self.lookat_offset_yaw) * np.cos(self.lookat_offset_pitch),
            np.sin(self.lookat_offset_pitch),
            np.sin(self.lookat_offset_yaw) * np.cos(self.lookat_offset_pitch),
        )

        self.cam_transform = mn.Matrix4.look_at(
            lookat + offset.normalized() * self.cam_zoom_dist,
            lookat,
            mn.Vector3(0, 1, 0),
        )
        post_sim_update_dict["cam_transform"] = self.cam_transform

        if self._update_cursor_style():
            post_sim_update_dict["application_cursor"] = self._cursor_style

        keyframes = (
            self.get_sim().gfx_replay_manager.write_incremental_saved_keyframes_to_string_array()
        )
        post_sim_update_dict["keyframes"] = keyframes

        if self._enable_gfx_replay_save:
            for keyframe in keyframes:
                self._recording_keyframes.append(keyframe)

        def depth_to_rgb(obs):
            converted_obs = np.concatenate(
                [obs * 255.0 for _ in range(3)], axis=2
            ).astype(np.uint8)
            return converted_obs

        # reference code for visualizing a camera sensor in the app GUI
        assert set(self._debug_images).issubset(set(self._obs.keys())), (
            f"Camera sensors ids: {list(set(self._debug_images).difference(set(self._obs.keys())))} "
            f"not in available sensors ids: {list(self._obs.keys())}"
        )
        debug_images = (
            depth_to_rgb(self._obs[k]) if "depth" in k else self._obs[k]
            for k in self._debug_images
        )
        post_sim_update_dict["debug_images"] = [
            np.flipud(image) for image in debug_images
        ]

        self._update_help_text()

        return post_sim_update_dict

    @staticmethod
    def _to_zero_2pi_range(radians):
        """Helper method to properly clip radians to [0, 2pi] range."""
        return (
            (2 * np.pi) - ((-radians) % (2 * np.pi))
            if radians < 0
            else radians % (2 * np.pi)
        )


def evaluate_cubic_bezier(ctrl_pts, t):
    assert len(ctrl_pts) == 4
    weights = (
        pow(1 - t, 3),
        3 * t * pow(1 - t, 2),
        3 * pow(t, 2) * (1 - t),
        pow(t, 3),
    )

    result = weights[0] * ctrl_pts[0]
    for i in range(1, 4):
        result += weights[i] * ctrl_pts[i]

    return result


def parse_debug_third_person(args, framebuffer_size):
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
        "--gui-controlled-agent-index",
        type=int,
        default=None,
        help=(
            "GUI-controlled agent index (must be >= 0 and < number of agents). "
            "Defaults to None, indicating that all the agents are policy-controlled. "
            "If none of the agents is GUI-controlled, the camera is switched to 'free camera' mode "
            "that lets the user observe the scene (instead of controlling one of the agents)"
        ),
    )
    parser.add_argument(
        "--disable-inverse-kinematics",
        action="store_true",
        help="If specified, does not add the inverse kinematics end-effector control. Only relevant for a user-controlled *robot* agent.",
    )
    parser.add_argument("--cfg", type=str, default=DEFAULT_CFG)
    parser.add_argument(
        "--cfg-opts",
        nargs="*",
        default=list(),
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--debug-images",
        nargs="*",
        default=list(),
        help=(
            "Visualize camera sensors (corresponding to `--debug-images` keys) in the app GUI."
            "For example, to visualize agent1's head depth sensor set: --debug-images agent_1_head_depth"
        ),
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
    parser.add_argument(
        "--debug-third-person-width",
        default=0,
        type=int,
        help="If specified, enable the debug third-person camera (habitat.simulator.debug_render) with specified viewport width",
    )
    parser.add_argument(
        "--debug-third-person-height",
        default=0,
        type=int,
        help="If specified, use the specified viewport height for the debug third-person camera",
    )
    parser.add_argument(
        "--max-look-up-angle",
        default=15,
        type=float,
        help="Look up angle limit.",
    )
    parser.add_argument(
        "--min-look-down-angle",
        default=-60,
        type=float,
        help="Look down angle limit.",
    )
    parser.add_argument(
        "--first-person-mode",
        action="store_true",
        default=False,
        help="Choose between classic and batch renderer",
    )
    parser.add_argument(
        "--can-grasp-place-threshold",
        default=1.2,
        type=float,
        help="Object grasp/place proximity threshold",
    )
    # temp argument:
    # allowed to switch between oracle baseline nav
    # and random base vel action
    parser.add_argument(
        "--sample-random-baseline-base-vel",
        action="store_true",
        default=False,
        help="Sample random BaselinesController base vel",
    )
    parser.add_argument(
        "--enable-gfx-replay-save",
        action="store_true",
        default=False,
        help="Save the gfx-replay keyframes to file. Use --gfx-replay-save-path to specify the save location.",
    )
    parser.add_argument(
        "--gfx-replay-save-path",
        default="./data/gfx-replay.json",
        type=str,
        help="Path where the captured graphics replay file is saved.",
    )

    args = parser.parse_args()

    glfw_config = Application.Configuration()
    glfw_config.title = "Sandbox App"
    glfw_config.size = (args.width, args.height)
    gui_app_wrapper = GuiApplication(glfw_config, args.target_sps)
    # on Mac Retina displays, this will be 2x the window size
    framebuffer_size = gui_app_wrapper.get_framebuffer_size()

    (
        show_debug_third_person,
        debug_third_person_width,
        debug_third_person_height,
    ) = parse_debug_third_person(args, framebuffer_size)

    config = get_baselines_config(args.cfg, args.cfg_opts)
    # config = habitat.get_config(args.cfg, args.cfg_opts)
    with habitat.config.read_write(config):
        habitat_config = config.habitat
        env_config = habitat_config.environment
        sim_config = habitat_config.simulator
        task_config = habitat_config.task
        task_config.actions["pddl_apply_action"] = PddlApplyActionConfig()
        # task_config.actions[
        #     "agent_1_oracle_nav_action"
        # ] = OracleNavActionConfig(agent_index=1)

        agent_config = get_agent_config(sim_config=sim_config)

        if show_debug_third_person:
            sim_config.debug_render = True
            agent_config.sim_sensors.update(
                {
                    "third_rgb_sensor": ThirdRGBSensorConfig(
                        height=debug_third_person_height,
                        width=debug_third_person_width,
                    )
                }
            )
            agent_key = "" if len(sim_config.agents) == 1 else "agent_0_"
            args.debug_images.append(f"{agent_key}third_rgb")

        # Code below is ported from interactive_play.py. I'm not sure what it is for.
        if True:
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

        if args.gui_controlled_agent_index is not None:
            if not (
                args.gui_controlled_agent_index >= 0
                and args.gui_controlled_agent_index < len(sim_config.agents)
            ):
                print(
                    f"--gui-controlled-agent-index argument value ({args.gui_controlled_agent_index}) "
                    f"must be >= 0 and < number of agents ({len(sim_config.agents)})"
                )
                exit()

            gui_agent_key = sim_config.agents_order[
                args.gui_controlled_agent_index
            ]
            if (
                sim_config.agents[gui_agent_key].articulated_agent_type
                != "KinematicHumanoid"
            ):
                print(
                    f"Selected agent for GUI control is of type {sim_config.agents[gui_agent_key].articulated_agent_type}, "
                    "but only KinematicHumanoid is supported at the moment."
                )
                exit()

            task_actions = task_config.actions
            gui_agent_actions = [
                action_key
                for action_key in task_actions.keys()
                if action_key.startswith(gui_agent_key)
            ]
            for action_key in gui_agent_actions:
                task_actions.pop(action_key)

            action_prefix = (
                f"{gui_agent_key}_" if len(sim_config.agents) > 1 else ""
            )
            task_actions[
                f"{action_prefix}humanoidjoint_action"
            ] = HumanoidJointActionConfig(
                agent_index=args.gui_controlled_agent_index
            )

    driver = SandboxDriver(args, config, gui_app_wrapper.get_sim_input())

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
    gui_app_wrapper.set_driver_and_renderer(driver, app_renderer)

    # sloppy: provide replay app renderer's debug_line_render to our driver
    driver.set_debug_line_render(
        app_renderer._replay_renderer.debug_line_render(0)
    )
    # sloppy: provide app renderer's text_drawer to our driver
    driver.set_text_drawer(app_renderer._text_drawer)

    gui_app_wrapper.exec()
