#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from app_states.app_state_abc import AppState
from camera_helper import CameraHelper
from controllers.baselines_controller import FetchState
from controllers.gui_controller import GuiHumanoidController
from gui_navigation_helper import GuiNavigationHelper
from gui_pick_helper import GuiPickHelper
from gui_throw_helper import GuiThrowHelper
from hablab_utils import get_agent_art_obj_transform, get_grasped_objects_idxs

from habitat.gui.gui_input import GuiInput
from habitat.gui.text_drawer import TextOnScreenAlignment


class AppStateFetch(AppState):
    def __init__(self, sandbox_service, gui_agent_ctrl, robot_agent_ctrl):
        self._sandbox_service = sandbox_service
        self._gui_agent_ctrl = gui_agent_ctrl
        self.state_machine_agent_ctrl = robot_agent_ctrl
        self._can_grasp_place_threshold = (
            self._sandbox_service.args.can_grasp_place_threshold
        )

        self._cam_transform = None

        self._held_target_obj_idx = None

        # will be set in on_environment_reset
        self._target_obj_ids = None

        self._camera_helper = CameraHelper(
            self._sandbox_service.args, self._sandbox_service.gui_input
        )
        self._first_person_mode = self._sandbox_service.args.first_person_mode

        self._nav_helper = GuiNavigationHelper(
            self._sandbox_service, self.get_gui_controlled_agent_index()
        )
        self._throw_helper = GuiThrowHelper(
            self._sandbox_service, self.get_gui_controlled_agent_index()
        )
        self._pick_helper = GuiPickHelper(
            self._sandbox_service,
            self.get_gui_controlled_agent_index(),
            self._get_agent_feet_height(),
        )
        self._prepare_throw = False

        self._gui_agent_ctrl.line_renderer = sandbox_service.line_render

    def on_environment_reset(self, episode_recorder_dict):
        self._held_target_obj_idx = None

        sim = self.get_sim()
        temp_ids, _ = sim.get_targets()
        self._target_obj_ids = [
            sim._scene_obj_ids[temp_id] for temp_id in temp_ids
        ]

        self._nav_helper.on_environment_reset()
        self._pick_helper.on_environment_reset(
            agent_feet_height=self._get_agent_feet_height()
        )

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

        self._prepare_throw = False

    def get_sim(self):
        return self._sandbox_service.sim

    def _update_grasping_and_set_act_hints(self):
        drop_pos = None
        grasp_object_id = None
        will_throw = False
        if self._held_target_obj_idx is not None:
            if self._sandbox_service.gui_input.get_key_down(
                GuiInput.KeyNS.SPACE
            ):
                self._prepare_throw = True
                # # temp: drop object right where it is
                # drop_pos = self._get_target_object_position(
                #     self._held_target_obj_idx
                # )
                # drop_pos.y = 0.0
                # self._held_target_obj_idx = None
            elif self._sandbox_service.gui_input.get_key_up(
                GuiInput.KeyNS.SPACE
            ):
                if self._prepare_throw:
                    will_throw = True
                    throw_obj_id = (
                        self._gui_agent_ctrl._get_grasp_mgr().snap_idx
                    )
                    self.state_machine_agent_ctrl.object_interest_id = (
                        throw_obj_id
                    )
                    sim = self.get_sim()
                    self.state_machine_agent_ctrl.rigid_obj_interest = (
                        sim.get_rigid_object_manager().get_object_by_id(
                            throw_obj_id
                        )
                    )
                    self._held_target_obj_idx = None
                self._prepare_throw = False

            # todo: implement throwing, including viz
        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:
                assert not self._gui_agent_ctrl.is_grasped
                # pick up an object
                if self._sandbox_service.gui_input.get_key_down(
                    GuiInput.KeyNS.SPACE
                ):
                    translation = self._get_agent_translation()
                    obj_pick = self._pick_helper.viz_and_get_pick_object()

                    if obj_pick is not None:
                        # We will use obj0 as our object of interest
                        self._target_obj_ids[0] = obj_pick
                        min_dist = self._can_grasp_place_threshold

                        i = 0
                        this_target_pos = self._get_target_object_position(i)

                        # compute distance in xz plane
                        offset = this_target_pos - translation
                        offset.y = 0
                        dist_xz = offset.length()
                        if dist_xz < min_dist:
                            self._held_target_obj_idx = 0
                            grasp_object_id = self._target_obj_ids[
                                self._held_target_obj_idx
                            ]

        walk_dir = None
        if not self._prepare_throw and not will_throw:
            if not self._first_person_mode:
                candidate_walk_dir = (
                    self._nav_helper.viz_and_get_humanoid_walk_dir()
                    if not self._first_person_mode
                    else None
                )
                obj_pick = self._pick_helper.viz_and_get_pick_object()

                if self._sandbox_service.gui_input.get_mouse_button(
                    GuiInput.MouseNS.RIGHT
                ):
                    walk_dir = candidate_walk_dir
            self._gui_agent_ctrl.set_act_hints(
                walk_dir,
                grasp_object_id,
                drop_pos,
                self._camera_helper.lookat_offset_yaw,
            )
            # TODO: move this in set_act_hints
            if obj_pick is not None:
                self._gui_agent_ctrl.selected_obj = self.get_sim().get_rigid_object_manager().get_object_by_id(
                    obj_pick
                )
        else:
            computed_throw_vel = (
                self._throw_helper.viz_and_get_humanoid_throw()
            )
            throw_vel = None

            if will_throw:
                throw_vel = computed_throw_vel
                self._gui_agent_ctrl.set_act_hints(
                    walk_dir,
                    None,
                    drop_pos,
                    self._camera_helper.lookat_offset_yaw,
                    throw_vel=throw_vel,
                )
            if will_throw:
                # pass
                self.state_machine_agent_ctrl.current_state = FetchState.PICK

            will_throw = False
        return drop_pos

    def _get_target_object_position(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id).translation

    def _get_target_object_positions(self):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        return np.array(
            [
                rom.get_object_by_id(obj_id).translation
                for obj_id in self._target_obj_ids
            ]
        )

    def _viz_objects(self):
        if self._held_target_obj_idx is not None:
            return

        grasped_objects_idxs = get_grasped_objects_idxs(self.get_sim())

        # draw nav_hint and target box
        for i in range(len(self._target_obj_ids)):
            # object is grasped
            if i in grasped_objects_idxs:
                continue

            color = mn.Color3(255 / 255, 128 / 255, 0)  # orange

            this_target_pos = self._get_target_object_position(i)
            box_half_size = 0.15
            box_offset = mn.Vector3(
                box_half_size, box_half_size, box_half_size
            )
            self._sandbox_service.line_render.draw_box(
                this_target_pos - box_offset,
                this_target_pos + box_offset,
                color,
            )

            # draw can grasp area
            can_grasp_position = mn.Vector3(this_target_pos)
            can_grasp_position[1] = self._get_agent_feet_height()
            self._sandbox_service.line_render.draw_circle(
                can_grasp_position,
                self._can_grasp_place_threshold,
                mn.Color3(255 / 255, 255 / 255, 0),
                24,
            )

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def _get_agent_translation(self):
        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        return (
            self._gui_agent_ctrl._humanoid_controller.obj_transform_base.translation
        )

    def _get_agent_feet_height(self):
        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        base_offset = (
            self._gui_agent_ctrl.get_articulated_agent().params.base_offset
        )
        agent_feet_translation = self._get_agent_translation() + base_offset
        return agent_feet_translation[1]

    def _get_controls_text(self):
        def get_grasp_release_controls_text():
            if self._held_target_obj_idx is not None:
                return "Spacebar: throw\n"
            else:
                return "Spacebar: pick up\n"

        controls_str: str = ""
        controls_str += "ESC: exit\n"
        controls_str += "M: change scene\n"

        if self._first_person_mode:
            # controls_str += "Left-click: toggle cursor\n"  # make this "unofficial" for now
            controls_str += "I, K: look up, down\n"
            controls_str += "A, D: turn\n"
            controls_str += "W, S: walk\n"
            controls_str += get_grasp_release_controls_text()
        # third-person mode
        else:
            controls_str += "R + drag: rotate camera\n"
            controls_str += "Right-click: walk\n"
            controls_str += "A, D: turn\n"
            controls_str += "W, S: walk\n"
            controls_str += "Scroll: zoom\n"
            controls_str += get_grasp_release_controls_text()

        return controls_str

    def _get_status_text(self):
        status_str = ""

        if self._held_target_obj_idx is not None:
            status_str += "Throw the object!"
        else:
            status_str += "Grab an object!\n"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len) for line in status_str.split("/n")
        )

        return status_str

    def _update_help_text(self):
        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._sandbox_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index()
        )
        lookat = agent_root.translation + mn.Vector3(0, 1, 0)
        return lookat

    def sim_update(self, dt, post_sim_update_dict):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._sandbox_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.M):
            self._sandbox_service.end_episode(do_reset=True)

        # self._viz_objects()
        self._update_grasping_and_set_act_hints()
        self._sandbox_service.compute_action_and_step_env()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()
