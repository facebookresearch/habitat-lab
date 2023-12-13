#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from controllers.gui_controller import GuiHumanoidController
from gui_navigation_helper import GuiNavigationHelper
from utils.hablab_utils import get_grasped_objects_idxs

from utils.gui.gui_input import GuiInput


class GuiRearrangeHelper:
    """Helper class for the Rearrange task.

    Encapsulates grasping and visualization logic and is shared between
    AppStateRearrange and AppStateSocialNavStudy app state classes.
    """

    def __init__(
        self,
        sandbox_service,
        gui_agent_ctrl,
        camera_helper,
    ):
        self._sandbox_service = sandbox_service
        self._gui_agent_ctrl = gui_agent_ctrl
        self._camera_helper = camera_helper
        self._nav_helper = GuiNavigationHelper(
            self._sandbox_service, self.get_gui_controlled_agent_index()
        )

        # will be set in on_environment_reset
        self._target_obj_ids = None
        self._goal_positions = None

        self._can_grasp_place_threshold = (
            self._sandbox_service.args.can_grasp_place_threshold
        )

        self._held_target_obj_idx = None
        self._num_remaining_objects = None  # resting, not at goal location yet
        self._num_busy_objects = None  # currently held by non-gui agents

        # cache items from config; config is expensive to access at runtime
        config = self._sandbox_service.config
        self._end_on_success = config.habitat.task.end_on_success
        self._obj_succ_thresh = config.habitat.task.obj_succ_thresh
        self._success_measure_name = config.habitat.task.success_measure

    @property
    def target_obj_ids(self):
        return self._target_obj_ids

    @property
    def goal_positions(self):
        return self._goal_positions

    @property
    def held_target_obj_idx(self):
        return self._held_target_obj_idx

    @property
    def end_on_success(self):
        return self._end_on_success

    @property
    def success_measure_name(self):
        return self._success_measure_name

    @property
    def num_busy_objects(self):
        return self._num_busy_objects

    @property
    def num_remaining_objects(self):
        return self._num_remaining_objects

    def on_environment_reset(self):
        self._held_target_obj_idx = None
        self._num_remaining_objects = None  # resting, not at goal location yet
        self._num_busy_objects = None  # currently held by non-gui agents

        sim = self.get_sim()
        temp_ids, goal_positions_np = sim.get_targets()
        self._target_obj_ids = [
            sim._scene_obj_ids[temp_id] for temp_id in temp_ids
        ]
        self._goal_positions = [mn.Vector3(pos) for pos in goal_positions_np]

        self._nav_helper.on_environment_reset()

    def _update_grasping_and_set_act_hints(self):
        end_radius = self._obj_succ_thresh

        drop_pos = None
        grasp_object_id = None
        reach_pos = None

        if self._held_target_obj_idx is not None:
            color = mn.Color3(0, 255 / 255, 0)  # green
            goal_position = self._goal_positions[self._held_target_obj_idx]
            self._sandbox_service.line_render.draw_circle(
                goal_position, end_radius, color, 24
            )

            self._nav_helper._draw_nav_hint_from_agent(
                self._camera_helper.get_xz_forward(),
                mn.Vector3(goal_position),
                end_radius,
                color,
            )

            # draw can place area
            can_place_position = mn.Vector3(goal_position)
            can_place_position[1] = self._get_agent_feet_height()
            self._sandbox_service.line_render.draw_circle(
                can_place_position,
                self._can_grasp_place_threshold,
                mn.Color3(255 / 255, 255 / 255, 0),
                24,
            )

            if self._sandbox_service.gui_input.get_key_down(
                GuiInput.KeyNS.SPACE
            ):
                translation = self._get_agent_translation()
                dist_to_obj = np.linalg.norm(goal_position - translation)
                if dist_to_obj < self._can_grasp_place_threshold:
                    self._held_target_obj_idx = None
                    drop_pos = goal_position
        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:
                assert not self._gui_agent_ctrl.is_grasped
                # pick up an object
                if self._sandbox_service.gui_input.get_key_down(
                    GuiInput.KeyNS.SPACE
                ):
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
                        # reach towards this position until spacebar is released
                        reach_pos = self._get_target_object_position(min_i)

        walk_dir = None
        distance_multiplier = 1.0
        if not self._camera_helper.first_person_mode:
            (
                candidate_walk_dir,
                candidate_distance_multiplier,
            ) = self._nav_helper.get_humanoid_walk_hints_from_ray_cast(
                visualize_path=True
            )
            if self._sandbox_service.gui_input.get_mouse_button(
                GuiInput.MouseNS.RIGHT
            ):
                walk_dir = candidate_walk_dir
                distance_multiplier = candidate_distance_multiplier

        self._gui_agent_ctrl.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helper.lookat_offset_yaw,
            reach_pos=reach_pos,
        )

    def _update_task(self):
        end_radius = self._obj_succ_thresh

        grasped_objects_idxs = get_grasped_objects_idxs(
            self.get_sim(),
            agent_idx_to_skip=self.get_gui_controlled_agent_index(),
        )
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
                self._sandbox_service.line_render.draw_box(
                    this_target_pos - box_offset,
                    this_target_pos + box_offset,
                    color,
                )

                self._nav_helper._draw_nav_hint_from_agent(
                    self._camera_helper.get_xz_forward(),
                    mn.Vector3(this_target_pos),
                    end_radius,
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

    def update(self):
        self._update_task()
        self._update_grasping_and_set_act_hints()

    def get_sim(self):
        return self._sandbox_service.sim

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def get_target_object_positions(self):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        return np.array(
            [
                rom.get_object_by_id(obj_id).translation
                for obj_id in self._target_obj_ids
            ]
        )

    def _get_target_object_position(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id).translation

    def _is_target_object_at_goal_position(self, target_obj_idx):
        this_target_pos = self._get_target_object_position(target_obj_idx)
        end_radius = self._obj_succ_thresh
        return (
            this_target_pos - self._goal_positions[target_obj_idx]
        ).length() < end_radius

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
