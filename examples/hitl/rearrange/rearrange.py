#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import hydra
import magnum as mn
import numpy as np

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.app_states.app_state_tutorial import AppStateTutorial
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
)
from habitat_hitl.environment.gui_navigation_helper import GuiNavigationHelper
from habitat_hitl.environment.hablab_utils import (
    get_agent_art_obj_transform,
    get_grasped_objects_idxs,
)


class AppStateRearrange(AppState):
    """
    A user-controlled human and policy-controlled robot must accomplish a collaborative rearrangement task in HSSD scenes.

    See habitat-hitl/README.md for more about AppStates and HITL apps.

    Overview of the main pieces of this class:
    * sim_update: per-frame entrypoint. Step the habitat env, update the camera and help text, and restart episodes as necessary.
    * _update_task: visualize the task for the user, namely, show a 3D navigation hint to the target objects.
    * _update_grasping_and_set_act_hints: Update the user-controlled humanoid agent, e.g. grasp a nearby object when the user presses spacebar. Includes "click-to-walk": see get_humanoid_walk_hints_from_ray_cast and set_act_hints.
    * record_state: collect some key task-specific data. Note: aside from this function, most habitat-lab built-in metrics are collected automatically by the HITL framework.
    """

    def __init__(
        self,
        app_service: AppService,
    ):
        self._app_service = app_service
        self._gui_agent_ctrl: Any = (
            self._app_service.gui_agent_controllers[0]
            if len(self._app_service.gui_agent_controllers)
            else None
        )

        # cache items from config; config is expensive to access at runtime
        config = self._app_service.config
        self._end_on_success = config.habitat.task.end_on_success
        self._obj_succ_thresh = config.habitat.task.obj_succ_thresh
        self._success_measure_name = config.habitat.task.success_measure

        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )

        self._cam_transform: Optional[mn.Matrix4] = None

        self._held_target_obj_idx: Optional[int] = None

        # resting, not at goal location yet
        self._num_remaining_objects: Optional[int] = None

        # currently held by non-gui agents
        self._num_busy_objects: Optional[int] = None

        # will be set in on_environment_reset
        self._target_obj_ids: Optional[List[str]] = None
        self._goal_positions: Optional[List[mn.Vector3]] = None

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        self._first_person_mode = (
            self._app_service.hitl_config.camera.first_person_mode
        )

        self._nav_helper = GuiNavigationHelper(
            self._app_service,
            self.get_gui_controlled_agent_index(),
            user_index=0,
        )
        self._episode_helper = self._app_service.episode_helper

    def on_environment_reset(self, episode_recorder_dict):
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

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

        if episode_recorder_dict:
            episode_recorder_dict["target_obj_ids"] = self._target_obj_ids
            episode_recorder_dict["goal_positions"] = self._goal_positions

    def get_sim(self):
        return self._app_service.sim

    def _update_grasping_and_set_act_hints(self):
        end_radius = self._obj_succ_thresh

        drop_pos = None
        grasp_object_id = None

        if self._held_target_obj_idx is not None:
            color = mn.Color3(0, 255 / 255, 0)  # green
            goal_position = self._goal_positions[self._held_target_obj_idx]
            self._app_service.gui_drawer.draw_circle(
                goal_position, end_radius, color
            )

            self._nav_helper.draw_nav_hint_from_agent(
                self._camera_helper.get_xz_forward(),
                mn.Vector3(goal_position),
                end_radius,
                color,
            )

            # draw can place area
            can_place_position = mn.Vector3(goal_position)
            can_place_position[1] = self._get_agent_feet_height()
            self._app_service.gui_drawer.draw_circle(
                can_place_position,
                self._can_grasp_place_threshold,
                mn.Color3(255 / 255, 255 / 255, 0),
            )

            if self._app_service.gui_input.get_key_down(KeyCode.SPACE):
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
                if self._app_service.gui_input.get_key_down(KeyCode.SPACE):
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

        walk_dir = None
        distance_multiplier = 1.0
        if not self._first_person_mode:
            (
                candidate_walk_dir,
                candidate_distance_multiplier,
            ) = self._nav_helper.get_humanoid_walk_hints_from_ray_cast(
                visualize_path=True
            )
            if self._app_service.gui_input.get_mouse_button(MouseButton.RIGHT):
                walk_dir = candidate_walk_dir
                distance_multiplier = candidate_distance_multiplier

        self._gui_agent_ctrl.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helper.lookat_offset_yaw,
        )

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

    def _is_target_object_at_goal_position(self, target_obj_idx):
        this_target_pos = self._get_target_object_position(target_obj_idx)
        end_radius = self._obj_succ_thresh
        return (
            this_target_pos - self._goal_positions[target_obj_idx]
        ).length() < end_radius

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
                self._app_service.gui_drawer.draw_box(
                    this_target_pos - box_offset,
                    this_target_pos + box_offset,
                    color,
                )

                self._nav_helper.draw_nav_hint_from_agent(
                    self._camera_helper.get_xz_forward(),
                    mn.Vector3(this_target_pos),
                    end_radius,
                    color,
                )

                # draw can grasp area
                can_grasp_position = mn.Vector3(this_target_pos)
                can_grasp_position[1] = self._get_agent_feet_height()
                self._app_service.gui_drawer.draw_circle(
                    can_grasp_position,
                    self._can_grasp_place_threshold,
                    mn.Color3(255 / 255, 255 / 255, 0),
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
                return "Spacebar: put down\n"
            else:
                return "Spacebar: pick up\n"

        controls_str: str = ""
        controls_str += "ESC: exit\n"
        if self._episode_helper.next_episode_exists():
            controls_str += "M: next episode\n"

        if self._env_episode_active():
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

    @property
    def _env_task_complete(self):
        return (
            self._end_on_success
            and self._app_service.get_metrics()[self._success_measure_name]
        )

    def _env_episode_active(self) -> bool:
        """
        Returns True if current episode is active:
        1) not self._app_service.env.episode_over - none of the constraints is violated, or
        2) not self._env_task_complete - success measure value is not True
        """
        return not (
            self._app_service.env.episode_over or self._env_task_complete
        )

    def _get_status_text(self):
        status_str = ""

        assert self._num_remaining_objects is not None
        assert self._num_busy_objects is not None

        if not self._env_episode_active():
            if self._env_task_complete:
                status_str += "Task complete!\n"
            else:
                status_str += "Oops! Something went wrong.\n"
        elif self._held_target_obj_idx is not None:
            status_str += "Place the object at its goal location!\n"
        elif self._num_remaining_objects > 0:
            status_str += "Move the remaining {} object{}!".format(
                self._num_remaining_objects,
                "s" if self._num_remaining_objects > 1 else "",
            )
        elif self._num_busy_objects > 0:
            status_str += "Just wait! The robot is moving the last object.\n"
        else:
            # we don't expect to hit this case ever
            status_str += "Oops! Something went wrong.\n"

        # center align the status_str
        max_status_str_len = 50
        status_str = "/n".join(
            line.center(max_status_str_len) for line in status_str.split("/n")
        )

        return status_str

    def _update_help_text(self):
        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

        num_episodes_remaining = (
            self._episode_helper.num_iter_episodes
            - self._episode_helper.num_episodes_done
        )
        progress_str = f"{num_episodes_remaining} episodes left"
        self._app_service.text_drawer.add_text(
            progress_str,
            TextOnScreenAlignment.TOP_RIGHT,
            text_delta_x=370,
        )

    def get_num_agents(self):
        return len(self.get_sim().agents_mgr._all_agent_data)

    def record_state(self):
        agent_states = []
        for agent_idx in range(self.get_num_agents()):
            agent_root = get_agent_art_obj_transform(self.get_sim(), agent_idx)
            rotation_quat = mn.Quaternion.from_matrix(agent_root.rotation())
            rotation_list = list(rotation_quat.vector) + [rotation_quat.scalar]
            pos = agent_root.translation

            snap_idx = (
                self.get_sim()
                .agents_mgr._all_agent_data[agent_idx]
                .grasp_mgr.snap_idx
            )

            agent_states.append(
                {
                    "position": pos,
                    "rotation_xyzw": rotation_list,
                    "grasp_mgr_snap_idx": snap_idx,
                }
            )

        self._app_service.step_recorder.record("agent_states", agent_states)

        self._app_service.step_recorder.record(
            "target_object_positions", self._get_target_object_positions()
        )

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index()
        )
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def sim_update(self, dt, post_sim_update_dict):
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        if (
            self._app_service.gui_input.get_key_down(KeyCode.M)
            and self._episode_helper.next_episode_exists()
        ):
            self._app_service.end_episode(do_reset=True)

        if self._env_episode_active():
            self._update_task()
            self._update_grasping_and_set_act_hints()
            self._app_service.compute_action_and_step_env()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()


class AppStateRearrangeTutorialTransition(AppState):
    """
    Helper class to manage the transition from the tutorial AppState to the rearrange AppState.

    Each episode starts with the tutorial (a camera flythrough sequence with help text), and then we switch to rearrange when the tutorial ends.
    """

    def __init__(self, app_service):
        self._app_state_rearrange = AppStateRearrange(app_service)
        self._app_state_tutorial = AppStateTutorial(app_service)
        self._is_tutorial_active = False
        self._only_show_tutorial_once = True
        self._is_first_reset = True

    def _start_tutorial(self):
        assert not self._is_tutorial_active
        final_eye_pos = self._app_state_rearrange._camera_helper.get_eye_pos()
        final_lookat_pos = (
            self._app_state_rearrange._camera_helper.get_lookat_pos()
        )
        self._app_state_tutorial.on_enter(final_eye_pos, final_lookat_pos)
        self._is_tutorial_active = True

    def _get_active_app_state(self):
        return (
            self._app_state_tutorial
            if self._is_tutorial_active
            else self._app_state_rearrange
        )

    def sim_update(self, dt, post_sim_update_dict):
        self._get_active_app_state().sim_update(dt, post_sim_update_dict)

        if (
            self._is_tutorial_active
            and self._app_state_tutorial._tutorial.is_completed()
        ):
            self._is_tutorial_active = False

    def on_environment_reset(self, episode_recorder_dict):
        self._app_state_rearrange.on_environment_reset(episode_recorder_dict)
        self._app_state_tutorial.on_environment_reset(episode_recorder_dict)

        if self._is_first_reset or not self._only_show_tutorial_once:
            self._start_tutorial()

        self._is_first_reset = False

    def record_state(self):
        self._get_active_app_state().record_state()


def create_app_state(app_service):
    app_state_class = (
        AppStateRearrangeTutorialTransition
        if app_service.config.rearrange.show_tutorial
        else AppStateRearrange
    )
    return app_state_class(app_service)


@hydra.main(
    version_base=None, config_path="config", config_name="hitl_rearrange"
)
def main(config):
    hitl_main(config, create_app_state)


if __name__ == "__main__":
    register_hydra_plugins()
    main()
