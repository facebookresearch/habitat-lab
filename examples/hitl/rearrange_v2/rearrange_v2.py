#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, Optional, Set
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.user_mask import UserMask
from habitat_hitl.environment.controllers.controller_abc import GuiController

import hydra
import magnum as mn

import habitat_sim
from habitat.sims.habitat_simulator import sim_utilities
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.client_helper import ClientHelper
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.gui_pick_helper import GuiPickHelper
from habitat_hitl.environment.gui_placement_helper import GuiPlacementHelper
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform

ENABLE_ARTICULATED_OPEN_CLOSE = False
# Visually snap picked objects into the humanoid's hand. May be useful in third-person mode. Beware that this conflicts with GuiPlacementHelper.
DO_HUMANOID_GRASP_OBJECTS = False


class AppStateRearrangeV2(AppState):
    """
    Todo
    """

    def __init__(self, app_service: AppService):
        # We don't sync the server camera. Instead, we maintain one camera per user.
        assert(app_service.hitl_config.networking.client_sync.camera_transform == False)

        # Sloppy: _num_users is defined as the number of agents.
        self._app_service = app_service
        self._gui_agent_controllers: List[GuiController] = self._app_service.gui_agent_controllers
        self._num_users = len(self._gui_agent_controllers)
        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )

        self._sim = app_service.sim
        self._ao_root_bbs: Dict = None
        self._opened_ao_set: Set = set()
        self._paused = False

        self._server_user_index = 0
        self._server_gui_input = self._app_service.gui_input

        self._gui_inputs: List[GuiInput] = app_service.remote_client_state.get_gui_inputs()

        self._held_obj_ids: List[Optional[Any]] = []  # TODO: Unknown type.
        self._has_grasp_preview: List[bool] = []
        self._cam_transforms: List[mn.Matrix4] = []
        self._camera_helpers: List[CameraHelper] = []
        self._client_helpers: List[ClientHelper] = []
        for user_index in range(self._num_users):
            self._held_obj_ids.append(None)
            self._has_grasp_preview.append(False)
            self._cam_transforms.append(mn.Matrix4())
            self._camera_helpers.append(CameraHelper(
                self._app_service.hitl_config,
                self._gui_inputs[user_index],
            ))
            if self._app_service.hitl_config.networking.enable:
                self._client_helpers.append(ClientHelper(self._app_service, user_index))
            # HACK: Inject GuiInput into agents
            self._gui_agent_controllers[user_index]._gui_input = self._gui_inputs[user_index]
            
            
        # TODO: These variables need to be per-user
        self._hide_gui_text = False

        self._pick_helper = GuiPickHelper(
            self._app_service,
        )
        self._placement_helper = GuiPlacementHelper(self._app_service)

        self._frame_counter = 0
        self._sps_tracker = AverageRateTracker(2.0)

    # needed to avoid spurious mypy attr-defined errors
    @staticmethod
    def get_sim_utilities() -> Any:
        return sim_utilities

    def _open_close_ao(self, ao_handle: str):
        if not ENABLE_ARTICULATED_OPEN_CLOSE:
            return

        ao = self.get_sim_utilities().get_obj_from_handle(self._sim, ao_handle)

        # Check whether the ao is opened
        is_opened = ao_handle in self._opened_ao_set

        # Set ao joint positions
        joint_limits = ao.joint_position_limits
        joint_limits = joint_limits[0] if is_opened else joint_limits[1]
        ao.joint_positions = joint_limits
        ao.clamp_joint_limits()

        # Remove ao from opened set
        if is_opened:
            self._opened_ao_set.remove(ao_handle)
        else:
            self._opened_ao_set.add(ao_handle)

    def _find_reachable_ao(self, player_pos) -> str:
        """Returns the handle of the nearest reachable articulated object. Returns None if none is found."""
        if not ENABLE_ARTICULATED_OPEN_CLOSE:
            return None

        max_distance = 2.0  # TODO: Const
        player_pos_xz = mn.Vector3(player_pos.x, 0.0, player_pos.z)
        min_dist: float = max_distance
        output: str = None

        # TODO: Caching
        # TODO: Improve heuristic using bounding box sizes and view angle
        for handle, _ in self._ao_root_bbs.items():
            ao = self.get_sim_utilities().get_obj_from_handle(
                self._sim, handle
            )
            ao_pos = ao.translation
            ao_pos_xz = mn.Vector3(ao_pos.x, 0.0, ao_pos.z)
            dist_xz = (ao_pos_xz - player_pos_xz).length()
            if dist_xz < max_distance and dist_xz < min_dist:
                min_dist = dist_xz
                output = handle

        return output

    def _highlight_ao(self, handle: str):
        assert ENABLE_ARTICULATED_OPEN_CLOSE
        bb = self._ao_root_bbs[handle]
        ao = self.get_sim_utilities().get_obj_from_handle(self._sim, handle)
        ao_pos = ao.translation
        ao_pos.y = 0.0  # project to ground
        radius = max(bb.size_x(), bb.size_y(), bb.size_z()) / 2.0
        # sloppy: use private GuiPickHelper._add_highlight_ring
        self._pick_helper._add_highlight_ring(
            ao_pos, mn.Color3(0, 1, 0), radius, do_pulse=False, billboard=False
        )

    def on_environment_reset(self, episode_recorder_dict):
        if ENABLE_ARTICULATED_OPEN_CLOSE:
            self._ao_root_bbs = self.get_sim_utilities().get_ao_root_bbs(
                self._sim
            )
            # HACK: Remove humans and spot from the AO collections
            handle_filter = ["male", "female", "hab_spot_arm"]
            for key in list(self._ao_root_bbs.keys()):
                if any(handle in key for handle in handle_filter):
                    del self._ao_root_bbs[key]

        for user_index in range(self._num_users):
            self._held_obj_ids[user_index] = None

        self._pick_helper.on_environment_reset()

        for user_index in range(self._num_users):
            self._camera_helpers[user_index].update(self._get_camera_lookat_pos(user_index), dt=0)

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change(UserMask.BROADCAST)

    def get_sim(self):
        return self._app_service.sim

    def _get_gui_agent_translation(self, user_index):
        return get_agent_art_obj_transform(
            self.get_sim(), self.get_agent_index_from_user_index(user_index)
        ).translation

    def _update_grasping_and_set_act_hints(self, user_index):
        drop_pos = None
        grasp_object_id = None
        throw_vel = None
        reach_pos = None

        self._has_grasp_preview[user_index] = False

        if self._held_obj_ids[user_index] is not None:
            if self._gui_inputs[user_index].get_key_down(GuiInput.KeyNS.SPACE):
                if DO_HUMANOID_GRASP_OBJECTS:
                    # todo: better drop pos
                    drop_pos = self._get_gui_agent_translation(
                        user_index
                    )  # self._gui_agent_controllers.get_base_translation()
                else:
                    # GuiPlacementHelper has already placed this object, so nothing to do here
                    pass
                self._held_obj_ids[user_index] = None
        else:
            query_pos = self._get_gui_agent_translation(user_index)
            obj_id = self._pick_helper.get_pick_object_near_query_position(
                query_pos
            )
            if obj_id:
                if self._gui_inputs[user_index].get_key_down(GuiInput.KeyNS.SPACE):
                    if DO_HUMANOID_GRASP_OBJECTS:
                        grasp_object_id = obj_id
                    self._held_obj_ids[user_index] = obj_id
                else:
                    self._has_grasp_preview[user_index] = True

        walk_dir = None
        distance_multiplier = 1.0

        # reference code for click-to-walk
        # if self._app_service.gui_input.get_mouse_button(
        #     GuiInput.MouseNS.RIGHT
        # ):
        #     (
        #         candidate_walk_dir,
        #         candidate_distance_multiplier,
        #     ) = self._nav_helper.get_humanoid_walk_hints_from_ray_cast(
        #         visualize_path=True
        #     )
        #     walk_dir = candidate_walk_dir
        #     distance_multiplier = candidate_distance_multiplier

        gui_agent_controller = self._gui_agent_controllers[user_index]
        assert isinstance(
            gui_agent_controller, (GuiHumanoidController, GuiRobotController)
        )
        gui_agent_controller.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helpers[user_index].lookat_offset_yaw,
            throw_vel=throw_vel,
            reach_pos=reach_pos,
        )

        return drop_pos
    
    def get_agent_index_from_user_index(self, user_index: int) -> int:
        return self._gui_agent_controllers[user_index]._agent_idx

    def _get_controls_text(self, user_index: int) -> str:
        def get_grasp_release_controls_text():
            if self._held_obj_ids[user_index] is not None:
                return "Space/N: put down\n"
            elif self._has_grasp_preview[user_index]:
                return "Space/N: pick up\n"
            else:
                return ""

        controls_str: str = ""
        if not self._hide_gui_text:
            if self._sps_tracker.get_smoothed_rate() is not None:
                controls_str += f"server SPS: {self._sps_tracker.get_smoothed_rate():.1f}\n"
            if user_index in self._client_helpers and self._client_helpers[user_index].display_latency_ms:
                controls_str += f"latency: {self._client_helpers[user_index].display_latency_ms:.0f}ms\n"
            controls_str += "H: show/hide help text\n"
            controls_str += "P: pause\n"
            controls_str += "I, K: look up, down\n"
            controls_str += "A, D: turn\n"
            controls_str += "W/F, S/V: walk\n"
            if ENABLE_ARTICULATED_OPEN_CLOSE:
                controls_str += "Z/X: open/close receptacle\n"
            controls_str += get_grasp_release_controls_text()
            if self._num_users > 1 and self._held_obj_ids[user_index] is None:
                controls_str += "T: toggle camera user\n"

        return controls_str

    def _get_status_text(self, user_index: int) -> str:
        status_str = ""

        if self._paused:
            status_str += "\n\npaused\n"
        if (
            user_index in self._client_helpers
            and self._client_helpers[user_index].do_show_idle_kick_warning
        ):
            status_str += (
                "\n\nAre you still there?\nPress any key to keep playing!\n"
            )

        return status_str

    def _update_help_text(self, user_index: int) -> None:
        controls_str = self._get_controls_text(user_index)
        if len(controls_str) > 0:
            # TODO: Vectorize text_drawer
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

        status_str = self._get_status_text(user_index)
        if len(status_str) > 0:
            # TODO: Vectorize text_drawer
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
            )

    def _get_camera_lookat_pos(self, user_index: int) -> mn.Vector3:
        agent_root = get_agent_art_obj_transform(
            self.get_sim(),
            self.get_agent_index_from_user_index(user_index),
        )
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def is_user_idle_this_frame(self, user_index: int) -> bool:
        return not self._gui_inputs[user_index].get_any_key_down()

    def _check_change_episode(self):
        if self._paused:
            return

        # episode_id should be a string, e.g. "5"
        episode_ids_by_dataset = {
            "data/datasets/hssd/rearrange/{split}/social_rearrange.json.gz": [
                "23775",
                "23776",
            ]
        }
        fallback_episode_ids = ["0", "1"]
        dataset_key = self._app_service.config.habitat.dataset.data_path
        episode_ids = (
            episode_ids_by_dataset[dataset_key]
            if dataset_key in episode_ids_by_dataset
            else fallback_episode_ids
        )

        # use number keys to select episode
        episode_index_by_key = {
            GuiInput.KeyNS.ONE: 0,
            GuiInput.KeyNS.TWO: 1,
        }
        assert len(episode_index_by_key) == len(episode_ids)

        # TODO: add random episode picker
        for key in episode_index_by_key:
            if self._server_gui_input.get_key_down(key):
                episode_id = episode_ids[episode_index_by_key[key]]
                # episode_id should be a string, e.g. "5"
                assert isinstance(episode_id, str)
                self._app_service.episode_helper.set_next_episode_by_id(
                    episode_id
                )
                self._app_service.end_episode(do_reset=True)

    def _update_held_object_placement(self, user_index: int) -> None:
        if not self._held_obj_ids[user_index]:
            return

        ray = habitat_sim.geo.Ray()
        ray.origin = self._camera_helpers[user_index].get_eye_pos()
        ray.direction = (
            self._camera_helpers[user_index].get_lookat_pos()
            - self._camera_helpers[user_index].get_eye_pos()
        ).normalized()

        if self._placement_helper.update(ray, self._held_obj_ids[user_index]):
            # sloppy: save another keyframe here since we just moved the held object
            self.get_sim().gfx_replay_manager.save_keyframe()

    def sim_update(self, dt, post_sim_update_dict):
        if (
            self._server_gui_input.get_key_down(GuiInput.KeyNS.ESC)
        ):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True
            return

        self._sps_tracker.increment()

        for user_index in range(self._num_users):
            self._client_helpers[user_index].update(
                self.is_user_idle_this_frame(user_index),
                self._sps_tracker.get_smoothed_rate(),
            )

        if self._server_gui_input.get_key_down(GuiInput.KeyNS.P):
            self._paused = not self._paused

        # TODO: Per-user
        if self._server_gui_input.get_key_down(GuiInput.KeyNS.H):
            self._hide_gui_text = not self._hide_gui_text

        self._check_change_episode()

        if not self._paused:
            for user_index in range(self._num_users):
                reachable_ao_handle = self._find_reachable_ao(
                    self._get_gui_agent_translation(user_index)
                )
                if reachable_ao_handle is not None:
                    self._highlight_ao(reachable_ao_handle)
                    if self._gui_inputs[user_index].get_key_down(GuiInput.KeyNS.Z):
                        self._open_close_ao(reachable_ao_handle)

            for user_index in range(self._num_users):
                self._update_grasping_and_set_act_hints(user_index)
            self._app_service.compute_action_and_step_env()
        else:
            # temp hack: manually add a keyframe while paused
            self.get_sim().gfx_replay_manager.save_keyframe()

        # todo: visualize objects properly for each user (this requires a separate debug_line_render per user!), or find a reasonable debug line visualization that can be shared between both users every frame.
        for user_index in range(self._num_users):
            if self._held_obj_ids[user_index] is None:
                self._pick_helper.viz_objects()

        # Switch the server-controller user.
        if (
            self._num_users > 0 and
            self._server_gui_input.get_key_down(GuiInput.KeyNS.T)
        ):
            self._server_user_index = (
                self._server_user_index + 1
            ) % self._num_users
        
        for user_index in range(self._num_users):
            self._camera_helpers[user_index].update(self._get_camera_lookat_pos(user_index), dt)
            # after camera update
            self._update_held_object_placement(user_index)

        for user_index in range(self._num_users):
            self._cam_transforms[user_index] = self._camera_helpers[user_index].get_cam_transform()
            self._update_help_text(user_index)
        
        post_sim_update_dict["cam_transform"] = self._cam_transforms[self._server_user_index]

        if self._app_service.hitl_config.networking.enable:
            for user_index in range(self._num_users):
                self._app_service._client_message_manager.update_camera_transform(
                    self._cam_transforms[user_index], UserMask.from_index(user_index)
                )


@hydra.main(
    version_base=None, config_path="config", config_name="rearrange_v2"
)
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStateRearrangeV2(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
