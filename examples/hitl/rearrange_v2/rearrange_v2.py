#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import hydra
import magnum as mn

from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
)
from habitat_hitl.environment.gui_navigation_helper import GuiNavigationHelper
from habitat_hitl.environment.gui_pick_helper import GuiPickHelper
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform


class AppStateRearrangeV2(AppState):
    """
    Todo
    """

    def __init__(self, app_service):
        self._app_service = app_service
        self._gui_agent_ctrl = self._app_service.gui_agent_controller
        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )

        self._cam_transform = None
        self._held_obj_id = None
        self._recent_reach_pos = None
        self._paused = False
        self._hide_gui_text = False

        # will be set in on_environment_reset
        self._target_obj_ids = None

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config,
            self._app_service.gui_input,
            self._app_service.client_message_manager,
        )

        self._nav_helper = GuiNavigationHelper(
            self._app_service, self.get_gui_controlled_agent_index()
        )
        self._pick_helper = GuiPickHelper(
            self._app_service,
            self.get_gui_controlled_agent_index(),
            self._get_gui_agent_feet_height(),
        )

        self._gui_agent_ctrl.line_renderer = app_service.line_render

        self._has_grasp_preview = False
        self._client_connection_id = None
        self._client_idle_frame_counter = None
        self._show_idle_kick_warning = False

    def _get_navmesh_triangle_vertices(self):
        """Return vertices (nonindexed triangles) for triangulated NavMesh polys"""
        largest_island_index = get_largest_island_index(
            self.get_sim().pathfinder, self.get_sim(), allow_outdoor=False
        )
        pts = self.get_sim().pathfinder.build_navmesh_vertices(
            largest_island_index
        )
        assert len(pts) > 0
        assert len(pts) % 3 == 0
        assert len(pts[0]) == 3
        navmesh_fixup_y = -0.17  # sloppy
        return [
            (
                float(point[0]),
                float(point[1]) + navmesh_fixup_y,
                float(point[2]),
            )
            for point in pts
        ]

    def on_environment_reset(self, episode_recorder_dict):
        self._held_obj_id = None

        sim = self.get_sim()
        self._target_obj_ids = sim._scene_obj_ids

        self._nav_helper.on_environment_reset()
        self._pick_helper.on_environment_reset(
            agent_feet_height=self._get_gui_agent_feet_height()
        )

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change()
            # Not currently needed since the browser client doesn't have a notion of a humanoid. Here for reference.
            # human_pos = (
            #     self.get_sim()
            #     .get_agent_data(self.get_gui_controlled_agent_index())
            #     .articulated_agent.base_pos
            # )
            # client_message_manager.change_humanoid_position(human_pos)
            # client_message_manager.update_navmesh_triangles(
            #     self._get_navmesh_triangle_vertices()
            # )

    def get_sim(self):
        return self._app_service.sim

    def _update_grasping_and_set_act_hints(self):
        drop_pos = None
        grasp_object_id = None
        throw_vel = None
        reach_pos = None

        self._has_grasp_preview = False

        if self._held_obj_id is not None:
            if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.SPACE):
                # todo: placement heuristic. Do something with drop_pos?
                drop_pos = self._get_gui_agent_translation()
                self._held_obj_id = None
        else:
            query_pos = self._get_gui_agent_translation()
            obj_id = self._pick_helper.get_pick_object_near_query_position(
                query_pos
            )
            if obj_id:
                if self._app_service.gui_input.get_key_down(
                    GuiInput.KeyNS.SPACE
                ):
                    grasp_object_id = obj_id
                    self._held_obj_id = obj_id
                else:
                    self._has_grasp_preview = True

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

        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        self._gui_agent_ctrl.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helper.lookat_offset_yaw,
            throw_vel=throw_vel,
            reach_pos=reach_pos,
        )

        return drop_pos

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def _get_gui_agent_translation(self):
        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        return (
            self._gui_agent_ctrl._humanoid_controller.obj_transform_base.translation
        )

    def _get_gui_agent_feet_height(self):
        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        base_offset = (
            self._gui_agent_ctrl.get_articulated_agent().params.base_offset
        )
        agent_feet_translation = (
            self._get_gui_agent_translation() + base_offset
        )
        return agent_feet_translation[1]

    def _get_controls_text(self):
        def get_grasp_release_controls_text():
            if self._held_obj_id is not None:
                return "Spacebar: put down\n"
            elif self._has_grasp_preview:
                return "Spacebar: pick up\n"
            else:
                return ""

        controls_str: str = ""
        if not self._hide_gui_text:
            controls_str += "H: show.hide help text\n"
            controls_str += "P: pause\n"
            controls_str += "I, K: look up, down\n"
            controls_str += "A, D: turn\n"
            controls_str += "W, S: walk\n"
            controls_str += get_grasp_release_controls_text()

        return controls_str

    def _get_status_text(self):
        status_str = ""

        if self._paused:
            status_str += "\n\npaused\n"
        if self._show_idle_kick_warning:
            status_str += (
                "\n\nAre you still there?\nPress any key to keep playing!\n"
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
            )

    def _get_agent_pose(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index()
        )
        return agent_root.translation, agent_root.rotation

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index()
        )
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def is_user_idle_this_frame(self):
        return not self._app_service.gui_input.get_any_key_down()

    def _update_for_remote_client_connect_and_idle(self):
        if not self._app_service.hitl_config.networking.enable:
            return
        hitl_config = self._app_service.hitl_config

        self._show_idle_kick_warning = False

        connection_records = (
            self._app_service.remote_gui_input.get_new_connection_records()
        )
        if len(connection_records):
            connection_record = connection_records[-1]
            # new connection
            self._client_connection_id = connection_record["connectionId"]
            print(f"new connection_record: {connection_record}")
            if hitl_config.networking.client_max_idle_duration is not None:
                self._client_idle_frame_counter = 0

        if self._client_idle_frame_counter is not None:
            if self.is_user_idle_this_frame():
                self._client_idle_frame_counter += 1
                assert hitl_config.networking.client_max_idle_duration > 0
                max_idle_frames = max(
                    int(
                        hitl_config.networking.client_max_idle_duration
                        * hitl_config.target_sps
                    ),
                    1,
                )

                if self._client_idle_frame_counter > max_idle_frames * 0.5:
                    self._show_idle_kick_warning = True

                if self._client_idle_frame_counter > max_idle_frames:
                    self._app_service.client_message_manager.signal_kick_client(
                        self._client_connection_id
                    )
                    self._client_idle_frame_counter = None
            else:
                # reset counter whenever the client isn't idle
                self._client_idle_frame_counter = 0

    def sim_update(self, dt, post_sim_update_dict):
        # Do NOT let the remote client make the server application exit!
        # if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
        #     self._app_service.end_episode()
        #     post_sim_update_dict["application_exit"] = True

        # # use 1-5 keys to select certain episodes corresponding to our 5 scenes
        # num_fetch_scenes = 5
        # # hand-picked episodes from hitl_vr_sample_episodes.json.gz
        # episode_id_by_scene_index = ["0", "5", "10", "15", "20"]
        # for scene_idx in range(num_fetch_scenes):
        #     key_map = [
        #         GuiInput.KeyNS.ONE,
        #         GuiInput.KeyNS.TWO,
        #         GuiInput.KeyNS.THREE,
        #         GuiInput.KeyNS.FOUR,
        #         GuiInput.KeyNS.FIVE,
        #     ]
        #     key = key_map[scene_idx]
        #     if self._app_service.gui_input.get_key_down(key):
        #         self._app_service.episode_helper.set_next_episode_by_id(
        #             episode_id_by_scene_index[scene_idx]
        #         )
        #         self._app_service.end_episode(do_reset=True)

        self._update_for_remote_client_connect_and_idle()

        if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.P):
            self._paused = not self._paused

        if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.H):
            self._hide_gui_text = not self._hide_gui_text

        if not self._paused:
            self._update_grasping_and_set_act_hints()
            self._app_service.compute_action_and_step_env()
        else:
            # temp hack: manually add a keyframe while paused
            self.get_sim().gfx_replay_manager.save_keyframe()

        self._pick_helper.viz_objects()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()


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
