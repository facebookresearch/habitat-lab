#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import magnum as mn
import numpy as np

from habitat.sims.habitat_simulator import sim_utilities
from habitat.tasks.rearrange.articulated_agent_manager import (
    ArticulatedAgentManager,
)
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.client_helper import ClientHelper
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import MouseButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.core.user_mask import Mask, Users
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
    GuiRobotController,
)
from habitat_hitl.environment.hablab_utils import get_agent_art_obj_transform
from habitat_sim.geo import Ray
from habitat_sim.physics import ManagedArticulatedObject, RayHitInfo
from habitat_sim.utils.common import quat_from_magnum, quat_to_coeffs

from selection import Selection

def upload_file_to_s3(local_file: str, file_name:str , s3_folder:str):
    try:
        import boto3
        # Check if local file exists
        if not os.path.isfile(local_file):
            raise ValueError(f"Local file {local_file} does not exist")

        s3_client = boto3.client('s3')
        if not s3_folder.endswith('/'):
            s3_folder += '/'

        s3_path = os.path.join(s3_folder, file_name)
        s3_path = s3_path.replace(os.path.sep, '/')

        if "S3_BUCKET" in os.environ:
            bucket_name = os.environ["S3_BUCKET"]
            print(f"Uploading {local_file} to {bucket_name}/{s3_path}")
            s3_client.upload_file(local_file, bucket_name, s3_path)
        else:
            print("'S3_BUCKET' environment variable is not set. Cannot upload.")
    except Exception as e:
        print(e)

# Visually snap picked objects into the humanoid's hand. May be useful in third-person mode. Beware that this conflicts with GuiPlacementHelper.
DO_HUMANOID_GRASP_OBJECTS = False

MINIMUM_DROP_VERTICALITY: float = 0.9
DOUBLE_CLICK_DELAY: float = 0.33

_HI = 0.8
_LO = 0.4
COLOR_VALID = mn.Color4(0.0, _HI, 0.0, 1.0)  # Green
COLOR_INVALID = mn.Color4(_HI, 0.0, 0.0, 1.0)  # Red
COLOR_GOALS: List[mn.Color4] = [
    mn.Color4(0.0, _HI, _HI, 1.0),  # Cyan
    mn.Color4(_HI, 0.0, _HI, 1.0),  # Magenta
    mn.Color4(_HI, _HI, 0.0, 1.0),  # Yellow
    mn.Color4(_HI, 0.0, _LO, 1.0),  # Purple
    mn.Color4(_LO, _HI, 0.0, 1.0),  # Orange
]


class DataLogger:
    def __init__(self, app_service):
        self._app_service = app_service
        self._sim = app_service.sim

    def get_num_agents(self):
        return len(self._sim.agents_mgr._all_agent_data)

    def get_agents_state(self):
        agent_states = []
        for agent_idx in range(self.get_num_agents()):
            agent_root = get_agent_art_obj_transform(self._sim, agent_idx)
            position = np.array(agent_root.translation).tolist()
            rotation = mn.Quaternion.from_matrix(agent_root.rotation())
            rotation = quat_to_coeffs(quat_from_magnum(rotation)).tolist()

            snap_idx = self._sim.agents_mgr._all_agent_data[
                agent_idx
            ].grasp_mgr.snap_idx
            agent_states.append(
                {
                    "position": position,
                    "rotation": rotation,
                    "grasp_mgr_snap_idx": snap_idx,
                }
            )
        return agent_states

    def get_objects_state(self):
        object_states = []
        rom = self._sim.get_rigid_object_manager()
        for object_handle, rel_idx in self._sim._handle_to_object_id.items():
            obj_id = self._sim._scene_obj_ids[rel_idx]
            ro = rom.get_object_by_id(obj_id)
            position = np.array(ro.translation).tolist()
            rotation = quat_to_coeffs(quat_from_magnum(ro.rotation)).tolist()
            object_states.append(
                {
                    "position": position,
                    "rotation": rotation,
                    "object_handle": object_handle,
                    "object_id": obj_id,
                }
            )
        return object_states

    def record_state(self, task_completed: bool = False):
        agent_states = self.get_agents_state()
        object_states = self.get_objects_state()

        self._app_service.step_recorder.record("agent_states", agent_states)
        self._app_service.step_recorder.record("object_states", object_states)
        self._app_service.step_recorder.record(
            "task_completed", task_completed
        )

    def record_session_metadata(self, connection_params: Dict[str, Any], session_metadata: Dict[str, Any]):
        self._app_service.step_recorder.record("connection", connection_params)
        self._app_service.step_recorder.record("session", session_metadata)


class AppStateRearrangeV2(AppState):
    """
    Todo
    """
    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._gui_agent_controllers = self._app_service.gui_agent_controllers
        self._num_users = len(self._gui_agent_controllers)
        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )
        self._gui_input = self._app_service.gui_input

        self._sim = app_service.sim
        self._ao_root_bbs: Dict = None
        self._opened_ao_set: Set = set()

        self._cam_transform = None
        self._camera_user_index = 0
        self._paused = self._app_service.hitl_config.networking.enable
        self._show_gui_text = True

        # Hack: We maintain our own episode iterator.
        self._episode_ids: List[str] = ["0"]
        self._current_episode_index = 0
        self._last_episodes_param_str = ""


        self._held_object_id: Optional[int] = None
        self._link_id_to_ao_map: Dict[int, int] = {}
        self._opened_link_set: Set = set()
        self._pickable_object_ids: Set[int] = set()
        self._interactable_object_ids: Set[int] = set()
        self._last_click_time: datetime = datetime.now()
        self._paired_goal_ids: List[Tuple[List[int], List[int]]] = []

        self._episode_start_time = datetime.now()

        self._connection_parameters: Dict[str, Any] = {}

        self._recording = False

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config,
            self._gui_input,
        )

        self._client_helper = None
        if self._app_service.hitl_config.networking.enable:
            self._client_helper = ClientHelper(self._app_service)

        self._has_grasp_preview = False
        self._frame_counter = 0
        self._sps_tracker = AverageRateTracker(2.0)

        self._task_instruction = ""
        self._data_logger = DataLogger(app_service=self._app_service)

        self._selections: List[Selection] = []
        self._hover_selection = Selection(
            self._sim,
            self._gui_input,
            Selection.hover_fn
        )
        self._selections.append(self._hover_selection)
        self._click_selection = Selection(
            self._sim,
            self._gui_input,
            Selection.left_click_fn,
        )
        self._selections.append(self._click_selection)
        def place_selection_fn(gui_input: GuiInput) -> bool:
            return gui_input.get_mouse_button(MouseButton.RIGHT) or gui_input.get_mouse_button_up(MouseButton.RIGHT)
        self._place_selection = Selection(
            self._sim,
            self._gui_input,
            place_selection_fn,
        )
        self._selections.append(self._place_selection)

    # needed to avoid spurious mypy attr-defined errors
    @staticmethod
    def get_sim_utilities() -> Any:
        return sim_utilities

    def _reset_state(self):
        self._held_object_id = None
        sim = self.get_sim()
        self._link_id_to_ao_map = sim_utilities.get_ao_link_id_map(sim)
        self._opened_link_set = set()
        for selection in self._selections:
            selection.deselect()
        
        # HACK: RearrangeSim removes collisions from clutter objects.
        self._pickable_object_ids = set(sim._scene_obj_ids)
        for pickable_obj_id in self._pickable_object_ids:
            rigid_obj = self._get_rigid_object(pickable_obj_id)
            rigid_obj.collidable = True

        # Get set of interactable articulated object links.
        # Exclude all agents.
        agent_ao_object_ids: Set[int] = set()
        agent_manager: ArticulatedAgentManager = sim.agents_mgr
        for agent_index in range(len(agent_manager)):
            agent = agent_manager[agent_index]
            agent_ao = agent.articulated_agent.sim_obj
            agent_ao_object_ids.add(agent_ao.object_id)
            # for link_object_id in agent_ao.link_object_ids:
            #    agent_ao_object_ids.add(link_object_id)
        self._interactable_object_ids = set()
        aom = sim.get_articulated_object_manager()
        all_ao: List[
            ManagedArticulatedObject
        ] = aom.get_objects_by_handle_substring().values()
        # All add non-root links that are not agents.
        for ao in all_ao:
            if ao.object_id not in agent_ao_object_ids:
                for link_object_id in ao.link_object_ids:
                    if link_object_id != ao.object_id:
                        self._interactable_object_ids.add(link_object_id)

    def on_environment_reset(self, episode_recorder_dict):  # Reset state.
        self._reset_state()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

        # Set the task instruction and goals
        self._paired_goal_ids = []
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("extra_info") is not None:
            extra_info = current_episode.info["extra_info"]
            self._task_instruction = extra_info["instruction"]
            for proposition in extra_info["evaluation_propositions"]:
                object_ids: List[int] = []
                object_handles = proposition["args"]["object_handles"]
                for object_handle in object_handles:
                    obj = sim_utilities.get_obj_from_handle(
                        self.get_sim(), object_handle
                    )
                    object_id = obj.object_id
                    object_ids.append(object_id)
                receptacle_ids: List[int] = []
                receptacle_handles = proposition["args"]["receptacle_handles"]
                for receptacle_handle in receptacle_handles:
                    obj = sim_utilities.get_obj_from_handle(
                        self.get_sim(), receptacle_handle
                    )
                    object_id = obj.object_id
                    # TODO: Support for finding links by handle.
                    receptacle_ids.append(object_id)
                self._paired_goal_ids.append((object_ids, receptacle_ids))

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change()

    def get_sim(self):
        return self._app_service.sim

    def _get_gui_agent_translation(self, user_index):
        return get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index(user_index)
        ).translation

    def _set_agent_act_hints(self, user_index):
        drop_pos = None
        grasp_object_id = None
        throw_vel = None
        reach_pos = None
        walk_dir = None
        distance_multiplier = 1.0

        gui_agent_controller = self._gui_agent_controllers[user_index]
        assert isinstance(
            gui_agent_controller, (GuiHumanoidController, GuiRobotController)
        )
        gui_agent_controller.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helper.lookat_offset_yaw,
            throw_vel=throw_vel,
            reach_pos=reach_pos,
        )

        return drop_pos

    def get_gui_controlled_agent_index(self, user_index):
        return self._gui_agent_controllers[user_index]._agent_idx

    def _get_controls_text(self):
        if self._paused:
            return "Session ended."

        controls_str: str = ""
        if self._show_gui_text:
            # if self._sps_tracker.get_smoothed_rate() is not None:
            #    controls_str += f"server SPS: {self._sps_tracker.get_smoothed_rate():.1f}\n"
            # if self._client_helper and self._client_helper.display_latency_ms:
            #    controls_str += f"latency: {self._client_helper.display_latency_ms:.0f}ms\n"
            controls_str += f"Episode: {str(self._current_episode_index)}\n"
            controls_str += "H: Toggle help\n"
            controls_str += "Look: Middle click (drag), I, K\n"
            controls_str += "Walk: W, S\n"
            controls_str += "Turn: A, D\n"
            controls_str += "Finish episode: Zero (0)\n"
            controls_str += "Open/close: Double-click\n"
            controls_str += "Pick object: Double-click\n"
            controls_str += "Place object: Right click (hold)\n"

        return controls_str

    def _get_status_text(self):
        if self._paused:
            return ""
        
        status_str = ""
        if len(self._task_instruction) > 0:
            status_str += "\nInstruction: " + self._task_instruction + "\n"
        if (
            self._client_helper
            and self._client_helper.do_show_idle_kick_warning
        ):
            status_str += (
                "\n\nAre you still there?\nPress any key to keep playing!\n"
            )

        return status_str

    def _update_help_text(self):
        status_str = self._get_status_text()
        if len(status_str) > 0:
            self._app_service.text_drawer.add_text(
                status_str,
                TextOnScreenAlignment.TOP_CENTER,
                text_delta_x=-280,
                text_delta_y=-50,
            )

        controls_str = self._get_controls_text()
        if len(controls_str) > 0:
            self._app_service.text_drawer.add_text(
                controls_str, TextOnScreenAlignment.TOP_LEFT
            )

    def _get_camera_lookat_pos(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(),
            self.get_gui_controlled_agent_index(self._camera_user_index),
        )
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_root.translation + lookat_y_offset
        return lookat

    def is_user_idle_this_frame(self):
        return not self._gui_input.get_any_key_down()

    def _check_change_episode(self):
        if self._paused or not self._gui_input.get_key_down(
            GuiInput.KeyNS.ZERO
        ):
            return

        self._increment_episode()

    def _increment_episode(self):
        if not self._paused:
            self._current_episode_index += 1
            delta_time = datetime.now() - self._episode_start_time
            session_metadata: Dict[str, Any] = {}
            session_metadata["elapsed_time"] = delta_time.total_seconds()
            if self._current_episode_index < len(self._episode_ids):
                self._set_episode(self._current_episode_index)
            else:
                self._end_session()
            self._data_logger.record_session_metadata(self._connection_parameters, session_metadata)

    def _set_episode(self, episode_index: int):
        # If an episode range was received...
        self._episode_start_time = datetime.now()
        if len(self._episode_ids) > 0:
            next_episode_id = self._episode_ids[episode_index]
            self._app_service.episode_helper.set_next_episode_by_id(
                next_episode_id
            )
            self._app_service.end_episode(do_reset=True)

    def _update_episode_set(
        self, connection_parameters: Optional[Dict[str, Any]]
    ):
        if (
            connection_parameters != None
            and "episodes" in connection_parameters
        ):
            episodes_param_str: str = connection_parameters["episodes"]
            if episodes_param_str != self._last_episodes_param_str:
                self._connection_parameters = connection_parameters
                self._last_episodes_param_str = episodes_param_str
                # Format: {lower_bound}-{upper_bound} E.g. 100-110
                # Upper bound is exclusive.
                episode_range_str = episodes_param_str.split("-")
                if len(episode_range_str) == 2:
                    start_episode_id = (
                        int(episode_range_str[0])
                        if episode_range_str[0].isdecimal()
                        else None
                    )
                    last_episode_id = (
                        int(episode_range_str[1])
                        if episode_range_str[0].isdecimal()
                        else None
                    )
                    if start_episode_id != None and last_episode_id != None:
                        # If in decreasing order, swap.
                        if start_episode_id > last_episode_id:
                            temp = last_episode_id
                            last_episode_id = start_episode_id
                            start_episode_id = temp
                        episode_ids = []
                        for episode_id_int in range(
                            start_episode_id, last_episode_id
                        ):
                            episode_ids.append(str(episode_id_int))
                        # Change episode.
                        self._start_session(episode_ids)
    
    def _start_session(self, episode_ids: List[str]):
        assert len(episode_ids) > 0
        # Delete previous output directory
        # TODO: Get the directory from the config value.
        if os.path.exists("output"):
            shutil.rmtree("output")
        self._episode_ids = episode_ids
        self._current_episode_index = 0
        self._recording = True
        self._paused = False
        self._set_episode(self._current_episode_index)

    def _end_session(self):        
        # Save keyframe for kick signal to reach backend.
        # TODO: Upload is blocking and slow. Client hangs for a while.
        if self._client_helper:
            self._client_helper.kick()
        self.get_sim().gfx_replay_manager.save_keyframe()

        self._app_service.end_episode()
        self._recording = False
        self._paused = True
        # TODO: Clean up any remaining memory.
        self._connection_parameters = {}
        self._gui_input.on_frame_end()
        self._app_service.remote_client_state.clear_history()

        # TODO: Move out this block.
        from os import listdir
        from os.path import isfile, join
        output_path = "output"
        output_files = [f for f in listdir(output_path) if isfile(join(output_path, f))]
        for output_file in output_files:
            timestamp = str(int(time.time()))
            output_file_path = os.path.join(output_path, output_file)
            upload_file_to_s3(output_file_path, output_file, f"Phase_0/{timestamp}")
        # Ready for the next user.


    def _update_held_object_placement(self, user_index: int):
        object_id = self._held_object_id
        if not object_id:
            return

        eye_position = self._camera_helper.get_eye_pos()
        forward_vector = (
            self._camera_helper.get_lookat_pos()
            - self._camera_helper.get_eye_pos()
        ).normalized()

        rigid_object = (
            self.get_sim()
            .get_rigid_object_manager()
            .get_object_by_id(object_id)
        )
        rigid_object.translation = eye_position + forward_vector

        # sloppy: save another keyframe here since we just moved the held object
        # self.get_sim().gfx_replay_manager.save_keyframe()

    def _is_object_id_selectable(self, object_id: Optional[int]):
        return object_id and (
            self._is_object_pickable(object_id)
            or self._is_object_interactable(object_id)
        )

    def _user_pos(self) -> mn.Vector3:
        return self._get_gui_agent_translation(user_index=0)

    def _get_rigid_object(self, object_id) -> Any:
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        return rom.get_object_by_id(object_id)

    def _get_articulated_object(self, object_id) -> Any:
        sim = self.get_sim()
        aom = sim.get_articulated_object_manager()
        return aom.get_object_by_id(object_id)

    def _horizontal_distance(self, a: mn.Vector3, b: mn.Vector3) -> float:
        displacement = a - b
        displacement.y = 0.0  # Horizontal plane.
        return displacement.length()

    def _is_object_pickable(self, object_id: int) -> bool:
        return object_id is not None and object_id in self._pickable_object_ids

    def _is_object_interactable(self, object_id: int) -> bool:
        return (
            object_id is not None and
            object_id in self._interactable_object_ids and
            object_id in self._link_id_to_ao_map
        )

    def _is_holding_object(self) -> bool:
        return self._held_object_id is not None

    def _send_error_report(self, user_index: int, clicked_object_id: int) -> None:
        if self._app_service.client_message_manager is None:
            return
        
        sim = self.get_sim()
        clicked_object = sim_utilities.get_obj_from_id(sim, clicked_object_id, self._link_id_to_ao_map)
        handle = clicked_object.handle if clicked_object is not None else ""

        scene_id = sim.curr_scene_name
        episode_id = self._episode_ids[self._current_episode_index]
        task_instruction=self._task_instruction
        sps=self._sps_tracker.get_smoothed_rate()

        msg_mgr = self._app_service.client_message_manager
        msg_mgr.error_report(
            connection_params=self._connection_parameters,
            clicked_object_handle=handle,
            scene_id=scene_id,
            episode_id=episode_id,
            task_instruction=task_instruction,
            sps=sps,
            destination_mask=Mask.from_index(user_index),
        )

    def _update_user_actions(self, user_index: int) -> None:
        def _handle_double_click() -> bool:
            time_since_last_click = datetime.now() - self._last_click_time
            double_clicking = time_since_last_click < timedelta(seconds=DOUBLE_CLICK_DELAY)
            if not double_clicking:
                self._last_click_time = datetime.now()
            return double_clicking
        
        for selection in self._selections:
            selection.update()

        if self._gui_input.get_mouse_button_down(MouseButton.LEFT):
            clicked_object_id = self._click_selection.object_id
            if _handle_double_click():
                # Double-click to select pickable.
                if self._is_object_pickable(clicked_object_id):
                    self._pick_object(clicked_object_id)
                # Double-click to interact.
                elif self._is_object_interactable(clicked_object_id):
                    self._interact_with_object(clicked_object_id)
            else:
                # Click to send error report.
                self._send_error_report(user_index, clicked_object_id)


        # Drop when releasing right click.
        if self._gui_input.get_mouse_button_up(MouseButton.RIGHT):
            self._place_object()
            self._place_selection.deselect()

    def _is_within_reach(
        self, user_pos: mn.Vector3, target_pos: mn.Vector3
    ) -> bool:
        return (
            self._horizontal_distance(user_pos, target_pos)
            < self._can_grasp_place_threshold
        )

    def _is_location_suitable_for_placement(self, normal: mn.Vector3) -> bool:
        placement_verticality = mn.math.dot(normal, mn.Vector3(0, 1, 0))
        placement_valid = placement_verticality > MINIMUM_DROP_VERTICALITY
        return placement_valid

    def _pick_object(self, object_id: int):
        if not self._is_holding_object() and self._is_object_pickable(object_id):
            rigid_object = self._get_rigid_object(object_id)
            if rigid_object is not None:
                rigid_pos = rigid_object.translation
                user_pos = self._user_pos()
                if self._is_within_reach(user_pos, rigid_pos):
                    # Pick the object.
                    self._held_object_id = object_id
                    self._place_selection.deselect()

    def _place_object(self):
        if not self._place_selection.selected:
            return
        
        object_id = self._held_object_id
        place_point = self._place_selection.point
        place_normal = self._place_selection.normal
        if (
            object_id is not None and
            object_id != self._place_selection.object_id and
            self._is_location_suitable_for_placement(place_normal)
        ):
            user_pos = self._user_pos()
            if self._is_within_reach(user_pos, place_point):
                # Drop the object.
                rigid_object = self._get_rigid_object(object_id)
                rigid_object.translation = place_point + mn.Vector3(0.0, rigid_object.collision_shape_aabb.size_y() / 2, 0.0)
                self._held_object_id = None
                self._place_selection.deselect()


    def _interact_with_object(self, object_id: int):
        if self._is_object_interactable(object_id):
            link_id = object_id
            link_index = self._get_link_index(link_id)
            if link_index:
                ao_id = self._link_id_to_ao_map[link_id]
                ao = self._get_articulated_object(ao_id)
                link_node = ao.get_link_scene_node(link_index)
                link_pos = link_node.translation
                user_pos = self._user_pos()
                if self._is_within_reach(user_pos, link_pos):
                    # Open/close object.
                    if link_id in self._opened_link_set:
                        sim_utilities.close_link(ao, link_index)
                        self._opened_link_set.remove(link_id)
                    else:
                        sim_utilities.open_link(ao, link_index)
                        self._opened_link_set.add(link_id)
    
    def _draw_aabb(self, user_index: int, aabb: mn.Range3D, transform: mn.Matrix4, color: mn.Color3):
        self._app_service.gui_drawer.push_transform(
            transform, destination_mask=Mask.from_index(user_index)
        )
        self._app_service.gui_drawer.draw_box(
            min_extent=aabb.back_bottom_left,
            max_extent=aabb.front_top_right,
            color=color,
            destination_mask=Mask.from_index(user_index),
        )
        self._app_service.gui_drawer.pop_transform(
            destination_mask=Mask.from_index(user_index)
        )
    
    def _draw_place_selection(self, user_index: int):
        if not self._place_selection.selected or self._held_object_id is None:
            return
        
        point = self._place_selection.point
        normal = self._place_selection.normal
        user_pos = self._get_gui_agent_translation(user_index)
        placement_valid = self._is_location_suitable_for_placement(normal)
        placement_valid &= self._is_within_reach(point, user_pos)
        color = COLOR_VALID if placement_valid else COLOR_INVALID
        radius = 0.15 if placement_valid else 0.05
        self._app_service.gui_drawer.draw_circle(
            translation=point,
            radius=radius,
            color=color,
            normal=normal,
            billboard=False,
            destination_mask=Mask.from_index(user_index),
        )

    def _draw_hovered_interactable(self, user_index: int):
        if not self._hover_selection.selected:
            return
        
        object_id = self._hover_selection.object_id
        if not self._is_object_interactable(object_id):
            return
        
        link_index = self._get_link_index(object_id)
        if link_index:
            ao = sim_utilities.get_obj_from_id(
                self.get_sim(), object_id, self._link_id_to_ao_map
            )
            link_node = ao.get_link_scene_node(link_index)
            aabb = link_node.cumulative_bb
            user_pos = self._get_gui_agent_translation(user_index)
            reachable = self._is_within_reach(
                user_pos, link_node.translation
            )
            color = COLOR_VALID if reachable else COLOR_INVALID
            self._draw_aabb(user_index, aabb, link_node.transformation, color)

    def _draw_hovered_pickable(self, user_index: int):
        if not self._hover_selection.selected or self._is_holding_object():
            return
        
        object_id = self._hover_selection.object_id
        if not self._is_object_pickable(object_id):
            return
        
        managed_object = sim_utilities.get_obj_from_id(
            self.get_sim(), object_id, self._link_id_to_ao_map
        )
        translation = managed_object.translation
        user_pos = self._get_gui_agent_translation(user_index)
        reachable = self._is_within_reach(user_pos, translation)
        color = COLOR_VALID if reachable else COLOR_INVALID
        aabb = managed_object.collision_shape_aabb
        self._draw_aabb(user_index, aabb, managed_object.transformation, color)

    def _draw_goals(self, user_index: int):
        # TODO: Cache
        for i in range(len(self._paired_goal_ids)):
            rigid_ids = self._paired_goal_ids[i][0]
            receptacle_ids = self._paired_goal_ids[i][1]
            goal_pair_color = COLOR_GOALS[i % len(COLOR_GOALS)]
            for rigid_id in rigid_ids:
                if self._hover_selection.object_id == rigid_id:
                    continue
                managed_object = sim_utilities.get_obj_from_id(
                    self.get_sim(), rigid_id, self._link_id_to_ao_map
                )
                translation = managed_object.translation
                self._app_service.gui_drawer.draw_circle(
                    translation=translation,
                    radius=0.25,
                    color=goal_pair_color,
                    billboard=True,
                    destination_mask=Mask.from_index(user_index),
                )
            for receptacle_id in receptacle_ids:
                managed_object = sim_utilities.get_obj_from_id(
                    self.get_sim(), receptacle_id, self._link_id_to_ao_map
                )
                aabb = None
                if hasattr(managed_object, "collision_shape_aabb"):
                    aabb = managed_object.collision_shape_aabb
                else:
                    link_index = self._get_link_index(receptacle_id)
                    if link_index is not None:
                        link_node = managed_object.get_link_scene_node(
                            link_index
                        )
                        aabb = link_node.cumulative_bb
                if aabb is not None:
                    self._draw_aabb(user_index, aabb, managed_object.transformation, goal_pair_color)

    def _draw_ui(self, user_index: int):
        self._draw_place_selection(user_index)
        self._draw_hovered_interactable(user_index)
        self._draw_hovered_pickable(user_index)
        self._draw_goals(user_index)

    def _get_link_index(self, object_id: int):
        link_id = object_id
        if link_id in self._link_id_to_ao_map:
            ao_id = self._link_id_to_ao_map[link_id]
            ao = self._get_articulated_object(ao_id)
            link_id_to_index: Dict[int, int] = ao.link_object_ids
            if link_id in link_id_to_index:
                return link_id_to_index[link_id]
        return None

    def sim_update(self, dt, post_sim_update_dict):
        if (
            not self._app_service.hitl_config.networking.enable
            and self._gui_input.get_key_down(GuiInput.KeyNS.ESC)
        ):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True
            return
        
        if self._app_service.hitl_config.networking.enable:
            params = (
                self._app_service.remote_client_state.get_connection_parameters()
            )
            self._update_episode_set(params)

        self._sps_tracker.increment()

        if self._client_helper:
            self._client_helper.update(
                self.is_user_idle_this_frame(),
                self._sps_tracker.get_smoothed_rate(),
            )

        if self._gui_input.get_key_down(GuiInput.KeyNS.H):
            self._show_gui_text = not self._show_gui_text

        self._check_change_episode()

        if not self._paused:
            for user_index in range(self._num_users):
                self._update_user_actions(user_index)
                self._draw_ui(user_index)
                self._set_agent_act_hints(user_index)
                self._update_held_object_placement(user_index)
            self._app_service.compute_action_and_step_env()
            self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()

    def record_state(self):
        task_completed = self._gui_input.get_key_down(
            GuiInput.KeyNS.ZERO
        )
        self._data_logger.record_state(task_completed=task_completed)


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
