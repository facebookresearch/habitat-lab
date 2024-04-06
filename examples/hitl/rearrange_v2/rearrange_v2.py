#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from habitat.tasks.rearrange.articulated_agent_manager import ArticulatedAgentManager
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_hitl.core.key_mapping import MouseButton
from habitat_hitl.core.user_mask import Mask
import hydra
import magnum as mn

import habitat_sim
from habitat.sims.habitat_simulator import sim_utilities
from habitat_sim.geo import Ray
from habitat_sim.physics import RayHitInfo, ManagedArticulatedObject, ManagedRigidObject
from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.app_states.app_service import AppService
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
from habitat_hitl.environment.hablab_utils import get_agent_art_obj, get_agent_art_obj_transform

# Visually snap picked objects into the humanoid's hand. May be useful in third-person mode. Beware that this conflicts with GuiPlacementHelper.
DO_HUMANOID_GRASP_OBJECTS = False


MINIMUM_DROP_VERTICALITY: float = 0.9
DOUBLE_CLICK_DELAY: float = 0.33

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

        self._sim = app_service.sim
        self._ao_root_bbs: Dict = None
        self._opened_ao_set: Set = set()

        self._cam_transform = None
        self._camera_user_index = 0
        self._held_obj_id: Optional[int] = None
        self._recent_reach_pos = None
        self._paused = False
        self._hide_gui_text = False

        # New stuff.
        #self._clicked_object_id: Optional[int] = None
        self._selected_object_id: Optional[int] = None
        self._hovered_object_id: Optional[int] = None
        self._selected_point: Optional[mn.Vector3] = None
        self._drop_ao_link_node: Optional[Any] = None
        self._hovered_point: Optional[mn.Vector3] = None
        self._selected_normal: Optional[mn.Vector3] = None
        self._hovered_normal: Optional[mn.Vector3] = None
        self._held_object_id: Optional[int] = None
        self._link_id_to_ao_map: Dict[int, int] = {}
        self._opened_link_set: Set = set()
        self._pickable_object_ids: Set[int] = set()
        self._interactable_object_ids: Set[int] = set()
        
        # Object that is being double-clicked
        self._interacting_object_id: Optional[int] = None
        self._interacting_timestamp: Optional[datetime] = None

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config,
            self._app_service.gui_input,
        )

        self._pick_helper = GuiPickHelper(self._app_service, user_index=0)
        #self._placement_helper = GuiPlacementHelper(
        #    self._app_service, user_index=0
        #)
        self._client_helper = None
        if self._app_service.hitl_config.networking.enable:
            self._client_helper = ClientHelper(self._app_service)

        self._frame_counter = 0
        self._sps_tracker = AverageRateTracker(2.0)

        self._task_instruction = ""

    # needed to avoid spurious mypy attr-defined errors
    @staticmethod
    def get_sim_utilities() -> Any:
        return sim_utilities

    def _reset_state(self):
        sim = self.get_sim()
        self._link_id_to_ao_map = sim_utilities.get_ao_link_id_map(sim)
        self._opened_link_set = set()
        self._reset_selection()
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
            #for link_object_id in agent_ao.link_object_ids:
            #    agent_ao_object_ids.add(link_object_id)
        self._interactable_object_ids = set()
        aom = sim.get_articulated_object_manager()
        all_ao: List[ManagedArticulatedObject] = aom.get_objects_by_handle_substring().values()
        # All add non-root links that are not agents.
        for ao in all_ao:
            if ao.object_id not in agent_ao_object_ids:
                for link_object_id in ao.link_object_ids:
                    if link_object_id != ao.object_id:
                        self._interactable_object_ids.add(link_object_id)




    def on_environment_reset(self, episode_recorder_dict):
        # Reset state.
        self._reset_state()

        self._held_obj_id = None

        self._pick_helper.on_environment_reset()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

        # Set the task instruction
        current_episode = self._app_service.env.current_episode
        if current_episode.info.get("extra_info") is not None:
            self._task_instruction = current_episode.info["extra_info"][
                "instruction"
            ]

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
        def get_grasp_release_controls_text():
            if self._held_object_id is not None:
                if self._selected_point is None:
                    return "Right click: Set object placement location.\n"
                else:
                    return "Space: Place object.\n"
            elif self._selected_object_id is None:
                return "Left click: Select object to pick up.\n"
            elif self._selected_object_id is not None:
                return "Space: Pick up object.\n"
            else:
                return ""

        controls_str: str = ""
        if not self._hide_gui_text:
            if self._sps_tracker.get_smoothed_rate() is not None:
                controls_str += f"server SPS: {self._sps_tracker.get_smoothed_rate():.1f}\n"
            if self._client_helper and self._client_helper.display_latency_ms:
                controls_str += f"latency: {self._client_helper.display_latency_ms:.0f}ms\n"
            controls_str += "H: show/hide help text\n"
            controls_str += "P: pause\n"
            controls_str += "I, K: look up, down\n"
            controls_str += "A, D: turn\n"
            controls_str += "W, S: walk\n"
            controls_str += "N: next episode\n"
            controls_str += "Left click: open/close receptacle\n"
            controls_str += get_grasp_release_controls_text()
            #if self._num_users > 1 and self._held_obj_id is None:
            #    controls_str += "T: toggle camera user\n"

        return controls_str

    def _get_status_text(self):
        status_str = ""

        if len(self._task_instruction) > 0:
            status_str += "\nInstruction: " + self._task_instruction + "\n"
        if self._paused:
            status_str += "\n\npaused\n"
        if (
            self._client_helper
            and self._client_helper.do_show_idle_kick_warning
        ):
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
                text_delta_x=-280,
                text_delta_y=-50,
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
        return not self._app_service.gui_input.get_any_key_down()

    def _check_change_episode(self):
        if self._paused or not self._app_service.gui_input.get_key_down(
            GuiInput.KeyNS.N
        ):
            return

        if self._app_service.episode_helper.next_episode_exists():
            self._app_service.end_episode(do_reset=True)

    def _update_held_object_placement(self):
        object_id = self._held_object_id
        if not object_id:
            return

        eye_position = self._camera_helper.get_eye_pos()
        forward_vector = (
            self._camera_helper.get_lookat_pos()
            - self._camera_helper.get_eye_pos()
        ).normalized()

        rigid_object = self.get_sim().get_rigid_object_manager().get_object_by_id(
            object_id
        )
        rigid_object.translation = eye_position + forward_vector

        # sloppy: save another keyframe here since we just moved the held object
        #self.get_sim().gfx_replay_manager.save_keyframe()

    def _is_object_id_selectable(self, object_id: Optional[int]):
        return object_id and (
                self._is_object_pickable(object_id) or
                self._is_object_interactable(object_id)
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

    def _raycast(self, ray: Ray) -> Optional[RayHitInfo]:
        raycast_results = self.get_sim().cast_ray(ray=ray)
        if not raycast_results.has_hits():
            return None
        # Results are sorted by distance. [0] is the nearest one.
        hit_info = raycast_results.hits[0]
        return hit_info

    def _horizontal_distance(self, a: mn.Vector3, b: mn.Vector3) -> float:
        displacement = a - b
        displacement.y = 0.0  # Horizontal plane.
        return displacement.length()
    
    def _is_object_pickable(self, object_id: int) -> bool:
        return object_id and object_id in self._pickable_object_ids
    
    def _is_object_interactable(self, object_id: int) -> bool:
        return object_id and object_id in self._interactable_object_ids

    def _mouse_controls(self) -> None:
        ray = self._app_service.gui_input.mouse_ray
        if ray is not None:
            hit_info = self._raycast(ray)
            if hit_info is None:
                return
            object_id: int = hit_info.object_id

            # Hover.
            self._hovered_object_id = object_id
            self._hovered_point = hit_info.point
            self._hovered_normal = hit_info.normal

            # Left click.
            if self._app_service.gui_input.get_mouse_button_down(MouseButton.LEFT):
                # Click to select pickable.
                if self._is_object_pickable(object_id):
                    self._selected_object_id = object_id
                # Double-click to interact with interactable.
                # TODO: Single-click is probably enough.
                # TODO: Reach.
                elif self._is_object_interactable(object_id):
                    if self._interacting_object_id == object_id:
                        time_since_last_click = datetime.now() - self._interacting_timestamp
                        if time_since_last_click < timedelta(seconds=DOUBLE_CLICK_DELAY):
                            self._interact_with_object(object_id)
                    self._interacting_object_id = object_id
                    self._interacting_timestamp = datetime.now()

            # Select point.
            if self._app_service.gui_input.get_mouse_button(MouseButton.RIGHT):
                self._selected_point = hit_info.point
                self._selected_normal = hit_info.normal
                # If the point is on a link, record it so that the object can be parented.
                # TODO: rigid_object does not expose its node.
                #if self._is_object_interactable(self._hovered_object_id):
                #    link_index = self._get_link_index(self._hovered_object_id)
                #    ao = sim_utilities.get_obj_from_id(self.get_sim(), self._hovered_object_id, self._link_id_to_ao_map)
                #    link_node = ao.get_link_scene_node(link_index)
                #    self._drop_ao_link_node = link_node

            # Clear selection.
            if self._app_service.gui_input.get_mouse_button_down(MouseButton.MIDDLE):
                self._selected_point = None
                self._selected_object_id = None
                self._selected_normal = None

    def _interact_with_object(self, object_id: int):
        if object_id in self._link_id_to_ao_map:
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
    
    def _pick_object(self, object_id: int):
        # Grab an object.
        assert self._held_object_id is None
        if object_id is not None and self._is_object_pickable(object_id):
            rigid_object = self._get_rigid_object(object_id)
            if rigid_object is not None:
                rigid_pos = rigid_object.translation
                user_pos = self._user_pos()
                if self._is_within_reach(user_pos, rigid_pos):
                    # Pick the object.
                    self._held_object_id = object_id
                    self._selected_object_id = None
                    # Unparent
                    # TODO: rigid_object does not expose its node.
                    #rigid_object.node.set_parent(None)

    def _is_within_reach(self, user_pos: mn.Vector3, target_pos: mn.Vector3) -> bool:
        return self._horizontal_distance(user_pos, target_pos) < self._can_grasp_place_threshold
    
    def _is_location_suitable_for_placement(self, normal: mn.Vector3) -> bool:
        placement_verticality = mn.math.dot(normal, mn.Vector3(0, 1, 0))
        placement_valid = placement_verticality > MINIMUM_DROP_VERTICALITY
        return placement_valid

    def _place_object(self):
        # Place an object
        object_id = self._held_object_id
        assert object_id is not None
        if self._selected_point is not None and self._is_location_suitable_for_placement(self._selected_normal):
            # Get the distance to the candidate drop point on the XZ plane.
            user_pos = self._user_pos()
            if self._is_within_reach(user_pos, self._selected_point):
                # Drop the object.
                rigid_object = self._get_rigid_object(object_id)
                rigid_object.translation = self._selected_point
                self._held_object_id = None
                self._reset_selection()
                # Set parent. This allows the object to follow, for example, a closing drawer.
                # TODO: rigid_object does not expose its node.
                #if self._drop_ao_link_node:
                #    rigid_object.node.set_parent(self._drop_ao_link_node)

    def _pick_place(self):
        # Pick, place.
        if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.SPACE):
            # If object is held:
            if self._held_object_id is not None:
                self._place_object()
            # If selection is an object:
            elif self._selected_object_id is not None:
                if self._is_object_pickable(self._selected_object_id):
                    self._pick_object(self._selected_object_id)


    def _draw_ui(self):
        # Highlight selected point.
        if self._selected_point is not None:
            placement_verticality = mn.math.dot(self._selected_normal, mn.Vector3(0, 1, 0))
            placement_valid = placement_verticality > 0.90
            color = mn.Color4.green() if placement_valid else mn.Color4.red()
            radius = 0.15 if placement_valid else 0.05
            self._app_service.gui_drawer.draw_circle(
                translation=self._selected_point,
                radius=radius,
                color=color,
                normal=self._selected_normal,
                billboard=False
            )

        # Highlight selected object.
        if self._selected_object_id:
            managed_object = sim_utilities.get_obj_from_id(self.get_sim(), self._selected_object_id, self._link_id_to_ao_map)
            translation: Optional[mn.Vector3] = managed_object.translation
            self._app_service.gui_drawer.draw_circle(
                translation=translation,
                radius=0.25, 
                color=mn.Color4.green(),
                billboard=True
            )

        # Highlight hovered articulated object link.
        hovered_id = self._hovered_object_id
        if hovered_id and self._is_object_interactable(hovered_id):
            link_index = self._get_link_index(hovered_id)
            if link_index:
                ao = sim_utilities.get_obj_from_id(self.get_sim(), hovered_id, self._link_id_to_ao_map)
                link_node = ao.get_link_scene_node(link_index)
                aabb = link_node.cumulative_bb
                color = mn.Color4.green()
                self._app_service.gui_drawer.push_transform(link_node.transformation)
                self._app_service.gui_drawer.draw_box(
                    min_extent=aabb.back_bottom_left,
                    max_extent=aabb.front_top_right,
                    color=color,
                    destination_mask=Mask.ALL,
                )
                self._app_service.gui_drawer.pop_transform()

        # Highlight hovered pickable object.
        hovered_id = self._hovered_object_id
        if hovered_id and self._is_object_pickable(hovered_id) and self._held_object_id is None:
            managed_object = sim_utilities.get_obj_from_id(self.get_sim(), hovered_id, self._link_id_to_ao_map)
            translation: Optional[mn.Vector3] = managed_object.translation
            aabb = managed_object.collision_shape_aabb
            color = mn.Color4.green()
            self._app_service.gui_drawer.push_transform(managed_object.transformation)
            self._app_service.gui_drawer.draw_box(
                min_extent=aabb.back_bottom_left,
                max_extent=aabb.front_top_right,
                color=color,
                destination_mask=Mask.ALL,
            )
            self._app_service.gui_drawer.pop_transform()

    def _reset_selection(self):
        self._selected_object_id = None
        self._hovered_object_id = None
        self._selected_point = None
        self._hovered_point = None
        self._selected_normal = None
        self._hovered_normal = None
        self._drop_ao_link_node = None

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
            and self._app_service.gui_input.get_key_down(GuiInput.KeyNS.ESC)
        ):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True
            return

        self._sps_tracker.increment()

        if self._client_helper:
            self._client_helper.update(
                self.is_user_idle_this_frame(),
                self._sps_tracker.get_smoothed_rate(),
            )

        if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.P):
            self._paused = not self._paused

        if self._app_service.gui_input.get_key_down(GuiInput.KeyNS.H):
            self._hide_gui_text = not self._hide_gui_text

        self._check_change_episode()

        if not self._paused:
            for user_index in range(self._num_users):
                self._mouse_controls()
                self._pick_place()
                self._draw_ui()
                self._set_agent_act_hints(user_index)
                self._update_held_object_placement()
            self._app_service.compute_action_and_step_env()
        else:
            # temp hack: manually add a keyframe while paused
            self.get_sim().gfx_replay_manager.save_keyframe()

        if self._held_object_id is None:
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
