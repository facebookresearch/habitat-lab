#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Final, List, Optional

import hydra
import magnum as mn
import numpy as np

from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.app_states.app_state_abc import AppState
from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.text_drawer import TextOnScreenAlignment
from habitat_hitl.environment.avatar_switcher import AvatarSwitcher
from habitat_hitl.environment.camera_helper import CameraHelper
from habitat_hitl.environment.controllers.gui_controller import (
    GuiHumanoidController,
)
from habitat_hitl.environment.gui_navigation_helper import GuiNavigationHelper
from habitat_hitl.environment.gui_throw_helper import GuiThrowHelper
from habitat_hitl.environment.hablab_utils import (
    get_agent_art_obj_transform,
    get_grasped_objects_idxs,
)
from habitat_sim.physics import MotionType

COLOR_GRASPABLE: Final[mn.Color3] = mn.Color3(1, 0.75, 0)
COLOR_GRASP_PREVIEW: Final[mn.Color3] = mn.Color3(0.5, 1, 0)
COLOR_FOCUS_OBJECT: Final[mn.Color3] = mn.Color3(1, 1, 1)
RADIUS_GRASPABLE: Final[float] = 0.15
RADIUS_GRASP_PREVIEW: Final[float] = 0.15
RADIUS_FOCUS_OBJECT: Final[float] = 0.2
RING_PULSE_SIZE: Final[float] = 0.03
MIN_STEPS_STOP: Final[int] = 15
GRASP_THRESHOLD: Final[float] = 0.2


class AppStatePickThrowVr(AppState):
    """
    This app state allows to evaluate a Spot robot, interacting with a GUI-controlled human.
    The human can pick up and throw objects.
    The human can either be controlled with mouse and keyboard, or a VR headset.
    See VR_HITL.md for instructions on controlling the human from a VR device.
    """

    def __init__(self, app_service: AppService):
        self._app_service = app_service
        self._gui_agent_ctrl: Any = (
            self._app_service.gui_agent_controllers[0]
            if len(self._app_service.gui_agent_controllers)
            else None
        )
        self._can_grasp_place_threshold = (
            self._app_service.hitl_config.can_grasp_place_threshold
        )
        # Activate the remote user.
        app_service.users.activate_user(0)

        self._cam_transform: Optional[mn.Matrix4] = None
        self._held_target_obj_idx: Optional[int] = None
        self._recent_reach_pos: Optional[mn.Vector3] = None
        self._paused: bool = False
        self._hide_gui_text: bool = False

        # Index of the remote-controlled hand holding an object
        self._remote_held_hand_idx: Optional[int] = None

        # will be set in on_environment_reset
        self._target_obj_ids: Optional[List[str]] = None

        self._camera_helper = CameraHelper(
            self._app_service.hitl_config, self._app_service.gui_input
        )
        # not supported for pick_throw_vr
        assert not self._app_service.hitl_config.camera.first_person_mode

        self._nav_helper = GuiNavigationHelper(
            self._app_service,
            self.get_gui_controlled_agent_index(),
            user_index=0,
        )
        self._throw_helper = GuiThrowHelper(
            self._app_service, self.get_gui_controlled_agent_index()
        )

        self._avatar_switch_helper = AvatarSwitcher(
            self._app_service, self._gui_agent_ctrl
        )

        self._is_remote_active_toggle: bool = False
        self._count_tsteps_stop: int = 0
        self._has_grasp_preview: bool = False

    def _is_remote_active(self):
        return self._is_remote_active_toggle

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
        self._held_target_obj_idx = None

        sim = self.get_sim()
        self._target_obj_ids = sim._scene_obj_ids

        self._nav_helper.on_environment_reset()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)
        self._count_tsteps_stop = 0

        human_pos = (
            self.get_sim()
            .get_agent_data(self.get_gui_controlled_agent_index())
            .articulated_agent.base_pos
        )

        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.change_humanoid_position(human_pos)
            client_message_manager.signal_scene_change()
            client_message_manager.update_navmesh_triangles(
                self._get_navmesh_triangle_vertices()
            )

    def get_sim(self):
        return self._app_service.sim

    def get_grasp_keys_by_hand(self, hand_idx):
        if hand_idx == 0:
            return [KeyCode.ZERO, KeyCode.ONE]
        else:
            assert hand_idx == 1
            return [KeyCode.TWO, KeyCode.THREE]

    def _try_grasp_remote(self):
        assert not self._held_target_obj_idx
        self._recent_reach_pos = None
        self._recent_hand_idx = None
        remote_client_state = self._app_service.remote_client_state

        hand_positions = []
        num_hands = 2
        for i in range(num_hands):
            hand_pos, _ = remote_client_state.get_hand_pose(
                user_index=0, hand_idx=i
            )
            if hand_pos:
                hand_positions.append(hand_pos)
        if len(hand_positions) == 0:
            return
        assert len(hand_positions) == num_hands

        grasped_objects_idxs = get_grasped_objects_idxs(self.get_sim())

        remote_button_input = remote_client_state.get_gui_input()

        found_obj_idx = None
        found_hand_idx = None
        self._recent_reach_pos = None

        self._has_grasp_preview = False
        for hand_idx in range(num_hands):
            hand_pos = hand_positions[hand_idx]

            min_dist = GRASP_THRESHOLD
            min_i = None
            for i in range(len(self._target_obj_ids)):
                # object is already grasped by Spot
                if i in grasped_objects_idxs:
                    continue

                this_target_pos = self._get_target_object_position(i)

                dist = (this_target_pos - hand_pos).length()
                if dist < min_dist:
                    min_dist = dist
                    min_i = i

            if min_i is not None:
                self._add_target_object_highlight_ring(
                    min_i, COLOR_GRASP_PREVIEW, radius=RADIUS_GRASP_PREVIEW
                )
                self._has_grasp_preview = True

                for key in self.get_grasp_keys_by_hand(hand_idx):
                    if remote_button_input.get_key(key):
                        found_obj_idx = min_i
                        found_hand_idx = hand_idx
                        break

            if found_obj_idx is not None:
                break

        if found_obj_idx is None:
            # Track one of the hands
            for hand_idx in range(num_hands):
                hand_pos = hand_positions[hand_idx]
                for key in self.get_grasp_keys_by_hand(hand_idx):
                    if remote_button_input.get_key(key):
                        # Hand index/pos before we grab anything
                        self._recent_reach_pos = hand_pos
                        self._recent_hand_idx = hand_idx
            return

        self._held_target_obj_idx = found_obj_idx
        self._remote_held_hand_idx = found_hand_idx
        rom_obj = self._get_target_rigid_object(self._held_target_obj_idx)
        rom_obj.motion_type = MotionType.KINEMATIC

    def _update_held_and_try_throw_remote(self):
        assert self._held_target_obj_idx is not None
        assert self._remote_held_hand_idx is not None

        remote_client_state = self._app_service.remote_client_state
        remote_button_input = remote_client_state.get_gui_input()

        do_throw = False
        for key in self.get_grasp_keys_by_hand(self._remote_held_hand_idx):
            if remote_button_input.get_key_up(key):
                do_throw = True

        rom_obj = self._get_target_rigid_object(self._held_target_obj_idx)

        if do_throw:
            rom_obj.motion_type = MotionType.DYNAMIC
            rom_obj.collidable = True

            hand_idx = self._remote_held_hand_idx
            history_len = remote_client_state.get_history_length()
            assert history_len >= 2
            history_offset = 1
            pos1, _ = remote_client_state.get_hand_pose(
                user_index=0, hand_idx=hand_idx, history_index=history_offset
            )
            pos0, _ = remote_client_state.get_hand_pose(
                user_index=0, hand_idx=hand_idx, history_index=history_len - 1
            )
            if pos0 and pos1:
                vel = (pos1 - pos0) / (
                    remote_client_state.get_history_timestep()
                    * (history_len - history_offset)
                )
                rom_obj.linear_velocity = vel
            else:
                rom_obj.linear_velocity = mn.Vector3(0, 0, 0)

            self._held_target_obj_idx = None
            self._remote_held_hand_idx = None
        else:
            # snap to hand
            hand_pos, hand_rotation = remote_client_state.get_hand_pose(
                user_index=0, hand_idx=self._remote_held_hand_idx
            )
            assert hand_pos is not None

            rom_obj.translation = hand_pos
            rom_obj.rotation = hand_rotation
            rom_obj.linear_velocity = mn.Vector3(0, 0, 0)

    def _update_grasping_and_set_act_hints_remote(self):
        if self._held_target_obj_idx is None:
            self._try_grasp_remote()
        else:
            self._recent_reach_pos = None
            self._recent_hand_idx = None
            self._update_held_and_try_throw_remote()

        (
            walk_dir,
            distance_multiplier,
            forward_dir,
        ) = self._nav_helper.get_humanoid_walk_hints_from_remote_client_state(
            visualize_path=False
        )

        # Count number of steps since we stopped, this is to reduce jitter
        # with IK
        if distance_multiplier == 0:
            self._count_tsteps_stop += 1
        else:
            self._count_tsteps_stop = 0

        reach_pos = None
        hand_idx = None
        if (
            self._held_target_obj_idx is not None
            and distance_multiplier == 0.0
            and self._count_tsteps_stop > MIN_STEPS_STOP
        ):
            reach_pos = self._get_target_object_position(
                self._held_target_obj_idx
            )
            hand_idx = self._remote_held_hand_idx
        elif self._recent_reach_pos:
            # Track the state of the hand when trying to reach an object
            reach_pos = self._recent_reach_pos
            hand_idx = self._recent_hand_idx

        grasp_object_id = None
        drop_pos = None
        throw_vel = None

        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        self._gui_agent_ctrl.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helper.lookat_offset_yaw,
            throw_vel=throw_vel,
            reach_pos=reach_pos,
            hand_idx=hand_idx,
            target_dir=forward_dir,
        )

    def _update_grasping_and_set_act_hints(self):
        if self._is_remote_active():
            self._update_grasping_and_set_act_hints_remote()
        else:
            self._update_grasping_and_set_act_hints_local()

    def _update_grasping_and_set_act_hints_local(self):
        drop_pos = None
        grasp_object_id = None
        throw_vel = None
        reach_pos = None

        self._has_grasp_preview = False

        if self._held_target_obj_idx is not None:
            if self._app_service.gui_input.get_key(KeyCode.SPACE):
                if self._recent_reach_pos:
                    # Continue reaching towards the recent reach position (the
                    # original position of the grasped object before it was snapped
                    # into our hand).
                    reach_pos = self._recent_reach_pos
                else:
                    # Reach towards our held object. This doesn't make a lot of sense,
                    # but it creates a kind of hacky reaching animation over time,
                    # because as we reach, the object in our hand moves, affecting our
                    # reach pose on the next frame.
                    reach_pos = self._get_target_object_position(
                        self._held_target_obj_idx
                    )

                    # also preview throw
                    _ = self._throw_helper.viz_and_get_humanoid_throw()

            if self._app_service.gui_input.get_key_up(KeyCode.SPACE):
                if self._recent_reach_pos:
                    # this spacebar release means we've finished the reach-and-grasp motion
                    self._recent_reach_pos = None
                else:
                    # this spacebar release means we've finished the throwing motion
                    throw_obj_id = (
                        self._gui_agent_ctrl._get_grasp_mgr().snap_idx
                    )

                    # The robot can only pick up the object if there is a throw_obj_id
                    # throw_obj_id will be None if the users fast press and press again the space
                    if throw_obj_id is not None:
                        throw_vel = (
                            self._throw_helper.viz_and_get_humanoid_throw()
                        )
                        if throw_vel:
                            self._held_target_obj_idx = None
        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:
                assert not self._gui_agent_ctrl.is_grasped
                translation = self._gui_agent_ctrl.get_base_translation()

                min_dist = self._can_grasp_place_threshold
                min_i = None

                # find closest target object within our distance threshold
                for i in range(len(self._target_obj_ids)):
                    this_target_pos = self._get_target_object_position(i)

                    # compute distance in xz plane
                    offset = this_target_pos - translation
                    offset.y = 0
                    dist_xz = offset.length()
                    if dist_xz < min_dist:
                        min_dist = dist_xz
                        min_i = i

                if min_i is not None:
                    self._add_target_object_highlight_ring(
                        min_i, COLOR_GRASP_PREVIEW, radius=RADIUS_GRASP_PREVIEW
                    )
                    self._has_grasp_preview = True

                    if self._app_service.gui_input.get_key_down(KeyCode.SPACE):
                        self._recent_reach_pos = (
                            self._get_target_object_position(min_i)
                        )
                        # we will reach towards this position until spacebar is released
                        reach_pos = self._recent_reach_pos

                        self._held_target_obj_idx = min_i
                        grasp_object_id = self._target_obj_ids[
                            self._held_target_obj_idx
                        ]

        walk_dir = None
        distance_multiplier = 1.0

        if self._app_service.gui_input.get_mouse_button(MouseButton.RIGHT):
            (
                candidate_walk_dir,
                candidate_distance_multiplier,
            ) = self._nav_helper.get_humanoid_walk_hints_from_ray_cast(
                visualize_path=True
            )
            walk_dir = candidate_walk_dir
            distance_multiplier = candidate_distance_multiplier

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

    def _get_target_rigid_object(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id)

    def _fix_physics_for_target_objects(self):
        for i in range(len(self._target_obj_ids)):
            rom_obj = self._get_target_rigid_object(i)
            pos = rom_obj.translation
            if pos.y < 0.0:
                pos.y = 0
                # beware rom_obj.translation.y = 0 doesn't work as you'd expect
                rom_obj.translation = pos
                vel = rom_obj.linear_velocity
                vel.y = 0
                rom_obj.linear_velocity = vel

    def _get_target_object_position(self, target_obj_idx):
        return self._get_target_rigid_object(target_obj_idx).translation

    def _get_target_object_positions(self):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        return np.array(
            [
                rom.get_object_by_id(obj_id).translation
                for obj_id in self._target_obj_ids
            ]
        )

    def _draw_circle(self, pos, color, radius):
        self._app_service.gui_drawer.draw_circle(
            pos,
            radius,
            color,
        )

    def _add_target_object_highlight_ring(
        self, target_obj_idx, color, radius, do_pulse=False
    ):
        pos = self._get_target_object_position(target_obj_idx)
        self._add_target_highlight_ring(pos, color, radius, do_pulse)

    def _add_target_highlight_ring(self, pos, color, radius, do_pulse=False):
        if do_pulse:
            radius += self._app_service.get_anim_fraction() * RING_PULSE_SIZE

        if (
            self._app_service.client_message_manager
            and self._is_remote_active()
        ):
            client_radius = radius
            self._app_service.client_message_manager.add_highlight(
                pos, client_radius
            )

        self._draw_circle(pos, color, radius)

    def _viz_objects(self):
        focus_obj_idx = None
        if self._held_target_obj_idx is not None:
            focus_obj_idx = self._held_target_obj_idx

        if focus_obj_idx is None:
            # only show graspable objects if we aren't showing a grasp preview
            if not self._has_grasp_preview:
                for i in range(len(self._target_obj_ids)):
                    self._add_target_object_highlight_ring(
                        i,
                        COLOR_GRASPABLE,
                        radius=RADIUS_GRASPABLE,
                        do_pulse=True,
                    )
        else:
            self._add_target_object_highlight_ring(
                focus_obj_idx, COLOR_FOCUS_OBJECT, radius=RADIUS_FOCUS_OBJECT
            )

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def _get_gui_agent_feet_height(self):
        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        base_offset = (
            self._gui_agent_ctrl.get_articulated_agent().params.base_offset
        )
        agent_feet_translation = (
            self._gui_agent_ctrl.get_base_translation() + base_offset
        )
        return agent_feet_translation[1]

    def _get_controls_text(self):
        def get_grasp_release_controls_text():
            if self._held_target_obj_idx is not None:
                return "Spacebar: throw\n"
            elif self._has_grasp_preview:
                return "Spacebar: pick up\n"
            else:
                return ""

        controls_str: str = ""
        if not self._hide_gui_text:
            controls_str += "H: show.hide help text\n"
            controls_str += "1-5: select scene\n"
            controls_str += "R + drag: rotate camera\n"
            controls_str += "Scroll: zoom\n"
            if self._app_service.hitl_config.networking.enable:
                controls_str += "T: toggle keyboard.VR\n"
            controls_str += "P: pause\n"
            if not self._is_remote_active_toggle:
                controls_str += "Right-click: walk\n"
                controls_str += "WASD: walk\n"
                controls_str += get_grasp_release_controls_text()

        return controls_str

    def _get_status_text(self):
        status_str = ""

        if self._app_service.hitl_config.networking.enable:
            status_str += (
                "human control: VR\n"
                if self._is_remote_active()
                else "human control: keyboard\n"
            )

        if self._paused:
            status_str += "\npaused\n"

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
        agent_pos, _ = self._get_agent_pose()
        lookat_y_offset = mn.Vector3(0, 0, 0)
        lookat = agent_pos + lookat_y_offset
        return lookat

    def sim_update(self, dt, post_sim_update_dict):
        if self._app_service.gui_input.get_key_down(KeyCode.ESC):
            self._app_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        # use 1-5 keys to select certain episodes corresponding to our 5 scenes
        num_fetch_scenes = 5
        # hand-picked episodes from hitl_vr_sample_episodes.json.gz
        episode_id_by_scene_index = ["0", "5", "10", "15", "20"]
        for scene_idx in range(num_fetch_scenes):
            key_map = [
                KeyCode.ONE,
                KeyCode.TWO,
                KeyCode.THREE,
                KeyCode.FOUR,
                KeyCode.FIVE,
            ]
            key = key_map[scene_idx]
            if self._app_service.gui_input.get_key_down(key):
                self._app_service.episode_helper.set_next_episode_by_id(
                    episode_id_by_scene_index[scene_idx]
                )
                self._app_service.end_episode(do_reset=True)

        if self._app_service.gui_input.get_key_down(KeyCode.P):
            self._paused = not self._paused

        if self._app_service.gui_input.get_key_down(KeyCode.H):
            self._hide_gui_text = not self._hide_gui_text

        # toggle remote/local under certain conditions:
        # - must not be holding anything
        # - toggle on T keypress OR switch to remote if any remote button is pressed
        if (
            self._app_service.hitl_config.networking.enable
            and self._held_target_obj_idx is None
            and (
                self._app_service.gui_input.get_key_down(KeyCode.T)
                or (
                    not self._is_remote_active_toggle
                    and self._app_service.remote_client_state.get_gui_input().get_any_key_down()
                )
            )
        ):
            self._is_remote_active_toggle = not self._is_remote_active_toggle

        if self._app_service.gui_input.get_key_down(KeyCode.TAB):
            self._avatar_switch_helper.switch_avatar()

        if not self._paused:
            self._update_grasping_and_set_act_hints()
            self._app_service.compute_action_and_step_env()
            self._fix_physics_for_target_objects()

        self._viz_objects()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()

    def cast_ray(self, ray):
        if not ray:
            return None

        # special logic for raycasts originating from above the ceiling
        ceiling_y = 2.0
        if ray.origin.y >= ceiling_y:
            if ray.direction.y >= 0:
                return None

            # hack move ray below ceiling (todo: base this on humanoid agent base y, so that it works in multi-floor homes)
            if ray.origin.y < ceiling_y:
                return None

            dist_to_raycast_start_y = (
                ray.origin.y - ceiling_y
            ) / -ray.direction.y
            assert dist_to_raycast_start_y >= 0
            adjusted_origin = (
                ray.origin + ray.direction * dist_to_raycast_start_y
            )
            ray.origin = adjusted_origin

        # reference code for casting a ray into the scene
        raycast_results = self.get_sim().cast_ray(ray=ray)
        if not raycast_results.has_hits():
            return None

        hit_info = raycast_results.hits[0]

        return hit_info


@hydra.main(
    version_base=None, config_path="config", config_name="pick_throw_vr"
)
def main(config):
    hitl_main(
        config,
        lambda app_service: AppStatePickThrowVr(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
