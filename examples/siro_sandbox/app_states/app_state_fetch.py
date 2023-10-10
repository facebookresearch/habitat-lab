#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np
from app_states.app_state_abc import AppState
from camera_helper import CameraHelper
from controllers.fetch_baselines_controller import (
    FetchBaselinesController,
    FetchState,
)
from controllers.gui_controller import GuiHumanoidController
from gui_avatar_switch_helper import GuiAvatarSwitchHelper
from gui_navigation_helper import GuiNavigationHelper
from gui_pick_helper import GuiPickHelper
from gui_throw_helper import GuiThrowHelper
from hablab_utils import get_agent_art_obj_transform, get_grasped_objects_idxs

from habitat.gui.gui_input import GuiInput
from habitat.gui.text_drawer import TextOnScreenAlignment
from habitat_sim.physics import MotionType

COLOR_GRASPABLE = mn.Color3(1, 0.75, 0)
COLOR_GRASP_PREVIEW = mn.Color3(0.5, 1, 0)
COLOR_FOCUS_OBJECT = mn.Color3(1, 1, 1)
COLOR_FETCHER_NAV_PATH = mn.Color3(0, 153 / 255, 255 / 255)

RADIUS_GRASPABLE = 0.2
RADIUS_GRASP_PREVIEW = 0.35
RADIUS_FOCUS_OBJECT = 0.15
RADIUS_FETCHER_NAV_PATH = 0.45

RING_PULSE_SIZE = 0.05

disable_spot = False


class AppStateFetch(AppState):
    def __init__(self, sandbox_service, gui_agent_ctrl, robot_agent_ctrl):
        self._sandbox_service = sandbox_service
        self._gui_agent_ctrl = gui_agent_ctrl
        self.state_machine_agent_ctrl = robot_agent_ctrl
        self._can_grasp_place_threshold = (
            self._sandbox_service.args.can_grasp_place_threshold
        )

        self._cam_transform = None

        self._current_episode = None
        self._held_target_obj_idx = None
        self._held_hand_idx = None  # currently only used with remote_gui_input
        self._recent_reach_pos = None
        self._paused = False
        self._hide_gui_text = False

        # will be set in on_environment_reset
        self._target_obj_ids = None

        self._camera_helper = CameraHelper(
            self._sandbox_service.args, self._sandbox_service.gui_input
        )
        # not supported for fetch
        assert not self._sandbox_service.args.first_person_mode

        self._nav_helper = GuiNavigationHelper(
            self._sandbox_service, self.get_gui_controlled_agent_index()
        )
        self._throw_helper = GuiThrowHelper(
            self._sandbox_service, self.get_gui_controlled_agent_index()
        )
        self._pick_helper = GuiPickHelper(
            self._sandbox_service,
            self.get_gui_controlled_agent_index(),
            self._get_gui_agent_feet_height(),
        )

        self._avatar_switch_helper = GuiAvatarSwitchHelper(
            self._sandbox_service, self._gui_agent_ctrl
        )

        # self._gui_agent_ctrl.line_renderer = sandbox_service.line_render

        # choose initial value for toggle
        self._is_remote_active_toggle = (
            self._sandbox_service.args.remote_gui_mode
        )

    def _is_remote_active(self):
        return self._is_remote_active_toggle

    def on_environment_reset(self, episode_recorder_dict):
        self._current_episode = self._sandbox_service.env.current_episode
        self._held_target_obj_idx = None

        sim = self.get_sim()
        # temp_ids, _ = sim.get_targets()
        # self._target_obj_ids = [
        #     sim._scene_obj_ids[temp_id] for temp_id in temp_ids
        # ]
        self._target_obj_ids = sim._scene_obj_ids

        self._nav_helper.on_environment_reset()
        self._pick_helper.on_environment_reset(
            agent_feet_height=self._get_gui_agent_feet_height()
        )

        self._camera_helper.update(self._get_camera_lookat_pos(), dt=0)

    def get_sim(self):
        return self._sandbox_service.sim

    def get_grasp_keys_by_hand(self, hand_idx):
        if hand_idx == 0:
            return [GuiInput.KeyNS.ZERO, GuiInput.KeyNS.ONE]
        else:
            assert hand_idx == 1
            return [GuiInput.KeyNS.TWO, GuiInput.KeyNS.THREE]

    def _try_grasp_remote(self):
        assert not self._held_target_obj_idx

        # todo: rename remote_gui_input
        remote_gui_input = self._sandbox_service.remote_gui_input

        hand_positions = []
        num_hands = 2
        for i in range(num_hands):
            hand_pos, _ = remote_gui_input.get_hand_pose(i)
            if hand_pos:
                hand_positions.append(hand_pos)
        if len(hand_positions) == 0:
            return
        assert len(hand_positions) == num_hands

        grasp_threshold = 0.2
        grasped_objects_idxs = get_grasped_objects_idxs(self.get_sim())

        remote_button_input = remote_gui_input.get_gui_input()

        found_obj_idx = None
        found_hand_idx = None

        for i in range(len(self._target_obj_ids)):
            # object is already grasped by Spot
            if i in grasped_objects_idxs:
                continue

            this_target_pos = self._get_target_object_position(i)

            for hand_idx in range(num_hands):
                hand_pos = hand_positions[hand_idx]
                if (this_target_pos - hand_pos).length() < grasp_threshold:
                    # color = mn.Color3(0, 1, 0)  # green
                    # box_half_size = 0.20
                    # self._draw_box_in_pos(
                    #     this_target_pos, color=color, box_half_size=box_half_size
                    # )
                    # show grasp
                    self._add_target_object_highlight_ring(
                        i, COLOR_GRASP_PREVIEW, radius=RADIUS_GRASP_PREVIEW
                    )

                    for key in self.get_grasp_keys_by_hand(hand_idx):
                        if remote_button_input.get_key_down(key):
                            found_obj_idx = i
                            found_hand_idx = hand_idx
                            break

            if found_obj_idx is not None:
                break

        if found_obj_idx is None:
            return

        self._held_target_obj_idx = found_obj_idx
        self._held_hand_idx = found_hand_idx
        self.state_machine_agent_ctrl.cancel_fetch()
        rom_obj = self._get_target_rigid_object(self._held_target_obj_idx)
        rom_obj.motion_type = MotionType.KINEMATIC

    def _update_held_and_try_throw_remote(self):
        assert self._held_target_obj_idx is not None
        assert self._held_hand_idx is not None

        remote_gui_input = self._sandbox_service.remote_gui_input
        remote_button_input = remote_gui_input.get_gui_input()

        do_throw = False
        for key in self.get_grasp_keys_by_hand(self._held_hand_idx):
            if remote_button_input.get_key_up(key):
                do_throw = True

        rom_obj = self._get_target_rigid_object(self._held_target_obj_idx)

        if do_throw:
            rom_obj.motion_type = MotionType.DYNAMIC
            rom_obj.collidable = True

            hand_idx = self._held_hand_idx
            history_len = remote_gui_input.get_history_length()
            assert history_len >= 2
            history_offset = 1
            pos1, _ = remote_gui_input.get_hand_pose(
                hand_idx, history_index=history_offset
            )
            pos0, _ = remote_gui_input.get_hand_pose(
                hand_idx, history_index=history_len - 1
            )
            if pos0 and pos1:
                vel = (pos1 - pos0) / (
                    remote_gui_input.get_history_timestep()
                    * (history_len - history_offset)
                )
                rom_obj.linear_velocity = vel
            else:
                rom_obj.linear_velocity = mn.Vector3(0, 0, 0)

            self._fetch_object_remote(self._held_target_obj_idx)
            self._held_target_obj_idx = None
            self._held_hand_idx = None
        else:
            # snap to hand
            hand_pos, hand_rotation = remote_gui_input.get_hand_pose(
                self._held_hand_idx
            )
            assert hand_pos is not None

            rom_obj.translation = hand_pos
            rom_obj.rotation = hand_rotation
            rom_obj.linear_velocity = mn.Vector3(0, 0, 0)

    def _fetch_object_remote(self, obj_idx):
        object_id = self._target_obj_ids[obj_idx]
        throw_obj_id = object_id

        self.state_machine_agent_ctrl.object_interest_id = throw_obj_id
        sim = self.get_sim()
        self.state_machine_agent_ctrl.rigid_obj_interest = (
            sim.get_rigid_object_manager().get_object_by_id(throw_obj_id)
        )

        if not disable_spot:
            self.state_machine_agent_ctrl.current_state = FetchState.SEARCH

    def _update_grasping_and_set_act_hints_remote(self):
        if self._held_target_obj_idx is None:
            self._try_grasp_remote()
        else:
            self._update_held_and_try_throw_remote()

        (
            walk_dir,
            distance_multiplier,
        ) = self._nav_helper.get_humanoid_walk_hints_from_remote_gui_input(
            visualize_path=False
        )

        # If the humanoid is near enough to the VR user that it doesn't need to walk
        # (distance_multiplier == 0.0), then try to reach to held object.
        reach_pos = None
        if (
            self._held_target_obj_idx is not None
            and distance_multiplier == 0.0
        ):
            # vr_root_pos, _ = self._sandbox_service.remote_gui_input.get_head_pose()
            # humanoid_pos = self._get_gui_agent_translation()
            # dist_threshold = 0.25
            # if (vr_root_pos - humanoid_pos).length() < dist_threshold:
            # reach_pos = self._get_target_object_position(self._held_target_obj_idx)
            # distance_multiplier = 0.0  # disable walking, but allow rotation via walk_dir
            reach_pos = self._get_target_object_position(
                self._held_target_obj_idx
            )

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

        if self._held_target_obj_idx is not None:
            if self._sandbox_service.gui_input.get_key(GuiInput.KeyNS.SPACE):
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

            if self._sandbox_service.gui_input.get_key_up(
                GuiInput.KeyNS.SPACE
            ):
                if self._recent_reach_pos:
                    # this spacebar release means we've finished the reach-and-grasp motion
                    self._recent_reach_pos = None
                else:
                    # this spacebar release means we've finished the throwing motion
                    throw_obj_id = (
                        self._gui_agent_ctrl._get_grasp_mgr().snap_idx
                    )

                    # The robot can only pick up the object if there is a throw_obj_id
                    # throw_obj_id will be None if the users fast press and press again the sapce
                    if throw_obj_id is not None:
                        self.state_machine_agent_ctrl.object_interest_id = (
                            throw_obj_id
                        )
                        sim = self.get_sim()
                        self.state_machine_agent_ctrl.rigid_obj_interest = (
                            sim.get_rigid_object_manager().get_object_by_id(
                                throw_obj_id
                            )
                        )
                        throw_vel = (
                            self._throw_helper.viz_and_get_humanoid_throw()
                        )
                        if not disable_spot:
                            self.state_machine_agent_ctrl.current_state = (
                                FetchState.SEARCH
                            )

                        self._held_target_obj_idx = None
        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:
                assert not self._gui_agent_ctrl.is_grasped
                translation = self._get_gui_agent_translation()

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

                    if self._sandbox_service.gui_input.get_key_down(
                        GuiInput.KeyNS.SPACE
                    ):
                        self._recent_reach_pos = (
                            self._get_target_object_position(min_i)
                        )
                        # we will reach towards this position until spacebar is released
                        reach_pos = self._recent_reach_pos

                        self._held_target_obj_idx = min_i
                        self.state_machine_agent_ctrl.cancel_fetch()
                        grasp_object_id = self._target_obj_ids[
                            self._held_target_obj_idx
                        ]

        # Switch to the point nav with waypoints by keeping pressing "O" key
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.O) and (
            self.state_machine_agent_ctrl.current_state == FetchState.SEARCH
            or self.state_machine_agent_ctrl.current_state
            == FetchState.SEARCH_TIMEOUT_WAIT
            or self.state_machine_agent_ctrl.current_state == FetchState.BRING
            or self.state_machine_agent_ctrl.current_state
            == FetchState.BRING_TIMEOUT_WAIT
        ):
            if (
                self.state_machine_agent_ctrl.current_state
                == FetchState.SEARCH
                or self.state_machine_agent_ctrl.current_state
                == FetchState.SEARCH_TIMEOUT_WAIT
            ):
                self.state_machine_agent_ctrl.current_state = (
                    FetchState.SEARCH_ORACLE_NAV
                )
            else:
                self.state_machine_agent_ctrl.current_state = (
                    FetchState.BRING_ORACLE_NAV
                )

        if self._sandbox_service.gui_input.get_key_up(GuiInput.KeyNS.O) and (
            self.state_machine_agent_ctrl.current_state
            == FetchState.SEARCH_ORACLE_NAV
            or self.state_machine_agent_ctrl.current_state
            == FetchState.BRING_ORACLE_NAV
        ):
            if (
                self.state_machine_agent_ctrl.current_state
                == FetchState.SEARCH_ORACLE_NAV
            ):
                self.state_machine_agent_ctrl.current_state = FetchState.SEARCH
            else:
                self.state_machine_agent_ctrl.current_state = FetchState.BRING

        # Do vis on the real navigation target
        if self.state_machine_agent_ctrl.current_state != FetchState.WAIT:
            safe_pos = self.state_machine_agent_ctrl.safe_pos
            if safe_pos is not None:
                self._draw_circle(safe_pos, color=mn.Color3.red(), radius=0.25)

        walk_dir = None
        distance_multiplier = 1.0

        if self._sandbox_service.gui_input.get_mouse_button(
            GuiInput.MouseNS.RIGHT
        ):
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

    def _reposition_object_height(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        pos = rom.get_object_by_id(object_id).translation
        height = pos[1]
        # TODO: better way to handle the object's height when the robot drops the object
        if height < 0.0:
            rom.get_object_by_id(object_id).translation = mn.Vector3(
                [pos[0], 0.0, pos[2]]
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

    def _draw_box_in_pos(self, position, color, box_half_size=0.15):
        box_offset = mn.Vector3(box_half_size, box_half_size, box_half_size)
        self._sandbox_service.line_render.draw_box(
            position - box_offset,
            position + box_offset,
            color,
        )

    def _draw_circle(self, pos, color, radius):
        num_segments = 24
        self._sandbox_service.line_render.draw_circle(
            pos,
            radius,
            color,
            num_segments,
        )

    def _add_target_object_highlight_ring(
        self, target_obj_idx, color, radius, do_pulse=False
    ):
        pos = self._get_target_object_position(target_obj_idx)

        if do_pulse:
            radius += (
                self._sandbox_service.get_anim_fraction() * RING_PULSE_SIZE
            )
            # color = mn.Color4(color.r, color.g, color.b, 1.0 - self._sandbox_service.get_anim_fraction())

        num_segments = 24

        self._sandbox_service.line_render.draw_circle(
            pos,
            radius,
            color,
            num_segments,
        )

    def _viz_objects(self):
        # grasped_objects_idxs = get_grasped_objects_idxs(self.get_sim())

        focus_obj_idx = None
        if self._held_target_obj_idx is not None:
            focus_obj_idx = self._held_target_obj_idx
        elif self.state_machine_agent_ctrl.object_interest_id is not None:
            tmp_id = self.state_machine_agent_ctrl.object_interest_id
            # find focus_obj_idx
            assert tmp_id in self._target_obj_ids
            focus_obj_idx = self._target_obj_ids.index(tmp_id)

        if focus_obj_idx is None:
            for i in range(len(self._target_obj_ids)):
                self._add_target_object_highlight_ring(
                    i, COLOR_GRASPABLE, radius=RADIUS_GRASPABLE, do_pulse=True
                )
        else:
            self._add_target_object_highlight_ring(
                focus_obj_idx, COLOR_FOCUS_OBJECT, radius=RADIUS_FOCUS_OBJECT
            )
            # Prevent the object from sinking to the floor
            self._reposition_object_height(focus_obj_idx)

            # # draw can grasp area
            # can_grasp_position = mn.Vector3(this_target_pos)
            # can_grasp_position[1] = self._get_gui_agent_feet_height()
            # self._sandbox_service.line_render.draw_circle(
            #     can_grasp_position,
            #     self._can_grasp_place_threshold,
            #     mn.Color3(255 / 255, 255 / 255, 0),
            #     24,
            # )

    def get_gui_controlled_agent_index(self):
        return self._gui_agent_ctrl._agent_idx

    def _get_gui_agent_translation(self):
        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        return (
            self._gui_agent_ctrl._humanoid_controller.obj_transform_base.translation
        )

    def _get_state_machine_agent_translation(self):
        assert isinstance(
            self.state_machine_agent_ctrl, FetchBaselinesController
        )
        transform = (
            self.state_machine_agent_ctrl.get_articulated_agent().base_transformation
        )
        return transform.translation

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
            if self._held_target_obj_idx is not None:
                return "Spacebar: throw\n"
            else:
                return "Spacebar: pick up\n"

        controls_str: str = ""
        if not self._hide_gui_text:
            controls_str += "ESC: exit\n"
            controls_str += "X: reset episode\n"
            controls_str += "M: change scene\n"
            controls_str += "R + drag: rotate camera\n"
            controls_str += "Scroll: zoom\n"
            if self._sandbox_service.args.remote_gui_mode:
                controls_str += "T: toggle keyboard/VR\n"
            controls_str += "P: pause\n"
            controls_str += "H: show.hide controls text\n"
            if not self._is_remote_active_toggle:
                controls_str += "\nHumanoid controls\n"
                controls_str += "Right-click: walk\n"
                controls_str += "A, D: turn\n"
                controls_str += "W, S: walk\n"
                controls_str += get_grasp_release_controls_text()

        return controls_str

    def _get_status_text(self):
        status_str = ""

        # if self._held_target_obj_idx is not None:
        #     status_str += "Throw the object!"
        # else:
        #     status_str += "Grab an object!\n"

        if self._sandbox_service.args.remote_gui_mode:
            status_str += (
                "human control: VR\n"
                if self._is_remote_active()
                else "human control: keyboard\n"
            )

        fetch_state = self.state_machine_agent_ctrl.current_state
        fetch_state_names = {
            # FetchState.WAIT : "",
            FetchState.SEARCH: "searching for object",
            FetchState.SEARCH_ORACLE_NAV: "searching for object",
            FetchState.PICK: "picking object",
            FetchState.BRING: "searching for human",
            FetchState.BRING_ORACLE_NAV: "searching for human",
            FetchState.DROP: "dropping object",
            FetchState.BEG_RESET: "cannot pick up the object",
            FetchState.SEARCH_TIMEOUT_WAIT: "need help to search for object",
            FetchState.BRING_TIMEOUT_WAIT: "need help to search for human",
        }
        if fetch_state in fetch_state_names:
            status_str += f"spot: {fetch_state_names[fetch_state]}\n"

        if self._paused:
            status_str += "\npaused\n"

        # center align the status_str
        max_status_str_len = 50
        status_str = "\n".join(
            line.center(max_status_str_len) for line in status_str.split("\n")
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
                text_delta_x=-480,  # hack approximate centering based on our font size
                text_delta_y=-50,
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

    def _viz_fetcher(self, post_sim_update_dict):
        fetch_state = self.state_machine_agent_ctrl.current_state

        if (
            fetch_state == FetchState.SEARCH
            or fetch_state == FetchState.BRING
            or fetch_state == FetchState.SEARCH_ORACLE_NAV
            or fetch_state == FetchState.BRING_ORACLE_NAV
        ):
            fetcher_pos = self._get_state_machine_agent_translation()
            target_pos = (
                self.state_machine_agent_ctrl.rigid_obj_interest.translation
                if fetch_state == FetchState.SEARCH
                or fetch_state == FetchState.SEARCH_ORACLE_NAV
                else self._get_gui_agent_translation()
            )
            if self.state_machine_agent_ctrl.gt_path is not None and (
                fetch_state == FetchState.SEARCH_ORACLE_NAV
                or fetch_state == FetchState.BRING_ORACLE_NAV
            ):
                gt_path = [
                    mn.Vector3(pt)
                    for pt in self.state_machine_agent_ctrl.gt_path
                ]
                path_points = [fetcher_pos] + gt_path + [target_pos]
            else:
                path_points = [fetcher_pos, target_pos]
            floor_y = 0.0  # temp hack
            for pt in path_points:
                pt.y = floor_y
            path_endpoint_radius = RADIUS_FETCHER_NAV_PATH
            path_color = COLOR_FETCHER_NAV_PATH

            self._sandbox_service.line_render.draw_path_with_endpoint_circles(
                path_points, path_endpoint_radius, path_color
            )

        if not disable_spot:
            # sloppy: assume agent 0 and assume agent_0_articulated_agent_arm_depth obs key
            assert self.state_machine_agent_ctrl._agent_idx == 0
            post_sim_update_dict["debug_images"].append(
                (
                    "spot depth sensor",
                    self._sandbox_service.get_observation_as_debug_image(
                        "agent_0_articulated_agent_arm_depth"
                    ),
                )
            )

    def _reset_current_episode(self):
        # reset current episode
        # env's current_episode setter will set internal attribute
        # _episode_from_iter_on_reset to False as a result reset
        # will be called without the episode changing
        self._sandbox_service.env.current_episode = self._current_episode
        self._sandbox_service.end_episode(do_reset=True)

    def _set_next_episode(self):
        self._sandbox_service.end_episode(do_reset=True)

    def _end_episode(self):
        self._sandbox_service.end_episode(do_reset=False)

    def sim_update(self, dt, post_sim_update_dict):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._end_episode()
            post_sim_update_dict["application_exit"] = True

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.M):
            self._set_next_episode()

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.X):
            self._reset_current_episode()

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.P):
            self._paused = not self._paused

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.H):
            self._hide_gui_text = not self._hide_gui_text

        # toggle remote/local on keypress, if not holding anything
        if (
            self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.T)
            and self._sandbox_service.args.remote_gui_mode
            and self._held_target_obj_idx is None
        ):
            self._is_remote_active_toggle = not self._is_remote_active_toggle

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.TAB):
            self._avatar_switch_helper.switch_avatar()

        self._viz_fetcher(post_sim_update_dict)
        self._viz_objects()
        if not self._paused:
            self._update_grasping_and_set_act_hints()
            self._sandbox_service.compute_action_and_step_env()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()

    def is_app_state_done(self):
        # terminal neverending app state
        return False
