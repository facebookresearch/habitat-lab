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

from habitat_sim.physics import MotionType
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
        self._held_hand_idx = None  # currently only used with remote_gui_input
        self._recent_reach_pos = None

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
            self._get_agent_feet_height(),
        )

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

    def get_sim(self):
        return self._sandbox_service.sim

    def get_grasp_keys_by_hand(self, hand_idx):
        if hand_idx == 0:
            return [GuiInput.KeyNS.ZERO, GuiInput.KeyNS.ONE]
        else:
            assert hand_idx == 1
            return [GuiInput.KeyNS.TWO, GuiInput.KeyNS.THREE]

    def _try_grasp_remote(self):

        reach_pos = None

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

        grasp_threshold = 0.25
        grasped_objects_idxs = get_grasped_objects_idxs(self.get_sim())

        found_obj_idx = None
        found_hand_idx = None
        found_obj_pos = None
        for i in range(len(self._target_obj_ids)):
            # object is grasped
            if i in grasped_objects_idxs:
                continue

            this_target_pos = self._get_target_object_position(i)

            for hand_idx in range(num_hands):
                hand_pos = hand_positions[hand_idx]
                if (this_target_pos - hand_pos).length() < grasp_threshold:
                    found_obj_idx = i
                    found_hand_idx = hand_idx
                    found_obj_pos = this_target_pos
                    break

            if found_obj_idx is not None:
                color = mn.Color3(0, 1, 0)  # green
                box_half_size = 0.20
                self._draw_box_in_pos(
                    this_target_pos, color=color, box_half_size=box_half_size
                )

        if found_obj_idx is None:
            return None

        remote_button_input = remote_gui_input.get_gui_input()
        
        do_grasp = False
        do_try_reach = False
        for key in self.get_grasp_keys_by_hand(found_hand_idx):
            if remote_button_input.get_key(key):
                do_try_reach = True
                break
            if remote_button_input.get_key_up(key):
                do_grasp = True
                break

        if do_try_reach:
            agent_pos = self._get_agent_translation()
            reach_dist_threshold = 1.0
            dist_xz = (mn.Vector2(agent_pos.x, agent_pos.z) - mn.Vector2(found_obj_pos.x, found_obj_pos.z)).length()
            if dist_xz < reach_dist_threshold:
                reach_pos = found_obj_pos

        if do_grasp:
            self._held_target_obj_idx = found_obj_idx
            self._held_hand_idx = found_hand_idx
            self.state_machine_agent_ctrl.cancel_fetch()

        return reach_pos

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
            pos1, _ = remote_gui_input.get_hand_pose(hand_idx, history_index=history_offset)
            pos0, _ = remote_gui_input.get_hand_pose(hand_idx, history_index=history_len - 1)
            if pos0 and pos1:
                vel = (pos1 - pos0) / (remote_gui_input.get_history_timestep() * (history_len - history_offset))
                rom_obj.linear_velocity = vel
            else:
                rom_obj.linear_velocity = mn.Vector3(0, 0, 0)


            self._fetch_object_remote(self._held_target_obj_idx)
            self._held_target_obj_idx = None
            self._held_hand_idx = None
        else:
            # snap to hand
            hand_pos, _ = remote_gui_input.get_hand_pose(self._held_hand_idx)
            assert hand_pos is not None

            rom_obj.translation = hand_pos
            rom_obj.linear_velocity = mn.Vector3(0, 0, 0)
            

    def _fetch_object_remote(self, obj_idx):

        object_id = self._target_obj_ids[obj_idx]
        throw_obj_id = object_id

        self.state_machine_agent_ctrl.object_interest_id = (
            throw_obj_id
        )
        sim = self.get_sim()
        self.state_machine_agent_ctrl.rigid_obj_interest = (
            sim.get_rigid_object_manager().get_object_by_id(
                throw_obj_id
            )
        )

        self.state_machine_agent_ctrl.current_state = FetchState.SEARCH

    def _update_grasping_and_set_act_hints_remote(self):

        reach_pos = None
        if self._held_target_obj_idx is None:
            reach_pos = self._try_grasp_remote()
        else:
            self._update_held_and_try_throw_remote()

        walk_dir = None
        distance_multiplier = 1.0
        (
            candidate_walk_dir,
            candidate_distance_multiplier,
        ) = self._nav_helper.get_humanoid_walk_hints_from_remote_gui_input(
            visualize_path=False
        )
        walk_dir = candidate_walk_dir
        distance_multiplier = candidate_distance_multiplier

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
            reach_pos=reach_pos
        )        

    def _update_grasping_and_set_act_hints(self):

        if self._sandbox_service.args.remote_gui_mode:
            self._update_grasping_and_set_act_hints_remote()
        else:
            self._update_grasping_and_set_act_hints_local()

    def _update_grasping_and_set_act_hints_local(self):
        drop_pos = None
        grasp_object_id = None
        throw_vel = None
        obj_pick = None
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
                    obj_id = self._target_obj_ids[self._held_target_obj_idx]
                    reach_pos = self.get_sim().get_rigid_object_manager().get_object_by_id(obj_id).translation

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

                    assert throw_obj_id is not None
                    self.state_machine_agent_ctrl.object_interest_id = (
                        throw_obj_id
                    )
                    sim = self.get_sim()
                    self.state_machine_agent_ctrl.rigid_obj_interest = (
                        sim.get_rigid_object_manager().get_object_by_id(
                            throw_obj_id
                        )
                    )
                    throw_vel = self._throw_helper.viz_and_get_humanoid_throw()
                    self.state_machine_agent_ctrl.current_state = FetchState.SEARCH

                    self._held_target_obj_idx = None

        else:
            # check for new grasp and call gui_agent_ctrl.set_act_hints
            if self._held_target_obj_idx is None:

                obj_pick = self._pick_helper.viz_and_get_pick_object()

                assert not self._gui_agent_ctrl.is_grasped
                # pick up an object
                if self._sandbox_service.gui_input.get_key_down(
                    GuiInput.KeyNS.SPACE
                ):
                    translation = self._get_agent_translation()

                    if obj_pick is not None:
                        # Hack: we will use obj0 as our object of interest
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
                            self.state_machine_agent_ctrl.cancel_fetch()
                            grasp_object_id = self._target_obj_ids[
                                self._held_target_obj_idx
                            ]
                            # we will reach towards this position until spacebar is released
                            self._recent_reach_pos = this_target_pos
                            reach_pos = self._recent_reach_pos

        if self.state_machine_agent_ctrl.current_state != FetchState.WAIT:
            obj_pos = (
                self.state_machine_agent_ctrl.rigid_obj_interest.translation
            )
            self._draw_box_in_pos(obj_pos, color=mn.Color3.blue())

        walk_dir = None
        distance_multiplier = 1.0
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

        assert isinstance(self._gui_agent_ctrl, GuiHumanoidController)
        self._gui_agent_ctrl.set_act_hints(
            walk_dir,
            distance_multiplier,
            grasp_object_id,
            drop_pos,
            self._camera_helper.lookat_offset_yaw,
            throw_vel=throw_vel,
            reach_pos=reach_pos
        )

        return drop_pos

    def _get_target_rigid_object(self, target_obj_idx):
        sim = self.get_sim()
        rom = sim.get_rigid_object_manager()
        object_id = self._target_obj_ids[target_obj_idx]
        return rom.get_object_by_id(object_id)

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
            self._draw_box_in_pos(
                this_target_pos, color=color, box_half_size=box_half_size
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

    def _get_agent_pose(self):
        agent_root = get_agent_art_obj_transform(
            self.get_sim(), self.get_gui_controlled_agent_index()
        )
        return agent_root.translation, agent_root.rotation

    def _get_camera_lookat_pos(self):
        agent_pos, _ = self._get_agent_pose()
        lookat_y_offset = mn.Vector3(0, 1, 0)
        lookat = agent_pos + lookat_y_offset
        return lookat

    def sim_update(self, dt, post_sim_update_dict):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._sandbox_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.M):
            self._sandbox_service.end_episode(do_reset=True)

        self._viz_objects()
        self._update_grasping_and_set_act_hints()
        self._sandbox_service.compute_action_and_step_env()

        self._camera_helper.update(self._get_camera_lookat_pos(), dt)

        self._cam_transform = self._camera_helper.get_cam_transform()
        post_sim_update_dict["cam_transform"] = self._cam_transform

        self._update_help_text()

    def is_app_state_done(self):
        # terminal neverending app state
        return False
