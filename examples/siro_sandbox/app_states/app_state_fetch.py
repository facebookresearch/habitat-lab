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
from utils.hablab_utils import get_agent_art_obj_transform, get_grasped_objects_idxs

import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from utils.gui.gui_input import GuiInput
from utils.gui.text_drawer import TextOnScreenAlignment
from habitat.tasks.rearrange.utils import get_angle_to_pos
from habitat_sim.physics import MotionType

COLOR_GRASPABLE = mn.Color3(1, 0.75, 0)
COLOR_GRASP_PREVIEW = mn.Color3(0.5, 1, 0)
COLOR_FOCUS_OBJECT = mn.Color3(1, 1, 1)
COLOR_FETCHER_NAV_PATH = mn.Color3(0, 160 / 255, 171 / 255)
COLOR_FETCHER_ORACLE_NAV_PATH = mn.Color3(0, 153 / 255, 255 / 255)
COLOR_PLACE_GOAL = mn.Color3(255 / 255, 0, 220 / 255)
COLOR_PLACE_GOAL_INVALID = mn.Color3(255 / 255, 0, 0)

RADIUS_GRASPABLE = 0.15
RADIUS_GRASP_PREVIEW = 0.15
RADIUS_FOCUS_OBJECT = 0.2
RADIUS_FETCHER_NAV_PATH = 0.45
RADIUS_PLACE_GOAL = 0.25
RADIUS_PLACE_GOAL_PREVIEW = 0.15
RADIUS_PLACE_GOAL_INVALID = 0.03

RING_PULSE_SIZE = 0.03


MIN_STEPS_STOP = 15
disable_spot = False

# The distance constrain for the robot being placed near human
ROBOT_SPAWN_NEAR_HUMAN_DIS = 5.0
# Randomly reset the location of the robot or human
RANDOM_AGENT_LOCATION_RESET = False
# Sloppy: hardcode location of the robot and human
# Note that you do not have to provide the orientation information
# since both agents will automatically look at each other
FIX_AGENT_INIT_POS = {
    "data/scene_datasets/hssd-hab/scenes-uncluttered/102344193.scene_instance.json": (
        # Scene 1: human stands near the bed and the robot blocked the room door.
        # So the human needs to walk out of room and will observe how Spot avoid collision.
        # This showcases the social nav policy
        [0.773143, -4.09137],
        [1.53322, -1.53417],
    ),
    "data/scene_datasets/hssd-hab/scenes-uncluttered/102344280.scene_instance.json": (
        # Scene 2: human stands in the kitchen + living room area. Just showcase the scene
        [-1.4465, 1.36246],
        [-7.46346, 1.5259],
    ),
    "data/scene_datasets/hssd-hab/scenes-uncluttered/102817200.scene_instance.json": (
        # Scene 3: human stands in the living room with lots of furniture (sofa, clutter, etc),
        # and it is easier to trigger Spot begging motion.
        # This showcases the pick policy/begging motions
        [-3.63359, -8.33051],
        [-1.03828, -7.71454],
    ),
    "data/scene_datasets/hssd-hab/scenes-uncluttered/103997424_171030444.scene_instance.json": (
        # Scene 4: human stands near the shelf in which he/she can put the object on
        # multiple surfaces of the shelf, and at the beginning, human is able to
        # see robot come to find him/herself. This showcases the pick policy
        [-14.3757, -10.0978],
        [-18.5925, -12.9898],
    ),
    "data/scene_datasets/hssd-hab/scenes-uncluttered/103997541_171030615.scene_instance.json": (
        # Scene 5: human passes a long hallway to reach the target and get
        # the object. This showcases the social nav policy
        [-10.84, 5.24],
        [-10.86, 8.18],
    ),
}


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
        self.count_tsteps_stop = 0
        self._has_grasp_preview = False
        self._remote_place_goal_counter = 0

        # sloppy: set this private member so we get deterministic
        # agent starting positions
        self._sandbox_service.env.task._fixed_starting_position = True

    def _is_remote_active(self):
        return self._is_remote_active_toggle

    def _make_agent_face_target(self, target_pos, idx):
        agent_pos = (
            self.get_sim().get_agent_data(idx).articulated_agent.base_pos
        )
        relative_target = target_pos - agent_pos
        angle_to_object = get_angle_to_pos(relative_target)
        self.get_sim().get_agent_data(
            idx
        ).articulated_agent.base_rot = angle_to_object

    def _make_robot_near_human(self, human_pos):
        """Constrain the robot location to be near human pos"""
        robot_pos = self.get_sim().pathfinder.get_random_navigable_point_near(
            human_pos,
            ROBOT_SPAWN_NEAR_HUMAN_DIS,
            island_index=self.get_sim()._largest_island_idx,
        )

        # If the robot_pos is not NaN
        if not np.isnan(robot_pos[0]):
            # Sloppy: overwrite the location of the robot
            # Make sure the height is consistent
            robot_pos[1] = (
                self.get_sim()
                .get_agent_data(1 - self.get_gui_controlled_agent_index())
                .articulated_agent.base_pos[1]
            )
            self.get_sim().get_agent_data(
                1 - self.get_gui_controlled_agent_index()
            ).articulated_agent.base_pos = robot_pos
            # Make the robot face toward the human
            self._make_agent_face_target(
                human_pos, 1 - self.get_gui_controlled_agent_index()
            )

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
        self.count_tsteps_stop = 0

        # Get the scene id
        scene_id = (
            self._sandbox_service.episode_helper.current_episode.scene_id
        )

        # Reset the location of the robot so that it is near human
        # Sloppy: should move this flag to other places
        if RANDOM_AGENT_LOCATION_RESET:
            human_pos = (
                self.get_sim()
                .get_agent_data(self.get_gui_controlled_agent_index())
                .articulated_agent.base_pos
            )
            self._make_robot_near_human(human_pos)
            robot_pos = (
                self.get_sim()
                .get_agent_data(1 - self.get_gui_controlled_agent_index())
                .articulated_agent.base_pos
            )
        elif scene_id is not None:
            # Robot
            fix_robot_pos = FIX_AGENT_INIT_POS[scene_id][0]
            robot_height = (
                self.get_sim()
                .get_agent_data(1 - self.get_gui_controlled_agent_index())
                .articulated_agent.base_pos[1]
            )
            robot_pos = mn.Vector3(
                [fix_robot_pos[0], robot_height, fix_robot_pos[1]]
            )
            self.get_sim().get_agent_data(
                1 - self.get_gui_controlled_agent_index()
            ).articulated_agent.base_pos = robot_pos
            # Human
            fix_human_pos = FIX_AGENT_INIT_POS[scene_id][1]
            human_height = (
                self.get_sim()
                .get_agent_data(self.get_gui_controlled_agent_index())
                .articulated_agent.base_pos[1]
            )
            human_pos = mn.Vector3(
                [fix_human_pos[0], human_height, fix_human_pos[1]]
            )
            self.get_sim().get_agent_data(
                self.get_gui_controlled_agent_index()
            ).articulated_agent.base_pos = human_pos
        # Make agents look at each other
        self._make_agent_face_target(
            human_pos, self.state_machine_agent_ctrl._agent_idx
        )
        self._make_agent_face_target(
            robot_pos, 1 - self.state_machine_agent_ctrl._agent_idx
        )

        client_message_manager = self._sandbox_service.client_message_manager
        if client_message_manager:
            client_message_manager.change_humanoid_position(human_pos)
            client_message_manager.signal_scene_change()
            client_message_manager.update_navmesh_triangles(
                self._get_navmesh_triangle_vertices()
            )

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
        self._recent_reach_pos = None
        self._recent_hand_idx = None
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
        self._recent_reach_pos = None

        self._has_grasp_preview = False
        for hand_idx in range(num_hands):
            hand_pos = hand_positions[hand_idx]

            min_dist = grasp_threshold
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
            self._recent_reach_pos = None
            self._recent_hand_idx = None
            self._update_held_and_try_throw_remote()

        (
            walk_dir,
            distance_multiplier,
            forward_dir,
        ) = self._nav_helper.get_humanoid_walk_hints_from_remote_gui_input(
            visualize_path=False
        )

        # Count number of steps since we stopped, this is to reduce jitter
        # with IK
        if distance_multiplier == 0:
            self.count_tsteps_stop += 1
        else:
            self.count_tsteps_stop = 0

        reach_pos = None
        hand_idx = None
        if (
            self._held_target_obj_idx is not None
            and distance_multiplier == 0.0
            and self.count_tsteps_stop > MIN_STEPS_STOP
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
            hand_idx = self._held_hand_idx
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
                    # throw_obj_id will be None if the users fast press and press again the space
                    if throw_obj_id is not None:
                        throw_vel = (
                            self._throw_helper.viz_and_get_humanoid_throw()
                        )
                        if throw_vel:
                            self.state_machine_agent_ctrl.object_interest_id = (
                                throw_obj_id
                            )
                            sim = self.get_sim()
                            self.state_machine_agent_ctrl.rigid_obj_interest = sim.get_rigid_object_manager().get_object_by_id(
                                throw_obj_id
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
                    self._has_grasp_preview = True

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

        # Do vis on the real navigation target
        # if self.state_machine_agent_ctrl.current_state != FetchState.WAIT:
        #     safe_pos = self.state_machine_agent_ctrl.safe_pos
        #     if safe_pos is not None:
        #         self._draw_circle(safe_pos, color=mn.Color3.red(), radius=0.25)

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

    def _get_target_object_bounding_box(self, target_obj_idx) -> mn.Range3D:
        return self._get_target_rigid_object(
            target_obj_idx
        ).collision_shape_aabb

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
        self._add_target_highlight_ring(pos, color, radius, do_pulse)

    def _add_target_highlight_ring(self, pos, color, radius, do_pulse=False):
        if do_pulse:
            radius += (
                self._sandbox_service.get_anim_fraction() * RING_PULSE_SIZE
            )

        if (
            self._sandbox_service.client_message_manager
            and self._is_remote_active()
        ):
            client_radius = radius
            self._sandbox_service.client_message_manager.add_highlight(
                pos, client_radius
            )

        self._draw_circle(pos, color, radius)

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
            if self._sandbox_service.args.remote_gui_mode:
                controls_str += "T: toggle keyboard.VR\n"
            controls_str += "P: pause\n"
            if not self._is_remote_active_toggle:
                controls_str += "Right-click: walk\n"
                controls_str += "WASD: walk\n"
                controls_str += get_grasp_release_controls_text()

        return controls_str

    def _get_spot_status_text(self):
        spot_status_text = ""
        fetch_state = self.state_machine_agent_ctrl.current_state

        if (
            (fetch_state == FetchState.SEARCH or FetchState.SEARCH_ORACLE_NAV)
            and self.state_machine_agent_ctrl.human_block_robot_when_searching
        ):
            spot_status_text = "spot: blocked by human\n"
        else:
            place_goal_desc = (
                "target"
                if self.state_machine_agent_ctrl.user_place_goal_pos
                is not None
                else "human"
            )
            fetch_state_names = {
                # FetchState.WAIT : "",
                FetchState.SEARCH: "searching for object",
                FetchState.SEARCH_ORACLE_NAV: "oracle nav to object",
                FetchState.PICK: "picking object",
                FetchState.BRING: f"navigating to {place_goal_desc}",
                FetchState.BRING_ORACLE_NAV: f"oracle nav to {place_goal_desc}",
                FetchState.DROP: "dropping object",
                FetchState.BEG_RESET: "cannot pick up the object",
                FetchState.SEARCH_TIMEOUT_WAIT: "unable to reach object",
                FetchState.BRING_TIMEOUT_WAIT: f"unable to reach {place_goal_desc}",
            }
            if fetch_state in fetch_state_names:
                spot_status_text = f"spot: {fetch_state_names[fetch_state]}\n"

        return spot_status_text

    def _get_status_text(self):
        status_str = ""

        if self._sandbox_service.args.remote_gui_mode:
            status_str += (
                "human control: VR\n"
                if self._is_remote_active()
                else "human control: keyboard\n"
            )

        status_str += self._get_spot_status_text()

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
        nav_goal_pos = self.state_machine_agent_ctrl.get_nav_goal()
        if nav_goal_pos:
            fetcher_pos = self._get_state_machine_agent_translation()
            if self.state_machine_agent_ctrl.is_oracle_nav():
                # sloppy: we assume _nav_helper uses RADIUS_FETCHER_NAV_PATH and COLOR_FETCHER_ORACLE_NAV_PATH
                self._nav_helper.find_and_viz_path(fetcher_pos, nav_goal_pos)
            else:

                def move_to_ground_plane(pos):
                    ground_y = 0.17
                    return mn.Vector3(pos[0], ground_y, pos[2])

                path_points = [
                    move_to_ground_plane(fetcher_pos),
                    move_to_ground_plane(nav_goal_pos),
                ]
                self._sandbox_service.line_render.draw_path_with_endpoint_circles(
                    path_points,
                    RADIUS_FETCHER_NAV_PATH,
                    COLOR_FETCHER_NAV_PATH,
                )

        if self.state_machine_agent_ctrl.user_place_goal_pos is not None:
            self._add_target_highlight_ring(
                self.state_machine_agent_ctrl.user_place_goal_pos,
                COLOR_PLACE_GOAL,
                RADIUS_PLACE_GOAL,
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
                    2,  # scale
                )
            )

    def _update_place_goal(self):
        # let the user specify a place goal pos, with a few caveats:
        # * don't interfere with grasping
        # * Spot must have an object of interest and can't be already dropping (placing)
        do_allow_place = (
            not self._has_grasp_preview
            and self.state_machine_agent_ctrl.object_interest_id is not None
            and self.state_machine_agent_ctrl.current_state != FetchState.DROP
        )
        if not do_allow_place:
            return

        ray = None
        do_place = False
        if self._is_remote_active():
            hold_threshold = 30  # number of frames before placement starts
            is_either_hand_pressed = False
            remote_gui_input = self._sandbox_service.remote_gui_input
            remote_button_input = remote_gui_input.get_gui_input()

            num_hands = 2
            for hand_idx in range(num_hands):
                is_pressed = False
                is_released = False
                for key in self.get_grasp_keys_by_hand(hand_idx):
                    if remote_button_input.get_key(key):
                        is_pressed = True
                    elif remote_button_input.get_key_up(key):
                        is_released = True
                        break

                if is_pressed:
                    self._remote_place_goal_counter += 1
                    is_either_hand_pressed = True

                do_preview = False
                if self._remote_place_goal_counter >= hold_threshold:
                    if is_pressed:
                        do_preview = True
                    elif is_released:
                        do_place = True

                if do_preview or do_place:
                    hand_pos, hand_rotation = remote_gui_input.get_hand_pose(
                        hand_idx
                    )
                    if hand_pos is None or hand_rotation is None:
                        return

                    ray_dir = hand_rotation.transform_vector(
                        mn.Vector3(0, 0, 1)
                    )
                    # sloppy: start ray a short distance out from controller, because
                    # otherwise the ray is often hitting the humanoid. Todo: have
                    # raycast ignore humanoid.
                    ray_start_offset = 0.5
                    ray_origin = hand_pos + ray_dir * ray_start_offset
                    ray = habitat_sim.geo.Ray(ray_origin, ray_dir)
                    break

            if not is_either_hand_pressed:
                self._remote_place_goal_counter = 0

        else:
            ray = self._sandbox_service.gui_input.mouse_ray
            do_place = self._sandbox_service.gui_input.get_key(
                GuiInput.KeyNS.SPACE
            )

        if not ray:
            return

        hit_info = self.cast_ray(ray)
        if hit_info:
            y_threshold = 0.5
            # don't allow placement on non-horizontal surfaces
            is_valid = hit_info.normal.y >= y_threshold
            goal_pos = hit_info.point
            self._add_target_highlight_ring(
                goal_pos,
                COLOR_PLACE_GOAL if is_valid else COLOR_PLACE_GOAL_INVALID,
                RADIUS_PLACE_GOAL_PREVIEW
                if is_valid
                else RADIUS_PLACE_GOAL_INVALID,
                do_pulse=is_valid,
            )
            if do_place and is_valid:
                self.state_machine_agent_ctrl.user_place_goal_pos = goal_pos

    def _update_fetcher(self):
        fetcher_state = self.state_machine_agent_ctrl.current_state

        # Switch to the point nav with waypoints by keeping pressing "O" key
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.O) and (
            fetcher_state == FetchState.SEARCH
            or fetcher_state == FetchState.SEARCH_TIMEOUT_WAIT
            or fetcher_state == FetchState.BRING
            or fetcher_state == FetchState.BRING_TIMEOUT_WAIT
            or fetcher_state == FetchState.FOLLOW
        ):
            if (
                fetcher_state == FetchState.SEARCH
                or fetcher_state == FetchState.SEARCH_TIMEOUT_WAIT
            ):
                self.state_machine_agent_ctrl.current_state = (
                    FetchState.SEARCH_ORACLE_NAV
                )
            elif (
                fetcher_state == FetchState.BRING
                or fetcher_state == FetchState.BRING_TIMEOUT_WAIT
            ):
                self.state_machine_agent_ctrl.current_state = (
                    FetchState.BRING_ORACLE_NAV
                )
            else:
                self.state_machine_agent_ctrl.current_state = (
                    FetchState.FOLLOW_ORACLE
                )

        if self._sandbox_service.gui_input.get_key_up(GuiInput.KeyNS.O) and (
            fetcher_state == FetchState.SEARCH_ORACLE_NAV
            or fetcher_state == FetchState.BRING_ORACLE_NAV
            or fetcher_state == FetchState.FOLLOW_ORACLE
        ):
            if fetcher_state == FetchState.SEARCH_ORACLE_NAV:
                self.state_machine_agent_ctrl.current_state = FetchState.SEARCH
            elif fetcher_state == FetchState.BRING_ORACLE_NAV:
                self.state_machine_agent_ctrl.current_state = FetchState.BRING
            else:
                self.state_machine_agent_ctrl.current_state = FetchState.FOLLOW

        self._update_place_goal()

    def sim_update(self, dt, post_sim_update_dict):
        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.ESC):
            self._sandbox_service.end_episode()
            post_sim_update_dict["application_exit"] = True

        # use 1-5 keys to select certain episodes corresponding to our 5 scenes
        num_fetch_scenes = 5
        # hand-picked episodes from hitl_vr_sample_episodes.json.gz
        episode_id_by_scene_index = ["0", "5", "10", "15", "20"]
        for scene_idx in range(num_fetch_scenes):
            key = GuiInput.KeyNS(GuiInput.KeyNS.ONE.value + scene_idx)
            if self._sandbox_service.gui_input.get_key_down(key):
                self._sandbox_service.episode_helper.set_next_episode_by_id(
                    episode_id_by_scene_index[scene_idx]
                )
                self._sandbox_service.end_episode(do_reset=True)

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.P):
            self._paused = not self._paused

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.H):
            self._hide_gui_text = not self._hide_gui_text

        # toggle remote/local under certain conditions:
        # - must not be holding anything
        # - toggle on T keypress OR switch to remote if any remote button is pressed
        if (
            self._sandbox_service.args.remote_gui_mode
            and self._held_target_obj_idx is None
            and (
                self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.T)
                or (
                    not self._is_remote_active_toggle
                    and self._sandbox_service.remote_gui_input.get_gui_input().get_any_key_down()
                )
            )
        ):
            self._is_remote_active_toggle = not self._is_remote_active_toggle

        if self._sandbox_service.gui_input.get_key_down(GuiInput.KeyNS.TAB):
            self._avatar_switch_helper.switch_avatar()

        if not self._paused:
            self._update_grasping_and_set_act_hints()
            self._update_fetcher()
            self._sandbox_service.compute_action_and_step_env()
            self._fix_physics_for_target_objects()

        self._viz_fetcher(post_sim_update_dict)
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

    def is_app_state_done(self):
        # terminal neverending app state
        return False
