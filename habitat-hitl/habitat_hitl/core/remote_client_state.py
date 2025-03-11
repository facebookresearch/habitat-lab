#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn

from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.core.gui_input import GuiInput


# todo: rename to RemoteClientState
class RemoteGuiInput:
    def __init__(self, interprocess_record, debug_line_render):
        self._recent_client_states = []
        self._interprocess_record = interprocess_record
        self._gui_drawer = gui_drawer
        self._users = users

        self._receive_rate_tracker = AverageRateTracker(2.0)

        self._recent_client_states: List[ClientState] = []
        self._new_connection_records: List[ConnectionRecord] = []

        self._on_client_connected = Event()
        self._on_client_disconnected = Event()

        # Create one GuiInput per user to be controlled by remote clients.
        self._gui_inputs: List[GuiInput] = []
        for _ in users.indices(Mask.ALL):
            self._gui_inputs.append(GuiInput())

        # temp map VR button to key
        self._button_map = {
            0: GuiInput.KeyNS.ZERO,
            1: GuiInput.KeyNS.ONE,
            2: GuiInput.KeyNS.TWO,
            3: GuiInput.KeyNS.THREE,
        }

        self._gui_input = GuiInput()

        self._receive_count = 0

    def get_gui_input(self):
        """Internal GuiInput class."""
        return self._gui_input

    def get_history_length(self) -> int:
        """Length of client state history preserved. Anything beyond this horizon is discarded."""
        return 4

    def get_history_timestep(self) -> float:
        """Frequency at which client states are read."""
        return 1 / 60

    def pop_recent_server_keyframe_id(self) -> Optional[int]:
        """
        Removes and returns ("pops") the recentServerKeyframeId included in the latest client state.

        The removal behavior here is to help user code by only returning a keyframe ID when a new (unseen) one is available.
        """
        if len(self._recent_client_states) == 0:
            return None

        latest_client_state = self._recent_client_states[-1]
        if "recentServerKeyframeId" not in latest_client_state:
            return None

        retval = int(latest_client_state["recentServerKeyframeId"])
        del latest_client_state["recentServerKeyframeId"]
        return retval

    def get_recent_client_state_by_history_index(
        self, history_index: int
    ) -> Optional[ClientState]:
        assert history_index >= 0
        if history_index >= len(self._recent_client_states):
            return None

        return self._recent_client_states[-(1 + history_index)]

    def get_head_pose(
        self, history_index: int = 0
    ) -> Optional[Tuple[mn.Vector3, mn.Quaternion]]:
        """
        Get the latest head transform.
        Beware that this is in agent-space. Agents are flipped 180 degrees on the y-axis such as their z-axis faces forward.
        """
        client_state = self.get_recent_client_state_by_history_index(
            history_index
        )
        if not client_state:
            return None, None

        if "avatar" not in client_state:
            return None, None

        avatar_root_json = client_state["avatar"]["root"]

        pos_json = avatar_root_json["position"]
        pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
        rot_json = avatar_root_json["rotation"]
        rot_quat = mn.Quaternion(
            mn.Vector3(rot_json[1], rot_json[2], rot_json[3]), rot_json[0]
        )

        return pos, rot_quat

    def get_articulated_hand_pose(self, hand_idx, history_index=0):
        """
        Returns world-space positions and rotations of 26 hand bones.
        https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints
        typedef enum XrHandJointEXT {
            XR_HAND_JOINT_PALM_EXT = 0,
            XR_HAND_JOINT_WRIST_EXT = 1,
            XR_HAND_JOINT_THUMB_METACARPAL_EXT = 2,
            XR_HAND_JOINT_THUMB_PROXIMAL_EXT = 3,
            XR_HAND_JOINT_THUMB_DISTAL_EXT = 4,
            XR_HAND_JOINT_THUMB_TIP_EXT = 5,
            XR_HAND_JOINT_INDEX_METACARPAL_EXT = 6,
            XR_HAND_JOINT_INDEX_PROXIMAL_EXT = 7,
            XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT = 8,
            XR_HAND_JOINT_INDEX_DISTAL_EXT = 9,
            XR_HAND_JOINT_INDEX_TIP_EXT = 10,
            XR_HAND_JOINT_MIDDLE_METACARPAL_EXT = 11,
            XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT = 12,
            XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT = 13,
            XR_HAND_JOINT_MIDDLE_DISTAL_EXT = 14,
            XR_HAND_JOINT_MIDDLE_TIP_EXT = 15,
            XR_HAND_JOINT_RING_METACARPAL_EXT = 16,
            XR_HAND_JOINT_RING_PROXIMAL_EXT = 17,
            XR_HAND_JOINT_RING_INTERMEDIATE_EXT = 18,
            XR_HAND_JOINT_RING_DISTAL_EXT = 19,
            XR_HAND_JOINT_RING_TIP_EXT = 20,
            XR_HAND_JOINT_LITTLE_METACARPAL_EXT = 21,
            XR_HAND_JOINT_LITTLE_PROXIMAL_EXT = 22,
            XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT = 23,
            XR_HAND_JOINT_LITTLE_DISTAL_EXT = 24,
            XR_HAND_JOINT_LITTLE_TIP_EXT = 25,
            XR_HAND_JOINT_MAX_ENUM_EXT = 0x7FFFFFFF
        } XrHandJointEXT;
        """
        client_state = self.get_recent_client_state_by_history_index(
            history_index
        )
        if not client_state:
            return None, None

        if "avatar" not in client_state:
            return None, None

        art_hands_json = client_state["avatar"]["articulatedHands"]
        assert hand_idx >= 0 and hand_idx < len(art_hands_json)

        art_hand_json = art_hands_json[hand_idx]
        pos_json = art_hand_json["positions"]
        num_positions = len(pos_json) // 3

        rot_json = art_hand_json["rotations"]
        num_rotations = len(rot_json) // 4
        assert num_positions == num_rotations

        positions = []
        rotations = []
        for i in range(num_positions):
            pos = mn.Vector3(
                pos_json[i * 3 + 0], pos_json[i * 3 + 1], pos_json[i * 3 + 2]
            )
            positions.append(pos)

            rot_quat = mn.Quaternion(
                mn.Vector3(
                    rot_json[i * 4 + 1],
                    rot_json[i * 4 + 2],
                    rot_json[i * 4 + 3],
                ),
                rot_json[i * 4 + 0],
            )
            rotations.append(rot_quat)

        return positions, rotations

    def get_hand_pose(self, hand_idx, history_index=0):
        """
        Get the latest hand transforms.
        Beware that this is in agent-space. Agents are flipped 180 degrees on the y-axis such as their z-axis faces forward.
        """
        client_state = self.get_recent_client_state_by_history_index(
            history_index
        )
        if not client_state:
            return None, None

        if "avatar" not in client_state:
            return None, None

        assert "hands" in client_state["avatar"]
        hands_json = client_state["avatar"]["hands"]
        assert hand_idx >= 0 and hand_idx < len(hands_json)

        hand_json = hands_json[hand_idx]
        pos_json = hand_json["position"]
        pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
        rot_json = hand_json["rotation"]
        rot_quat = mn.Quaternion(
            mn.Vector3(rot_json[1], rot_json[2], rot_json[3]), rot_json[0]
        )

        return pos, rot_quat

    def _update_input_state(self, client_states: List[ClientState]) -> None:
        """Update mouse/keyboard input based on new client states."""
        if not len(client_states) or not len(self._gui_inputs):
            return

        # gather all recent keyDown and keyUp events
        for client_state in client_states:
            # Beware client_state input has dicts of bools (unlike GuiInput, which uses sets)
            if "input" not in client_state:
                continue

            input_json = client_state["input"]

            if "buttonHeld" not in input_json:
                continue

            # assume button containers are sets of buttonIndices
            for button in input_json["buttonDown"]:
                if button not in self._button_map:
                    print(f"button {button} not mapped!")
                    continue
                if True:
                    self._gui_input._key_down.add(self._button_map[button])
            for button in input_json["buttonUp"]:
                if button not in self._button_map:
                    print(f"key {button} not mapped!")
                    continue
                if True:
                    self._gui_input._key_up.add(self._button_map[button])

            if mouse_json is not None:
                mouse_buttons = mouse_json["buttons"]
                for button in mouse_buttons["buttonDown"]:
                    if button not in MouseButton:
                        continue
                    gui_input._mouse_button_down.add(MouseButton(button))
                for button in mouse_buttons["buttonUp"]:
                    if button not in MouseButton:
                        continue
                    gui_input._mouse_button_up.add(MouseButton(button))

                if "scrollDelta" in mouse_json:
                    delta: List[Any] = mouse_json["scrollDelta"]
                    if len(delta) == 2:
                        gui_input._mouse_scroll_offset += (
                            delta[0]
                            if abs(delta[0]) > abs(delta[1])
                            else delta[1]
                        )

                if "mousePositionDelta" in mouse_json:
                    pos_delta: List[Any] = mouse_json["mousePositionDelta"]
                    if len(pos_delta) == 2:
                        gui_input._relative_mouse_position = [
                            pos_delta[0],
                            pos_delta[1],
                        ]

                if "rayOrigin" in mouse_json:
                    ray_origin: List[float] = mouse_json["rayOrigin"]
                    ray_direction: List[float] = mouse_json["rayDirection"]
                    if len(ray_origin) == 3 and len(ray_direction) == 3:
                        ray = Ray()
                        ray.origin = mn.Vector3(
                            ray_origin[0], ray_origin[1], ray_origin[2]
                        )
                        ray.direction = mn.Vector3(
                            ray_direction[0],
                            ray_direction[1],
                            ray_direction[2],
                        ).normalized()
                        gui_input._mouse_ray = ray

        # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
        # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
        client_state = client_states[-1]
        if "input" not in client_state:
            return

        input_json = client_state["input"]

        if "buttonHeld" not in input_json:
            return

        gui_input._key_held.clear()
        gui_input._mouse_button_held.clear()

        for button in input_json["buttonHeld"]:
            if button not in self._button_map:
                print(f"button {button} not mapped!")
                continue
            if True:  # input_json["buttonHeld"][button]:
                self._gui_input._key_held.add(self._button_map[button])

        if mouse_json is not None:
            mouse_buttons = mouse_json["buttons"]
            for button in mouse_buttons["buttonHeld"]:
                if button not in MouseButton:
                    continue
                gui_input._mouse_button_held.add(MouseButton(button))

    def _debug_visualize_client(self) -> None:
        """Visualize the received VR inputs (head and hands)."""
        if not self._gui_drawer:
            return

        server_only = Mask.NONE  # Render on the server only.
        avatar_color = mn.Color3(0.3, 1, 0.3)

        pos, rot_quat = self.get_head_pose()
        if pos is not None and rot_quat is not None:
            trans = mn.Matrix4.from_(rot_quat.to_matrix(), pos)
            self._gui_drawer.push_transform(
                trans, destination_mask=server_only
            )
            color0 = avatar_color
            color1 = mn.Color4(
                avatar_color.r, avatar_color.g, avatar_color.b, 0
            )
            size = 0.5
            # draw a frustum (forward is z+)
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(size, size, size),
                color0,
                color1,
            )
            self._gui_drawer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(-size, size, size),
                color0,
                color1,
                destination_mask=server_only,
            )
            self._gui_drawer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(size, -size, size),
                color0,
                color1,
                destination_mask=server_only,
            )
            self._gui_drawer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(-size, -size, size),
                color0,
                color1,
                destination_mask=server_only,
            )

            self._gui_drawer.pop_transform(destination_mask=server_only)

        for hand_idx in range(2):
            # hand_pos, hand_rot_quat = self.get_hand_pose(hand_idx)
            # if hand_pos is not None and hand_rot_quat is not None:
            #     trans = mn.Matrix4.from_(hand_rot_quat.to_matrix(), hand_pos)
            #     self._debug_line_render.push_transform(trans)
            #     pointer_len = 0.5
            #     self._debug_line_render.draw_transformed_line(
            #         mn.Vector3(0, 0, 0),
            #         mn.Vector3(0, 0, pointer_len),
            #         color0,
            #         color1,
            #     )
            #     self._debug_line_render.pop_transform()

            art_hand_positions, art_hand_rotations = (
                self.get_articulated_hand_pose(hand_idx)
            )
            if (
                art_hand_positions is not None
                and art_hand_rotations is not None
            ):
                num_bones = len(art_hand_positions)
                for i in range(num_bones):
                    bone_pos = art_hand_positions[i]
                    bone_rot_quat = art_hand_rotations[i]
                    trans = mn.Matrix4.from_(
                        bone_rot_quat.to_matrix(), bone_pos
                    )
                    self._debug_line_render.push_transform(trans)
                    pointer_len = 0.02
                    self._debug_line_render.draw_transformed_line(
                        mn.Vector3(0, 0, 0),
                        mn.Vector3(0, 0, pointer_len),
                        color0,
                        color1,
                    )
                    self._debug_line_render.pop_transform()

    def _clean_history_by_connection_id(
        self, client_states: List[ClientState]
    ) -> None:
        """
        Clear history by connection id.
        Typically done after a client disconnect.
        """
        if not len(client_states):
            return

        latest_client_state = client_states[-1]
        if "connectionId" not in latest_client_state:
            return
        latest_connection_id = latest_client_state["connectionId"]

        # discard older states that don't match the latest connection id
        # iterate over items in reverse, starting with second-most recent client state
        for i in range(len(client_states) - 2, -1, -1):
            if client_states[i]["connectionId"] != latest_connection_id:
                client_states = client_states[i + 1 :]
                break

        # discard recent client states if they don't match the latest connection id
        latest_recent_client_state = (
            self.get_recent_client_state_by_history_index(0)
        )
        if (
            latest_recent_client_state
            and latest_recent_client_state["connectionId"]
            != latest_connection_id
        ):
            self.clear_history()

    def update(self) -> None:
        """Get the latest received remote client states."""
        self._new_connection_records = (
            self._interprocess_record.get_queued_connection_records()
        )
        new_disconnection_records = (
            self._interprocess_record.get_queued_disconnection_records()
        )

        for record in self._new_connection_records:
            self._on_client_connected.invoke(record)
        for record in new_disconnection_records:
            self._on_client_disconnected.invoke(record)

        client_states = self._interprocess_record.get_queued_client_states()
        self._receive_rate_tracker.increment(len(client_states))
        self._receive_count += len(client_states)

        # We expect to only process ~1 new client state at a time. If this assert fails, something is going awry with networking.
        # disabling because this happens all the time when debugging the main process
        # assert len(client_states) < 100

        self._clean_history_by_connection_id(client_states)

        self._update_input_state(client_states)

        # append to _recent_client_states, discarding old states to limit length to get_history_length()
        for client_state in client_states:
            self._recent_client_states.append(client_state)
            if len(self._recent_client_states) > self.get_history_length():
                self._recent_client_states.pop(0)

        self._debug_visualize_client()

    def get_new_connection_records(self) -> List[ConnectionRecord]:
        return self._new_connection_records

    def on_frame_end(self) -> None:
        for user_index in self._users.indices(Mask.ALL):
            self._gui_inputs[user_index].on_frame_end()
        self._new_connection_records = None

    def get_receive_count(self):
        return self._receive_count

    def clear_history(self):
        self._recent_client_states.clear()
        self._receive_count = 0
