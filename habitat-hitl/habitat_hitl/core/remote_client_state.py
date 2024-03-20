#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List

import magnum as mn

from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
)
from habitat_hitl.core.gui_drawer import GuiDrawer
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import KeyCode


class RemoteClientState:
    """
    Class that tracks the state of a remote client.
    This includes handling of remote input and client messages.
    """

    def __init__(
        self,
        interprocess_record: InterprocessRecord,
        gui_drawer: GuiDrawer,
        gui_input: GuiInput,
    ):
        self._gui_input = gui_input
        self._recent_client_states: List[Any] = []
        self._interprocess_record = interprocess_record
        self._gui_drawer = gui_drawer

        self._receive_rate_tracker = AverageRateTracker(2.0)

        self._new_connection_records: List[Any] = []

        # temp map VR button to key
        self._button_map = {
            0: GuiInput.KeyNS.ZERO,
            1: GuiInput.KeyNS.ONE,
            2: GuiInput.KeyNS.TWO,
            3: GuiInput.KeyNS.THREE,
        }

    def get_gui_input(self):
        """Internal GuiInput class."""
        return self._gui_input

    def get_history_length(self):
        """Length of client state history preserved. Anything beyond this horizon is discarded."""
        return 4

    def get_history_timestep(self):
        """Frequency at which client states are read."""
        return 1 / 60

    def pop_recent_server_keyframe_id(self):
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

    def get_recent_client_state_by_history_index(self, history_index):
        assert history_index >= 0
        if history_index >= len(self._recent_client_states):
            return None

        return self._recent_client_states[-(1 + history_index)]

    def get_head_pose(self, history_index=0):
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
        # Beware that this is an agent-space quaternion.
        # Agents are flipped 180 degrees on the y-axis such as their z-axis faces forward.
        rot_quat = rot_quat * mn.Quaternion.rotation(
            mn.Rad(math.pi), mn.Vector3(0, 1.0, 0)
        )
        return pos, rot_quat

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
        # Beware that this is an agent-space quaternion.
        # Agents are flipped 180 degrees on the y-axis such as their z-axis faces forward.
        rot_quat = rot_quat * mn.Quaternion.rotation(
            mn.Rad(math.pi), mn.Vector3(0, 1.0, 0)
        )
        return pos, rot_quat

    def _update_input_state(self, client_states):
        """Update mouse/keyboard input based on new client states."""
        if not len(client_states):
            return

        # Gather all recent keyDown and keyUp events
        for client_state in client_states:
            input_json = (
                client_state["input"] if "input" in client_state else None
            )
            mouse_json = (
                client_state["mouse"] if "mouse" in client_state else None
            )

            if input_json is not None:
                for button in input_json["buttonDown"]:
                    if button not in KeyCode:
                        continue
                    self._gui_input._key_down.add(KeyCode(button))
                for button in input_json["buttonUp"]:
                    if button not in KeyCode:
                        continue
                    self._gui_input._key_up.add(KeyCode(button))

            if mouse_json is not None:
                mouse_buttons = mouse_json["buttons"]
                for button in mouse_buttons["buttonDown"]:
                    if button not in KeyCode:
                        continue
                    self._gui_input._mouse_button_down.add(KeyCode(button))
                for button in mouse_buttons["buttonUp"]:
                    if button not in KeyCode:
                        continue
                    self._gui_input._mouse_button_up.add(KeyCode(button))

                delta: List[Any] = mouse_json["scrollDelta"]
                if len(delta) == 2:
                    self._gui_input._mouse_scroll_offset += (
                        delta[0] if abs(delta[0]) > abs(delta[1]) else delta[1]
                    )

        # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
        # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
        last_client_state = client_states[-1]

        input_json = (
            last_client_state["input"]
            if "input" in last_client_state
            else None
        )
        mouse_json = (
            last_client_state["mouse"]
            if "mouse" in last_client_state
            else None
        )

        self._gui_input._key_held.clear()

        if input_json is not None:
            for button in input_json["buttonHeld"]:
                if button not in KeyCode:
                    continue
                self._gui_input._key_held.add(KeyCode(button))

        if mouse_json is not None:
            mouse_buttons = mouse_json["buttons"]
            for button in mouse_buttons["buttonHeld"]:
                if button not in KeyCode:
                    continue
                self._gui_input._mouse_button_held.add(KeyCode(button))

    def debug_visualize_client(self):
        """Visualize the received VR inputs (head and hands)."""
        # Sloppy: Use internal debug_line_render to render on server only.
        line_renderer = self._gui_drawer.get_sim_debug_line_render()
        if not line_renderer:
            return

        avatar_color = mn.Color3(0.3, 1, 0.3)

        pos, rot_quat = self.get_head_pose()
        if pos is not None and rot_quat is not None:
            trans = mn.Matrix4.from_(rot_quat.to_matrix(), pos)
            line_renderer.push_transform(trans)
            color0 = avatar_color
            color1 = mn.Color4(
                avatar_color.r, avatar_color.g, avatar_color.b, 0
            )
            size = 0.5

            # Draw a frustum (forward is flipped (z+))
            line_renderer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(size, size, size),
                color0,
                color1,
            )
            line_renderer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(-size, size, size),
                color0,
                color1,
            )
            line_renderer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(size, -size, size),
                color0,
                color1,
            )
            line_renderer.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(-size, -size, size),
                color0,
                color1,
            )

            line_renderer.pop_transform()

        # Draw controller rays (forward is flipped (z+))
        for hand_idx in range(2):
            hand_pos, hand_rot_quat = self.get_hand_pose(hand_idx)
            if hand_pos is not None and hand_rot_quat is not None:
                trans = mn.Matrix4.from_(hand_rot_quat.to_matrix(), hand_pos)
                line_renderer.push_transform(trans)
                pointer_len = 0.5
                line_renderer.draw_transformed_line(
                    mn.Vector3(0, 0, 0),
                    mn.Vector3(0, 0, pointer_len),
                    color0,
                    color1,
                )
                line_renderer.pop_transform()

    def _clean_history_by_connection_id(self, client_states):
        """
        Clear history by connection id.
        Typically done after a client disconnect.
        """
        if not len(client_states):
            return

        latest_client_state = client_states[-1]
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

    def update(self):
        """Get the latest received remote client states."""
        self._new_connection_records = (
            self._interprocess_record.get_queued_connection_records()
        )

        client_states = self._interprocess_record.get_queued_client_states()
        self._receive_rate_tracker.increment(len(client_states))

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

        self.debug_visualize_client()

    def get_new_connection_records(self):
        return self._new_connection_records

    def on_frame_end(self):
        self._gui_input.on_frame_end()
        self._new_connection_records = None

    def clear_history(self):
        self._recent_client_states.clear()
