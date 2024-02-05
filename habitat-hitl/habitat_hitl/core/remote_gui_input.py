#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn

from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl.core.gui_input import GuiInput


class RemoteGuiInput:
    def __init__(self, interprocess_record, debug_line_render):
        self._recent_client_states = []
        self._interprocess_record = interprocess_record
        self._debug_line_render = debug_line_render

        self._receive_rate_tracker = AverageRateTracker(2.0)

        # temp map VR button to key
        self._button_map = {
            0: GuiInput.KeyNS.ZERO,
            1: GuiInput.KeyNS.ONE,
            2: GuiInput.KeyNS.TWO,
            3: GuiInput.KeyNS.THREE,
        }

        self._gui_input = GuiInput()

    def get_gui_input(self):
        return self._gui_input

    def get_history_length(self):
        return 4

    def get_history_timestep(self):
        return 1 / 60

    def get_head_pose(self, history_index=0):
        if history_index >= len(self._recent_client_states):
            return None, None

        client_state = self._recent_client_states[history_index]

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

    def get_hand_pose(self, hand_idx, history_index=0):
        if history_index >= len(self._recent_client_states):
            return None, None

        client_state = self._recent_client_states[history_index]

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

    def update_input_state_from_remote_client_states(self, client_states):
        if not len(client_states):
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

        # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
        # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
        client_state = client_states[-1]
        if "input" not in client_state:
            return

        input_json = client_state["input"]

        if "buttonHeld" not in input_json:
            return

        self._gui_input._key_held.clear()

        for button in input_json["buttonHeld"]:
            if button not in self._button_map:
                print(f"button {button} not mapped!")
                continue
            if True:  # input_json["buttonHeld"][button]:
                self._gui_input._key_held.add(self._button_map[button])

    def debug_visualize_client(self):
        if not self._debug_line_render:
            return

        if not len(self._recent_client_states):
            return

        avatar_color = mn.Color3(0.3, 1, 0.3)

        if True:
            pos, rot_quat = self.get_head_pose()
            trans = mn.Matrix4.from_(rot_quat.to_matrix(), pos)
            self._debug_line_render.push_transform(trans)
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
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(-size, size, size),
                color0,
                color1,
            )
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(size, -size, size),
                color0,
                color1,
            )
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(-size, -size, size),
                color0,
                color1,
            )

            self._debug_line_render.pop_transform()

        for hand_idx in range(2):
            hand_pos, hand_rot_quat = self.get_hand_pose(hand_idx)
            trans = mn.Matrix4.from_(hand_rot_quat.to_matrix(), hand_pos)
            self._debug_line_render.push_transform(trans)
            pointer_len = 0.5
            self._debug_line_render.draw_transformed_line(
                mn.Vector3(0, 0, 0),
                mn.Vector3(0, 0, pointer_len),
                color0,
                color1,
            )

            self._debug_line_render.pop_transform()

    def update(self):
        client_states = self._interprocess_record.get_queued_client_states()
        self._receive_rate_tracker.increment(len(client_states))

        if len(client_states) > self.get_history_length():
            client_states = client_states[-self.get_history_length() :]

        for client_state in client_states:
            self._recent_client_states.insert(0, client_state)
            if (
                len(self._recent_client_states)
                == self.get_history_length() + 1
            ):
                self._recent_client_states.pop()

        self.update_input_state_from_remote_client_states(client_states)

        self.debug_visualize_client()

    def on_frame_end(self):
        self._gui_input.on_frame_end()

    def clear_history(self):
        self._recent_client_states.clear()
