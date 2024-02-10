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
    def __init__(self, interprocess_record, debug_line_render, gui_input):
        self._recent_client_states = []
        self._interprocess_record = interprocess_record
        self._debug_line_render = debug_line_render

        self._receive_rate_tracker = AverageRateTracker(2.0)

        self._connection_params = None

        # TODO: VR and keyboard share same keys.
        # temp map VR button to key
        self._button_map = {
            0x04: GuiInput.KeyNS.A,
            0x05: GuiInput.KeyNS.B,
            0x06: GuiInput.KeyNS.C,
            0x07: GuiInput.KeyNS.D,
            0x08: GuiInput.KeyNS.E,
            0x09: GuiInput.KeyNS.F,
            0x0A: GuiInput.KeyNS.G,
            0x0B: GuiInput.KeyNS.H,
            0x0C: GuiInput.KeyNS.I,
            0x0D: GuiInput.KeyNS.J,
            0x0E: GuiInput.KeyNS.K,
            0x0F: GuiInput.KeyNS.L,
            0x10: GuiInput.KeyNS.M,
            0x11: GuiInput.KeyNS.N,
            0x12: GuiInput.KeyNS.O,
            0x13: GuiInput.KeyNS.P,
            0x14: GuiInput.KeyNS.Q,
            0x15: GuiInput.KeyNS.R,
            0x16: GuiInput.KeyNS.S,
            0x17: GuiInput.KeyNS.T,
            0x18: GuiInput.KeyNS.U,
            0x19: GuiInput.KeyNS.V,
            0x1A: GuiInput.KeyNS.W,
            0x1B: GuiInput.KeyNS.X,
            0x1C: GuiInput.KeyNS.Y,
            0x1D: GuiInput.KeyNS.Z,
            0x27: GuiInput.KeyNS.ZERO,
            0x1E: GuiInput.KeyNS.ONE,
            0x1F: GuiInput.KeyNS.TWO,
            0x20: GuiInput.KeyNS.THREE,
            0x21: GuiInput.KeyNS.FOUR,
            0x22: GuiInput.KeyNS.FIVE,
            0x23: GuiInput.KeyNS.SIX,
            0x24: GuiInput.KeyNS.SEVEN,
            0x25: GuiInput.KeyNS.EIGHT,
            0x26: GuiInput.KeyNS.NINE,
            0x2C: GuiInput.KeyNS.SPACE,
            # TODO: Other keys
        }

        self._mouse_button_map = {
            0: GuiInput.MouseNS.LEFT,
            1: GuiInput.MouseNS.RIGHT,
            2: GuiInput.MouseNS.MIDDLE,
        }

        self._gui_input = gui_input #TODO: Apply refactor

    def get_gui_input(self):
        return self._gui_input

    def get_history_length(self):
        return 4

    def get_history_timestep(self):
        return 1 / 60

    def get_connection_params(self):
        return self._connection_params

    def pop_recent_server_keyframe_id(self):
        """
        Removes and returns ("pops") the recentServerKeyframeId included in the latest client state.

        The removal behavior here is to help user code by only returning a keyframe ID when a new (unseen) one is available.
        """
        if len(self._recent_client_states) == 0:
            return None

        latest_client_state = self._recent_client_states[0]
        if "recentServerKeyframeId" not in latest_client_state:
            return None

        retval = int(latest_client_state["recentServerKeyframeId"])
        del latest_client_state["recentServerKeyframeId"]
        return retval

    def get_head_pose(self, history_index=0):
        if history_index >= len(self._recent_client_states):
            return None, None

        client_state = self._recent_client_states[history_index]

        if "avatar" not in client_state:
            return None, None

        avatar_root_json = client_state["avatar"]["root"]

        pos_json = avatar_root_json["position"]
        # TODO: This is XR-specific
        if len(pos_json) == 0:
            return None, None
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
        # TODO: This is XR-specific
        if len(pos_json) > 0:
            pos = mn.Vector3(pos_json[0], pos_json[1], pos_json[2])
            rot_json = hand_json["rotation"]
            rot_quat = mn.Quaternion(
                mn.Vector3(rot_json[1], rot_json[2], rot_json[3]), rot_json[0]
            )

            return pos, rot_quat
        return None, None

    def update_input_state_from_remote_client_states(self, client_states):
        if not len(client_states):
            return

        # gather all recent keyDown and keyUp events
        for client_state in client_states:
            # Beware client_state input has dicts of bools (unlike GuiInput, which uses sets)

            input_json = client_state["input"] if "input" in client_state else None # TODO: Split keyboard and VR
            mouse_json = client_state["mouse"] if "mouse" in client_state else None

            # assume button containers are sets of buttonIndices
            if input_json:
                for button in input_json["buttonDown"]:
                    if button not in self._button_map:
                        print(f"button {button} not mapped!")
                        continue
                    self._gui_input._key_down.add(self._button_map[button])
                for button in input_json["buttonUp"]:
                    if button not in self._button_map:
                        print(f"key {button} not mapped!")
                        continue
                    self._gui_input._key_up.add(self._button_map[button])

            if mouse_json and "buttons" in mouse_json:
                mouse_buttons = mouse_json["buttons"]
                for button in mouse_buttons["buttonDown"]:
                    if button not in self._mouse_button_map:
                        print(f"button {button} not mapped!")
                        continue
                    self._gui_input._mouse_button_down.add(self._mouse_button_map[button])

                for button in mouse_buttons["buttonUp"]:
                    if button not in self._mouse_button_map:
                        print(f"button {button} not mapped!")
                        continue
                    self._gui_input._mouse_button_up.add(self._mouse_button_map[button])

                if "scrollDelta" in mouse_json:
                    delta = mouse_json["scrollDelta"]
                    if len(delta) == 2:
                        self._gui_input._mouse_scroll_offset += delta[0] if abs(delta[0]) > abs(delta[1]) else delta[1]

        # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
        # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
        client_state = client_states[-1]

        input_json = client_state["input"] if "input" in client_state else None # TODO: Split keyboard and VR
        mouse_json = client_state["mouse"] if "mouse" in client_state else None

        self._gui_input._key_held.clear()
        self._gui_input._mouse_button_held.clear()

        if input_json:
            for button in input_json["buttonHeld"]:
                if button not in self._button_map:
                    print(f"button {button} not mapped!")
                    continue
                self._gui_input._key_held.add(self._button_map[button])

        if mouse_json and "buttons" in mouse_json:
            mouse_buttons = mouse_json["buttons"]
            for button in mouse_buttons["buttonHeld"]:
                if button not in self._mouse_button_map:
                    print(f"button {button} not mapped!")
                    continue
                self._gui_input._mouse_button_held.add(self._mouse_button_map[button])

        # TODO: Implement headless-compatible mouse handling.
        # if "mousePosition" in input_json:
        #     ...

    def debug_visualize_client(self):
        if not self._debug_line_render:
            return

        if not len(self._recent_client_states):
            return

        avatar_color = mn.Color3(0.3, 1, 0.3)

        pos, rot_quat = self.get_head_pose()
        if pos is not None and rot_quat is not None:
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
            if hand_pos is not None and hand_rot_quat is not None:
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

    def _update_connection_params(self, client_states):
        # Note we only parse the first connection_params we find. The client is expected to only send this once.
        for client_state in client_states:
            if "connection_params_dict" in client_state:

                def validate_connection_params_dict(obj):
                    if not isinstance(obj, dict) and not (
                        hasattr(obj, "keys") and callable(obj.keys)
                    ):
                        raise TypeError(
                            "connection_params_dict is not dictionary-like."
                        )

                    if not all(isinstance(key, str) for key in obj.keys()):
                        raise ValueError(
                            "All keys in connection_params_dict must be strings."
                        )

                connection_params_dict = client_state["connection_params_dict"]
                validate_connection_params_dict(connection_params_dict)
                self._connection_params = connection_params_dict
                break

            elif "connection_params_query_string" in client_state:
                from urllib.parse import parse_qs

                def query_string_to_dict(query):
                    parsed_query = parse_qs(query)
                    # Convert each list of values to a single value (the first one)
                    return {k: v[0] for k, v in parsed_query.items()}

                self._connection_params = query_string_to_dict(
                    client_state["connection_params_query_string"]
                )
                break

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

        self._update_connection_params(client_states)

        self.debug_visualize_client()

    def on_frame_end(self):
        self._gui_input.on_frame_end()
        self._connection_params = None

    def clear_history(self):
        self._recent_client_states.clear()
