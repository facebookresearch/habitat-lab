#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import magnum as mn

from habitat_hitl._internal.networking.average_rate_tracker import (
    AverageRateTracker,
)
from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
)
from habitat_hitl.core.client_helper import ClientHelper
from habitat_hitl.core.client_message_manager import ClientMessageManager
from habitat_hitl.core.event import Event
from habitat_hitl.core.gui_drawer import GuiDrawer
from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.key_mapping import KeyCode, MouseButton
from habitat_hitl.core.types import (
    ClientState,
    ConnectionRecord,
    DisconnectionRecord,
)
from habitat_hitl.core.user_mask import Mask, Users
from habitat_sim.geo import Ray


class RemoteClientState:
    """
    Class that tracks the state of a remote client.
    This includes handling of remote input and client messages.
    """

    def __init__(
        self,
        hitl_config,  # TODO: Coupling with ClientHelper
        client_message_manager: ClientMessageManager,  # TODO: Coupling with ClientHelper
        interprocess_record: InterprocessRecord,
        gui_drawer: GuiDrawer,
        users: Users,
    ):
        self._interprocess_record = interprocess_record
        self._gui_drawer = gui_drawer
        self._users = users

        self._new_connection_records: List[ConnectionRecord] = []

        self._on_client_connected = Event()
        self._on_client_disconnected = Event()

        # TODO: Handle UI in a different class.
        self._pressed_ui_buttons: List[Set[str]] = []
        self._textboxes: List[Dict[str, str]] = []

        self._gui_inputs: List[GuiInput] = []
        self._client_state_history: List[List[ClientState]] = []
        self._receive_rate_trackers: List[AverageRateTracker] = []
        for _ in range(users.max_user_count):
            self._gui_inputs.append(GuiInput())
            self._client_state_history.append([])
            self._receive_rate_trackers.append(AverageRateTracker(2.0))
            self._pressed_ui_buttons.append(set())
            self._textboxes.append({})

        self._client_loading: List[bool] = [False] * users.max_user_count

        # TODO: Temporary coupling.
        #       ClientHelper lifetime is directly coupled with RemoteClientState.
        self._client_helper = ClientHelper(
            hitl_config, self, client_message_manager, users
        )

        # temp map VR button to key
        self._button_map = {
            0: KeyCode.ZERO,
            1: KeyCode.ONE,
            2: KeyCode.TWO,
            3: KeyCode.THREE,
        }

    @property
    def on_client_connected(self) -> Event:
        return self._on_client_connected

    @property
    def on_client_disconnected(self) -> Event:
        return self._on_client_disconnected

    def get_gui_input(self, user_index: int = 0) -> GuiInput:
        """Get the GuiInput for a specified user index."""
        return self._gui_inputs[user_index]

    def get_gui_inputs(self) -> List[GuiInput]:
        """Get a list of all GuiInputs indexed by user index."""
        return self._gui_inputs

    def is_user_loading(self, user_index: int) -> bool:
        """Return true if the specified user's client is in a loading state."""
        return self._client_loading[user_index]

    def bind_gui_input(self, gui_input: GuiInput, user_index: int) -> None:
        """
        Bind the specified GuiInput to a specified user, allowing the associated remote client to control it.
        Erases the previous GuiInput.
        """
        assert user_index < len(self._gui_inputs)
        self._gui_inputs[user_index] = gui_input

    def ui_button_pressed(self, user_index: int, button_id: str) -> bool:
        return button_id in self._pressed_ui_buttons[user_index]

    def get_textbox_content(self, user_index: int, textbox_id: str) -> str:
        user_textboxes = self._textboxes[user_index]
        return user_textboxes.get(textbox_id, "")

    def get_history_length(self) -> int:
        """Length of client state history preserved. Anything beyond this horizon is discarded."""
        return 4

    def get_history_timestep(self) -> float:
        """Frequency at which client states are read."""
        return 1 / 60

    def pop_recent_server_keyframe_id(self, user_index: int) -> Optional[int]:
        """
        Removes and returns ("pops") the recentServerKeyframeId included in the latest client state.

        The removal behavior here is to help user code by only returning a keyframe ID when a new (unseen) one is available.
        """
        if len(self._client_state_history[user_index]) == 0:
            return None

        latest_client_state = self._client_state_history[user_index][-1]
        if "recentServerKeyframeId" not in latest_client_state:
            return None

        retval = int(latest_client_state["recentServerKeyframeId"])
        del latest_client_state["recentServerKeyframeId"]
        return retval

    def get_recent_client_state_by_history_index(
        self, user_index: int, history_index: int
    ) -> Optional[ClientState]:
        assert history_index >= 0
        if history_index >= len(self._client_state_history[user_index]):
            return None

        return self._client_state_history[user_index][-(1 + history_index)]

    def get_head_pose(
        self, user_index: int, history_index: int = 0
    ) -> Optional[Tuple[mn.Vector3, mn.Quaternion]]:
        """
        Get the latest head transform.
        Beware that this is in agent-space. Agents are flipped 180 degrees on the y-axis such as their z-axis faces forward.
        """
        client_state = self.get_recent_client_state_by_history_index(
            user_index, history_index
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

    def get_hand_pose(
        self, user_index: int, hand_idx: int, history_index: int = 0
    ) -> Optional[Tuple[mn.Vector3, mn.Quaternion]]:
        """
        Get the latest hand transforms.
        Beware that this is in agent-space. Agents are flipped 180 degrees on the y-axis such as their z-axis faces forward.
        """
        client_state = self.get_recent_client_state_by_history_index(
            user_index, history_index
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

    def _group_client_states_by_user_index(
        self, client_states: List[ClientState]
    ) -> List[List[ClientState]]:
        """
        Group a list of client states by user index.
        """
        output: List[List[ClientState]] = []
        for _ in range(self._users.max_user_count):
            output.append([])

        for client_state in client_states:
            user_index = client_state["userIndex"]
            assert user_index < self._users.max_user_count
            output[user_index].append(client_state)
        return output

    def _update_input_state(
        self, all_client_states: List[List[ClientState]]
    ) -> None:
        """Update mouse/keyboard input based on new client states."""
        if len(all_client_states) == 0 or len(self._gui_inputs) == 0:
            return

        # Gather all input events.
        for user_index in range(len(all_client_states)):
            client_states = all_client_states[user_index]
            if len(client_states) == 0:
                continue
            gui_input = self._gui_inputs[user_index]
            mouse_scroll_offset: float = 0.0
            relative_mouse_position: List[int] = [0, 0]

            for client_state in client_states:
                # UI element events.
                for ui_dict in ["ui", "legacyUi"]:
                    ui = client_state.get(ui_dict, None)
                    if ui is not None:
                        for button in ui.get("buttonsPressed", []):
                            self._pressed_ui_buttons[user_index].add(button)
                        for textbox_id, text in ui.get(
                            "textboxes", {}
                        ).items():
                            self._textboxes[user_index][textbox_id] = text

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
                        gui_input._key_down.add(KeyCode(button))
                    for button in input_json["buttonUp"]:
                        if button not in KeyCode:
                            continue
                        gui_input._key_up.add(KeyCode(button))

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
                            mouse_scroll_offset += (
                                delta[0]
                                if abs(delta[0]) > abs(delta[1])
                                else delta[1]
                            )

                    if "mousePositionDelta" in mouse_json:
                        pos_delta: List[Any] = mouse_json["mousePositionDelta"]
                        if len(pos_delta) == 2:
                            relative_mouse_position[0] += pos_delta[0]
                            relative_mouse_position[1] += pos_delta[1]

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

            gui_input._mouse_scroll_offset = mouse_scroll_offset
            gui_input._relative_mouse_position = relative_mouse_position

            # todo: think about ambiguous GuiInput states (key-down and key-up events in the same
            # frame and other ways that keyHeld, keyDown, and keyUp can be inconsistent.
            last_client_state = client_states[-1]

            # Loading states.
            self._client_loading[user_index] = last_client_state.get(
                "isLoading", False
            )

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

            gui_input._key_held.clear()
            gui_input._mouse_button_held.clear()

            if input_json is not None:
                for button in input_json["buttonHeld"]:
                    if button not in KeyCode:
                        continue
                    gui_input._key_held.add(KeyCode(button))

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

        for user_index in self._users.indices(Mask.ALL):
            pos, rot_quat = self.get_head_pose(user_index)
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

                # Draw a frustum (forward is flipped (z+))
                self._gui_drawer.draw_transformed_line(
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

            # Draw controller rays (forward is flipped (z+))
            for hand_idx in range(2):
                hand_pos, hand_rot_quat = self.get_hand_pose(
                    user_index, hand_idx
                )
                if hand_pos is not None and hand_rot_quat is not None:
                    trans = mn.Matrix4.from_(
                        hand_rot_quat.to_matrix(), hand_pos
                    )
                    self._gui_drawer.push_transform(
                        trans, destination_mask=server_only
                    )
                    pointer_len = 0.5
                    self._gui_drawer.draw_transformed_line(
                        mn.Vector3(0, 0, 0),
                        mn.Vector3(0, 0, pointer_len),
                        color0,
                        color1,
                        destination_mask=server_only,
                    )
                    self._gui_drawer.pop_transform(
                        destination_mask=server_only
                    )

    def _clean_disconnected_user_history(
        self,
        disconnection_record: DisconnectionRecord,
        all_client_states: List[List[ClientState]],
    ) -> None:
        """
        Clear history by connection id. Done after a client disconnect.
        """
        user_index: int = disconnection_record["userIndex"]
        all_client_states[user_index].clear()

    def update(self) -> None:
        """Get the latest received remote client states."""
        self._new_connection_records = (
            self._interprocess_record.get_queued_connection_records()
        )
        new_disconnection_records = (
            self._interprocess_record.get_queued_disconnection_records()
        )

        assorted_client_states = (
            self._interprocess_record.get_queued_client_states()
        )
        client_states = self._group_client_states_by_user_index(
            assorted_client_states
        )
        for user_index in range(len(client_states)):
            user_client_states = client_states[user_index]
            self._receive_rate_trackers[user_index].increment(
                len(user_client_states)
            )

        for record in self._new_connection_records:
            self._on_client_connected.invoke(record)
        for record in new_disconnection_records:
            self._on_client_disconnected.invoke(record)
            self._clean_disconnected_user_history(record, client_states)

        # We expect to only process ~1 new client state at a time. If this assert fails, something is going awry with networking.
        # disabling because this happens all the time when debugging the main process
        # assert len(client_states) < 100

        self._update_input_state(client_states)

        # append to _recent_client_states, discarding old states to limit length to get_history_length()
        for user_index in range(len(client_states)):
            for client_state in client_states[user_index]:
                self._client_state_history[user_index].append(client_state)
                if (
                    len(self._client_state_history[user_index])
                    > self.get_history_length()
                ):
                    self._client_state_history[user_index].pop(0)

        self._debug_visualize_client()

    def get_new_connection_records(self) -> List[ConnectionRecord]:
        return self._new_connection_records

    def on_frame_end(self) -> None:
        for user_index in self._users.indices(Mask.ALL):
            self._gui_inputs[user_index].reset(reset_continuous_input=False)
            self._pressed_ui_buttons[user_index].clear()
            self._textboxes[user_index].clear()
        self._new_connection_records = None

    def clear_history(self, user_mask=Mask.ALL) -> None:
        for user_index in self._users.indices(user_mask):
            self._client_state_history[user_index].clear()
            self._pressed_ui_buttons[user_index].clear()
            self._textboxes[user_index].clear()

    def kick(self, user_mask: Mask) -> None:
        """
        Immediately kick the users matching the specified user mask.
        """
        for user_index in self._users.indices(user_mask):
            self._interprocess_record.send_kick_signal_to_networking_thread(
                user_index
            )
