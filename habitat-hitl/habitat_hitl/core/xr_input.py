#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import IntEnum
from typing import Final

from habitat_hitl.core.key_mapping import XRButton

NUM_CONTROLLERS: Final[int] = 2
HAND_LEFT: Final[int] = 0
HAND_RIGHT: Final[int] = 1


class XRController:
    """
    State of a single XR controller.
    """

    def __init__(self):
        self._buttons_held: set[XRButton] = set()
        self._buttons_down: set[XRButton] = set()
        self._buttons_up: set[XRButton] = set()
        self._buttons_touched: set[XRButton] = set()
        self._thumbstick_axis: list[float] = [0.0, 0.0]
        self._hand_trigger: float = 0.0
        self._index_trigger: float = 0.0
        self._is_controller_in_hand: bool = False

    def validate_button(button):
        assert isinstance(button, XRButton)

    def get_button(self, button):
        XRController.validate_button(button)
        return button in self._buttons_held

    def get_button_down(self, button):
        XRController.validate_button(button)
        return button in self._buttons_down

    def get_button_up(self, button):
        XRController.validate_button(button)
        return button in self._buttons_up

    def get_button_touched(self, button):
        XRController.validate_button(button)
        return button in self._buttons_touched

    def get_thumbstick(self):
        return self._thumbstick_axis

    def get_index_trigger(self):
        return self._index_trigger

    def get_hand_trigger(self):
        return self._hand_trigger

    def get_is_controller_in_hand(self):
        return self._is_controller_in_hand

    def reset(self, reset_continuous_input: bool = True):
        self._buttons_down.clear()
        self._buttons_up.clear()

        if reset_continuous_input:
            self._thumbstick_axis = [0, 0]
            self._hand_trigger = 0.0
            self._index_trigger = 0.0


class HandJoint(IntEnum):
    """
    Mapping of the 26 OpenXR hand joints.
    https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#convention-of-hand-joints
    """

    XR_HAND_JOINT_PALM_EXT = 0
    XR_HAND_JOINT_WRIST_EXT = 1
    XR_HAND_JOINT_THUMB_METACARPAL_EXT = 2
    XR_HAND_JOINT_THUMB_PROXIMAL_EXT = 3
    XR_HAND_JOINT_THUMB_DISTAL_EXT = 4
    XR_HAND_JOINT_THUMB_TIP_EXT = 5
    XR_HAND_JOINT_INDEX_METACARPAL_EXT = 6
    XR_HAND_JOINT_INDEX_PROXIMAL_EXT = 7
    XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT = 8
    XR_HAND_JOINT_INDEX_DISTAL_EXT = 9
    XR_HAND_JOINT_INDEX_TIP_EXT = 10
    XR_HAND_JOINT_MIDDLE_METACARPAL_EXT = 11
    XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT = 12
    XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT = 13
    XR_HAND_JOINT_MIDDLE_DISTAL_EXT = 14
    XR_HAND_JOINT_MIDDLE_TIP_EXT = 15
    XR_HAND_JOINT_RING_METACARPAL_EXT = 16
    XR_HAND_JOINT_RING_PROXIMAL_EXT = 17
    XR_HAND_JOINT_RING_INTERMEDIATE_EXT = 18
    XR_HAND_JOINT_RING_DISTAL_EXT = 19
    XR_HAND_JOINT_RING_TIP_EXT = 20
    XR_HAND_JOINT_LITTLE_METACARPAL_EXT = 21
    XR_HAND_JOINT_LITTLE_PROXIMAL_EXT = 22
    XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT = 23
    XR_HAND_JOINT_LITTLE_DISTAL_EXT = 24
    XR_HAND_JOINT_LITTLE_TIP_EXT = 25

    @staticmethod
    def joint_count():
        return 26


class XRHand:
    """State of a XR hand."""

    def __init__(self, hand_type: type[HandJoint] = HandJoint):
        self._is_tracked = False
        self._is_data_high_confidence = False
        self._is_data_valid = False
        self._joint_count = hand_type.joint_count()
        self._positions: list[list[float]] = [
            [0.0, 0.0, 0.0] for _ in range(self._joint_count)
        ]
        self._rotations: list[list[float]] = [
            [1.0, 0.0, 0.0, 0.0] for _ in range(self._joint_count)
        ]

    def update_hand_pose(self, positions: list[float], rotations: list[float]):
        """Update the hand pose from two lists of `joint_count` positions and rotations."""
        joint_count = min(
            self._joint_count, len(positions) // 3, len(rotations) // 4
        )

        for i in range(joint_count):
            pos = [
                positions[i * 3 + 0],
                positions[i * 3 + 1],
                positions[i * 3 + 2],
            ]
            self._positions[i] = pos
            # [wxyz] or [scalar(1), vector(3)]
            rot = [
                rotations[i * 4 + 0],
                rotations[i * 4 + 1],
                rotations[i * 4 + 2],
                rotations[i * 4 + 3],
            ]
            self._rotations[i] = rot

    @property
    def detected(self) -> bool:
        """Whether the hand is being tracked with high confidence."""
        return (
            self._is_tracked
            and self._is_data_valid
            and self._is_data_high_confidence
        )

    @property
    def joint_count(self) -> int:
        """Number of joints in the hand."""
        return self._joint_count

    @property
    def joint_positions(self) -> list[list[float]]:
        """List of joint positions, in the same order as the `HandJoint` enum."""
        return self._positions

    @property
    def joint_rotations(self) -> list[list[float]]:
        """List of joint rotations, in the same order as the `HandJoint` enum."""
        return self._rotations

    def get_joint_transform(
        self, index: int | HandJoint
    ) -> tuple[list[float], list[float]]:
        """Get the `(position, rotation)` of a specific hand joint."""
        assert (
            index < self._joint_count
        ), f"The specified joint index '{index}' is out of bounds ({self._joint_count})."
        return (self._positions[index], self._rotations[index])


class XRInput:
    """
    State of the XR input system (HMD and controllers).
    """

    def __init__(self):
        self._controllers: list[XRController] = [
            XRController() for _ in range(NUM_CONTROLLERS)
        ]
        self._hands: list[XRHand] = [XRHand() for _ in range(NUM_CONTROLLERS)]

        self._origin_position: list[float] = [0.0, 0.0, 0.0]
        # [wxyz] or [scalar(1), vector(3)]
        self._origin_rotation: list[float] = [0.0, 0.0, 0.0, 0.0]

    @property
    def controllers(self):
        return self._controllers

    @property
    def hands(self):
        return self._hands

    @property
    def left_controller(self):
        return self._controllers[HAND_LEFT]

    @property
    def right_controller(self):
        return self._controllers[HAND_RIGHT]

    @property
    def origin_position(self):
        return self._origin_position

    @property
    def origin_rotation(self):
        return self._origin_rotation

    def reset(self, reset_continuous_input: bool = True):
        """
        Reset the input states. To be called at the end of a frame.

        `reset_continuous_input`: controls whether to reset continuous input like scrolling or dragging.
        Remote clients send their input at a different frequency than the server framerate.
        To avoid choppiness, their continuous inputs should be reset before consolidating new remote inputs.
        This differs from discrete input like clicking, which must be reset every frame to avoid extending click events across multiple frames.
        """
        for controller in self._controllers:
            controller.reset(reset_continuous_input)
