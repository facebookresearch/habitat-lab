#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import magnum as mn
from spatialmath import SE3, Quaternion

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.key_mapping import XRButton


def btn(button_list: list[XRButton]) -> str:
    button_names = [
        button.name for button in button_list if isinstance(button, XRButton)
    ]
    return f"[{', '.join(button_names)}]"


def pos(legacy_tuple: Optional[tuple[mn.Vector3, mn.Quaternion]]) -> str:
    if legacy_tuple is None:
        return None

    v = legacy_tuple[0]
    q = legacy_tuple[1]

    if v is None or q is None:
        return None

    # apply a corrective rotation to the local frame in order to align the palm
    r = mn.Quaternion.rotation(-mn.Rad(0), mn.Vector3(0, 0, 1))
    q = q * r

    R = (
        (Quaternion([q.scalar, q.vector[0], q.vector[1], q.vector[2]]))
        .unit()
        .SE3()
    )
    t = SE3([v.x, v.y, v.z])
    pose = t * R

    return pose


class QuestReader:
    def __init__(self, app_service: AppService):
        self._xr_input = app_service.remote_client_state.get_xr_input()
        self._app_service = app_service

        # Set origin to None so user has to set it manually when they start the app.
        self.origin: SE3 = None

        # Set clutches to True so the user does not immediately start streaming poses.
        self.clutch = [True, True]

    def _get_pose(self, hand: int):
        """
        Returns the pose of the hand.
        """

        state = self._app_service.remote_client_state
        return pos(state.get_hand_pose(0, hand))

    def _clutching(self, hand, handID: int):
        """
        Returns True if the hand is clutched.
        """

        # Two independent if statements since user can press both buttons at the same time.
        # We mark clutching as True if that is the case.
        if "TWO" in btn(hand._buttons_held):
            self.clutch[handID] = False
        if "ONE" in btn(hand._buttons_held):
            self.clutch[handID] = True

        return self.clutch[handID]

    def _get_left_hand_pose(self):
        """
        Returns the pose of the left hand with respect to origin.
        """
        left = self._xr_input.left_controller

        # Check for clutching
        if self._clutching(left, 0):
            return None

        return self.origin.inv() * self._get_pose(0)

    def _get_right_hand_pose(self):
        """
        Returns the pose of the right hand with respect to origin.
        """
        right = self._xr_input.right_controller

        # Check for clutching
        if self._clutching(right, 1):
            return None

        return self.origin.inv() * self._get_pose(1)

    def set_origin(self):
        """
        Sets the origin to the current pose of the left hand minus some offset.
        """

        self.origin = self._get_pose(0)
        # Add an arbitary offset to the origin so target position is always infront of it.
        self.origin.t = [
            self.origin.t[0] + 0.3,
            self.origin.t[1] - 0.3,
            self.origin.t[2] + 0.25,
        ]

    def quest_reader(self):
        """
        This function reads the state of the controllers and
        returns the poses of the controllers with respect to the current origin
        if both controllers are in-hand and no clutching is being performed."""

        if "START" in btn(self._xr_input.left_controller._buttons_held):
            self.set_origin()

        if self.origin is None:
            # Wait for origin to be set.
            return None, None

        # Get poses of each hand if the controllers are in hand and not clutched.
        poseLeft = self._get_left_hand_pose()
        poseRight = self._get_right_hand_pose()

        return poseLeft, poseRight
