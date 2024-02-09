# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat.articulated_agents.static_manipulator import (
    StaticManipulator,
    StaticManipulatorParams,
)


class FrankaRobot(StaticManipulator):
    def _get_franka_params(self) -> StaticManipulatorParams:
        return StaticManipulatorParams(
            arm_joints=list(range(0, 7)),
            gripper_joints=[],
            arm_init_params=np.zeros((7,)),
            gripper_init_params=np.zeros((2,)),
            ee_offset=[mn.Vector3()],  # zeroed
            ee_links=[1],
            ee_constraint=np.array([[[0.4, 1.2], [-0.7, 0.7], [0.25, 1.5]]]),
            gripper_closed_state=np.array(
                [
                    0.0,
                    0.0,
                ]
            ),
            gripper_open_state=np.array(
                [
                    0.04,
                    0.04,
                ]
            ),
            gripper_state_eps=0.001,
            arm_mtr_pos_gain=0.3,
            arm_mtr_vel_gain=0.3,
            arm_mtr_max_impulse=10.0,
        )

    def __init__(self, *args, **kwargs):
        kwargs["params"] = self._get_franka_params()
        super().__init__(*args, **kwargs)
