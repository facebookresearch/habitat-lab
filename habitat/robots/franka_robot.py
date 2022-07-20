# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import magnum as mn
import numpy as np

from habitat.robots.static_manipulator import (
    StaticManipulator,
    StaticManipulatorParams,
)


class FrankaRobot(StaticManipulator):
    def _get_franka_params(self) -> StaticManipulatorParams:
        return StaticManipulatorParams(
            arm_joints=[],
            gripper_joints=[],
            arm_init_params=np.array([]),
            gripper_init_params=np.array([]),
            ee_offset=mn.Vector3(),
            ee_link=1,
            ee_constraint=np.array([]),
            gripper_closed_state=np.array([]),
            gripper_open_state=np.array([]),
            gripper_state_eps=0.0,
            arm_mtr_pos_gain=0.0,
            arm_mtr_vel_gain=0.0,
            arm_mtr_max_impulse=0.0,
        )

    def __init__(
        self,
        urdf_path,
        sim,
        limit_robo_joints=True,
    ):
        super().__init__(
            self._get_franka_params(),
            urdf_path,
            sim,
            limit_robo_joints,
        )
