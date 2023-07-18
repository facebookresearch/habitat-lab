# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from habitat.articulated_agents.robots import FetchRobot


class FetchSuctionRobot(FetchRobot):
    def _get_fetch_params(self):
        params = super()._get_fetch_params()
        params.gripper_init_params = None
        params.gripper_closed_state = np.array([0.0], dtype=np.float32)
        params.gripper_open_state = np.array([0.0], dtype=np.float32)
        params.gripper_joints = [23]

        params.ee_links = [23]
        return params
