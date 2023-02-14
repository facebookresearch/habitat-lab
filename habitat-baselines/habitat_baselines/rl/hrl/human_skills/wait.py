# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import gym.spaces as spaces
import torch

from habitat_baselines.rl.hrl.skills.wait import WaitSkillPolicy


class HumanWaitSkillPolicy(WaitSkillPolicy):
    def __init__(
        self, config, action_space: spaces.Space, batch_size, ignore_grip=True
    ):
        super().__init__(config, action_space, batch_size, ignore_grip)
