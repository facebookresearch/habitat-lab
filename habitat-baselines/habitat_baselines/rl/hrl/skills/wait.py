# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import gym.spaces as spaces
import torch

from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData


class WaitSkillPolicy(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._wait_time = -1

    def _parse_skill_arg(self, skill_name: str, skill_arg: str) -> Any:
        self._wait_time = int(skill_arg[0])
        self._internal_log(f"Requested wait time {self._wait_time}")

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        assert self._wait_time > 0
        return (self._cur_skill_step >= self._wait_time)[batch_idx]

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=prev_actions.device
        )
        return PolicyActionData(
            actions=action, rnn_hidden_states=rnn_hidden_states
        )
