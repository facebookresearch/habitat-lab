# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import gym.spaces as spaces
import torch

from habitat_baselines.rl.hrl.skills.skill import SkillPolicy


class WaitSkillPolicy(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._wait_time = -1

    def _parse_skill_arg(self, skill_arg: str) -> Any:
        self._wait_time = int(skill_arg[0])
        self._internal_log(f"Requested wait time {self._wait_time}")

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        assert self._wait_time > 0
        return self._cur_skill_step >= self._wait_time

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action = torch.zeros(prev_actions.shape, device=prev_actions.device)
        return action, rnn_hidden_states
