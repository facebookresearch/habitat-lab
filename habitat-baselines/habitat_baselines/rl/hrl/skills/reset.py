# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import gym.spaces as spaces
import numpy as np
import torch

from habitat_baselines.rl.hrl.skills.skill import SkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class ResetArmSkill(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, True)
        self._rest_state = np.array(
            [float(x) for x in config.reset_joint_state]
        )

        self._arm_ac_range = find_action_range(action_space, "arm_action")
        self._arm_ac_range = (self._arm_ac_range[0], self._rest_state.shape[0])

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ret = super().on_enter(
            skill_arg,
            batch_idxs,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )

        self._initial_delta = (
            self._rest_state - observations["joint"].cpu().numpy()
        )

        return ret

    def _parse_skill_arg(self, skill_name: str, skill_arg: str):
        return None

    @property
    def required_obs_keys(self) -> List[str]:
        return super().required_obs_keys + ["joint"]

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ):
        current_joint_pos = observations["joint"].cpu().numpy()

        return (
            torch.as_tensor(
                np.abs(current_joint_pos - self._rest_state).max(-1),
                dtype=torch.float32,
            )
            < 5e-2
        )

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        current_joint_pos = observations["joint"].cpu().numpy()
        delta = self._rest_state - current_joint_pos

        # Dividing by max initial delta means that the action will
        # always in [-1,1] and has the benefit of reducing the delta
        # amount was we converge to the target.
        delta = delta / np.maximum(
            self._initial_delta[cur_batch_idx].max(-1, keepdims=True), 1e-5
        )

        action = torch.zeros_like(prev_actions)
        # There is an extra grab action that we don't want to set.
        action[
            ..., self._arm_ac_range[0] : self._arm_ac_range[1]
        ] = torch.from_numpy(delta).to(
            device=action.device, dtype=action.dtype
        )

        return PolicyActionData(
            actions=action, rnn_hidden_states=rnn_hidden_states
        )
