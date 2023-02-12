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



class ResetArmHumanSkill(SkillPolicy):
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

        self._arm_ac_range = find_action_range(action_space, "human_pick_action")
        self._arm_ac_range = (self._arm_ac_range[0], self._rest_state.shape[0])

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ret = super().on_enter(
            skill_arg,
            batch_idxs,
            observations,
            rnn_hidden_states,
            prev_actions,
        )

        self._initial_delta = (
            self._rest_state - observations["joint"].cpu().numpy()
        )

        return ret

    def _parse_skill_arg(self, skill_arg: str):
        return None

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
        
        filtered_obs = self._get_filtered_obs(observations, cur_batch_idx)
        filtered_obs = self._select_obs(filtered_obs, cur_batch_idx)

        breakpoint()

        full_action = torch.zeros(prev_actions.shape, device=masks.device)
        # There is an extra grab action that we don't want to set.
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx + 1 for i in cur_batch_idx]
        )
        full_action[:, self._oracle_nav_ac_idx] = action_idxs
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
