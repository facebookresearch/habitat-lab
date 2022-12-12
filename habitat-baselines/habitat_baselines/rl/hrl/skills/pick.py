# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.common.logging import baselines_logger
from habitat.core.spaces import ActionSpace
from habitat_baselines.rl.hrl.utils import find_action_range


class PickSkillPolicy(NnSkillPolicy):
    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        # Is the agent holding the object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        return (is_holding * is_within_thresh).type(torch.bool)

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[0].split("|")[1])

    def _mask_pick(self, action, observations):
        # Mask out the release if the object is already held.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            action[i, self._grip_ac_idx] = 1.0
        return action

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action, hxs = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )
        action = self._mask_pick(action, observations)
        return action, hxs


class HumanPickSkillPolicy(NnSkillPolicy):
    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
            ignore_grip=True
        )
        self._pick_ac_idx, _ = find_action_range(
            action_space, "humanpick_action"
        )
        self._hand_ac_idx = self._pick_ac_idx + 1

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        return is_holding.type(torch.bool)

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        filtered_action_space = ActionSpace(
            {config.PICK_ACTION_NAME: action_space[config.PICK_ACTION_NAME]}
        )

        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size
            
        )
    
    def _mask_pick(self, action, observations):
        # Mask out the release if the object is already held.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            action[i, self._hand_ac_idx] = 1.0
        return action

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[0].split("|")[1])

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        action = torch.zeros(prev_actions.shape, device=masks.device)
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i] for i in cur_batch_idx]
        )
        action[:, self._pick_ac_idx] = action_idxs
        
        action = self._mask_pick(action, observations)

        action[:, self._hand_ac_idx] = 1.0
        return action, rnn_hidden_states
    