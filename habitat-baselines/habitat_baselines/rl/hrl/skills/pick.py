# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class PickSkillPolicy(NnSkillPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get the action space
        action_space = args[2]
        for k, _ in action_space.items():
            if k == "pick_base_velocity":
                self.pick_start_id, self.pick_len = find_action_range(
                    action_space, "pick_base_velocity"
                )
            if k == "base_velocity":
                self.org_pick_start_id, self.org_pick_len = find_action_range(
                    action_space, "base_velocity"
                )

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

    def _mask_pick(
        self, action: PolicyActionData, observations
    ) -> PolicyActionData:
        # Mask out the release if the object is already held.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            action.actions[i, self._grip_ac_idx] = 1.0
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
        action = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        action.actions[
            :, self.pick_start_id : self.pick_start_id + self.pick_len
        ] = action.actions[
            :,
            self.org_pick_start_id : self.org_pick_start_id
            + self.org_pick_len,
        ]
        size = action.actions[
            :,
            self.org_pick_start_id : self.org_pick_start_id
            + self.org_pick_len,
        ].shape
        action.actions[
            :,
            self.org_pick_start_id : self.org_pick_start_id
            + self.org_pick_len,
        ] = torch.zeros(size)

        action = self._mask_pick(action, observations)

        print("pick!!!")

        return action
