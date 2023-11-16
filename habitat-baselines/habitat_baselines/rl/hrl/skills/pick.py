# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class PickSkillPolicy(NnSkillPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get the action space
        action_space = args[2]
        for k, _ in action_space.items():
            if k == "arm_action":
                self.arm_start_id, self.arm_len = find_action_range(
                    action_space, "arm_action"
                )
        # Parameters for resetting the arm
        self._rest_state = np.array([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0])
        self._need_reset_arm = True
        self._arm_retract_success_threshold = 0.05

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states=None,
        prev_actions=None,
        masks=None,
        batch_idx=None,
    ) -> torch.BoolTensor:
        # Is the agent holding the object and is the end-effector at the
        # resting position?
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        current_joint_pos = observations["joint"].cpu().numpy()
        is_reset_done = (
            torch.as_tensor(
                np.abs(current_joint_pos - self._rest_state).max(-1),
                dtype=torch.float32,
            )
            < self._arm_retract_success_threshold
        )
        is_reset_done = is_reset_done.to(is_holding.device)

        return (is_holding * is_reset_done).type(torch.bool)

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[0].split("|")[1])

    def _retract_arm_action(self, observations, action):
        """Retract the arm"""
        current_joint_pos = observations["joint"].cpu().numpy()
        delta = self._rest_state - current_joint_pos
        action.actions[
            :, self.arm_start_id : self.arm_start_id + self.arm_len - 1
        ] = torch.from_numpy(delta).to(
            device=action.actions.device, dtype=action.actions.dtype
        )
        return action

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

        action = self._mask_pick(action, observations)
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        if is_holding and self._need_reset_arm:
            action = self._retract_arm_action(observations, action)

        return action
