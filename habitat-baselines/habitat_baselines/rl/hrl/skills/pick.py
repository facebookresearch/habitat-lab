# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range, skill_io_manager
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions


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
            if k == "arm_action":
                self.arm_start_id, self.arm_len = find_action_range(
                    action_space, "arm_action"
                )

        # Get the skill io manager
        config = args[1]
        self.sm = skill_io_manager()
        self._num_ac = get_num_actions(action_space)
        self._rest_state = np.array(config.reset_joint_state)
        self._need_reset_arm = True
        self._dist_to_rest_state = 0.05

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

        current_joint_pos = observations["joint"].cpu().numpy()
        is_reset_down = (
            torch.as_tensor(
                np.abs(current_joint_pos - self._rest_state).max(-1),
                dtype=torch.float32,
            )
            < self._dist_to_rest_state
        )
        is_reset_down = is_reset_down.to(is_holding.device)
        is_done = (is_holding * is_within_thresh * is_reset_down).type(
            torch.bool
        )
        if is_done.sum() > 0:
            self.sm.hidden_state[is_done] *= 0
            self.sm._prev_action[is_done] *= 0

        return is_done

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
        if self.sm.hidden_state is None:
            self.sm.init_hidden_state(
                observations,
                self._wrap_policy.net._hidden_size,
                self._wrap_policy.num_recurrent_layers,
            )
            self.sm.init_prev_action(prev_actions, self._num_ac)

        action = super()._internal_act(
            observations,
            self.sm.hidden_state,
            self.sm.prev_action,
            masks,
            cur_batch_idx,
            deterministic,
        )

        # The skill outputs a base velocity, but we want to
        # set it to pick_base_velocity. This is because we want
        # to potentially have different lin_speeds
        # For those velocities
        base_vel_index = slice(
            self.org_pick_start_id, self.org_pick_start_id + self.org_pick_len
        )
        ac_vel_index = slice(
            self.pick_start_id, self.pick_start_id + self.pick_len
        )
        arm_slice = slice(
            self.arm_start_id, self.arm_start_id + self.arm_len - 1
        )

        action.actions[:, ac_vel_index] = action.actions[:, base_vel_index]
        size = action.actions[:, base_vel_index].shape
        action.actions[:, base_vel_index] = torch.zeros(size)
        action = self._mask_pick(action, observations)

        # Update the hidden state / action
        self.sm.hidden_state = action.rnn_hidden_states
        self.sm.prev_action = action.actions

        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        # Do not release the object once it is held
        if self._need_reset_arm:
            rest_state = torch.from_numpy(self._rest_state).to(
                device=action.actions.device, dtype=action.actions.dtype
            )
            current_joint_pos = observations["joint"]
            delta = rest_state - current_joint_pos
            action.actions[torch.nonzero(is_holding), arm_slice] = delta
            # for i in torch.nonzero(is_holding):
            #     current_joint_pos = observations["joint"].cpu().numpy()
            #     delta = rest_state - current_joint_pos
            #     action.actions[:, arm_slice] = delta

        return action
