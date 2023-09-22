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
        self.sm = skill_io_manager()
        self._num_ac = get_num_actions(action_space)
        self._rest_state = np.array([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0])
        self._need_reset_arm = True
        self._pick_down = False

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
            < 0.05
        )
        is_reset_down = is_reset_down.to(is_holding.device)
        is_done = (is_holding * is_within_thresh * is_reset_down).type(
            torch.bool
        )
        if is_done:
            self.sm.hidden_state = None
            self._pick_down = False

        return is_done

    def _is_pick_done(
        self,
        observations,
        rnn_hidden_states=None,
        prev_actions=None,
        masks=None,
        batch_idx=None,
    ) -> torch.BoolTensor:
        # Is the agent holding the object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)

        is_done = (is_holding * is_within_thresh).type(torch.bool)
        if is_done:
            self.sm.hidden_state = None
            self._pick_down = False

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

        # Update the hidden state / action
        self.sm.hidden_state = action.rnn_hidden_states
        self.sm.prev_action = action.actions

        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_holding):
            # Do not release the object once it is held
            if self._need_reset_arm:
                current_joint_pos = observations["joint"].cpu().numpy()
                # self._compute_delta(current_joint_pos)
                delta = self._rest_state - current_joint_pos
                action.actions[
                    :, self.arm_start_id : self.arm_start_id + self.arm_len - 1
                ] = torch.from_numpy(delta).to(
                    device=action.actions.device, dtype=action.actions.dtype
                )
                self._pick_down = True
        # array([-2.6179938e+00, -3.1415927e+00, -1.0000000e+06,  0.0000000e+00, -2.7925267e+00, -1.8325957e+00, -2.8797932e+00], dtype=float32),
        # array([3.1415927e+00, 5.2359879e-01, 1.0000000e+06, 3.1415927e+00, 2.7925267e+00, 1.8325957e+00, 2.8797932e+00], dtype=float32)
        # np.array([0.0, -3.14, 0.0, 3.0, 0.0, 0.0, 0.0])

        return action
