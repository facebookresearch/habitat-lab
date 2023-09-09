# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.pick import PickSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range, skill_io_manager
from habitat_baselines.utils.common import get_num_actions


class PlaceSkillPolicy(PickSkillPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get the action space
        action_space = args[2]
        for k, _ in action_space.items():
            if k == "pick_base_velocity":
                self.pick_start_id, self.pick_len = find_action_range(
                    action_space, "pick_base_velocity"
                )
            if k == "place_base_velocity":
                self.place_start_id, self.place_len = find_action_range(
                    action_space, "place_base_velocity"
                )

        # Get the skill io manager
        self.sm = skill_io_manager()
        self._need_reset_arm = False

        self._num_ac = get_num_actions(action_space)

    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def _get_multi_sensor_index(self, batch_idx):
        return [self._cur_skill_args[i].targ for i in batch_idx]

    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        if torch.nonzero(is_not_holding).sum() > 0:
            # Do not regrasp the object once it is released.
            action.actions[
                torch.nonzero(is_not_holding), self._grip_ac_idx
            ] = -1.0
        return action

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        # Is the agent not holding an object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.at_resting_threshold
        is_holding = (
            observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)
        )
        is_done = is_within_thresh & (~is_holding)
        if is_done.sum() > 0:
            # self._internal_log(
            #     f"Terminating with {rel_resting_pos} and {is_holding}",
            # )
            self.sm.hidden_state[is_done] *= 0
        return is_done

    def _parse_skill_arg(self, skill_arg):
        obj = int(skill_arg[0].split("|")[1])
        targ = int(skill_arg[1].split("|")[1])
        return PlaceSkillPolicy.PlaceSkillArgs(obj=obj, targ=targ)

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
        action = self._mask_pick(action, observations)

        # The skill outputs a base velocity, but we want to
        # set it to pick_base_velocity. This is because we want
        # to potentially have different lin_speeds
        # For those velocities
        place_index = slice(
            self.place_start_id, self.place_start_id + self.place_len
        )
        pick_index = slice(
            self.pick_start_id, self.pick_start_id + self.pick_len
        )

        action.actions[:, place_index] = action.actions[:, pick_index]
        size = action.actions[:, pick_index].shape
        action.actions[:, pick_index] = torch.zeros(size)
        # Update the hidden state / action
        self.sm.hidden_state = action.rnn_hidden_states
        self.sm.prev_action = action.actions

        return action
