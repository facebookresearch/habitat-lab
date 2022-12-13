# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.pick import PickSkillPolicy, HumanPickSkillPolicy

from habitat_baselines.rl.hrl.utils import find_action_range, find_action_range_pddl

class PlaceSkillPolicy(PickSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def _get_multi_sensor_index(self, batch_idx):
        return [self._cur_skill_args[i].targ for i in batch_idx]

    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_not_holding):
            # Do not regrasp the object once it is released.
            action[i, self._grip_ac_idx] = -1.0
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
            self._internal_log(
                f"Terminating with {rel_resting_pos} and {is_holding}",
                observations,
            )
        return is_done

    def _parse_skill_arg(self, skill_arg):
        obj = int(skill_arg[0].split("|")[1])
        targ = int(skill_arg[1].split("|")[1])
        return PlaceSkillPolicy.PlaceSkillArgs(obj=obj, targ=targ)


class HumanPlaceSkillPolicy(HumanPickSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int


    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config

    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
            pddl_domain_path,
            pddl_task_path,
            task_config)

        self.pddl_action_idx = find_action_range_pddl(
            self._pddl_problem.get_ordered_actions(), "place"
        )

    def _get_multi_sensor_index(self, batch_idx):
        return [self._cur_skill_args[i].targ for i in batch_idx]


    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_not_holding):
            # Do not regrasp the object once it is released.
            action[i, self._grip_ac_idx] = 0.0
            action[i, self._desnap_ac_idx] = 1.0
        return action

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        is_holding = (
                observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)
            )
        return ~is_holding

    def _parse_skill_arg(self, skill_arg):
        obj = int(skill_arg[0].split("|")[1])
        targ = int(skill_arg[1].split("|")[1])
        return HumanPlaceSkillPolicy.PlaceSkillArgs(obj=obj, targ=targ)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # breakpoint()
        action = torch.zeros(prev_actions.shape, device=masks.device)
        # action_idxs = torch.FloatTensor(
        #     [self._cur_skill_args[i] for i in cur_batch_idx]
        # )
        # action[:, self._pick_ac_idx] = action_idxs

        # action = self._mask_pick(action, observations)
        # action[:, self._hand_ac_idx] = 0.0


        action[:, self._pick_ac_idx+self.pddl_action_idx[0]] = 7
        action[:, self._pick_ac_idx+self.pddl_action_idx[0]+1] = 1
        action[:, self._pick_ac_idx+self.pddl_action_idx[0]+2] = 8

        return action, rnn_hidden_states
