# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np
import torch

from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.rl.hrl.skills.pick import PickSkillPolicy


class PlaceSkillPolicy(PickSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pick_logic = False

    def _get_multi_sensor_index(self, batch_idx):
        return [self._cur_skill_args[i].targ for i in batch_idx]

    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_not_holding):
            # Do not regrasp the object once it is released.
            action.actions[i, self._grip_ac_idx] = -1.0
        return action

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states=None,
        prev_actions=None,
        masks=None,
        batch_idx=None,
    ) -> torch.BoolTensor:
        # Is the agent not holding an object and is the end-effector at the
        # resting position?
        is_holding = (
            observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)
        )
        current_joint_pos = observations["joint"].cpu().numpy()
        is_reset_done = (
            torch.as_tensor(
                np.abs(current_joint_pos - self._rest_state).max(-1),
                dtype=torch.float32,
            )
            < 0.05
        )
        is_done = is_reset_done & (~is_holding)
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
        action = super()._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )
        return action
