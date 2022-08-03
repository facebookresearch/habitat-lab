from dataclasses import dataclass

import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.pick import PickSkillPolicy


class PlaceSkillPolicy(PickSkillPolicy):
    @dataclass(frozen=True)
    class PlaceSkillArgs:
        obj: int
        targ: int

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        return self._cur_skill_args[batch_idx].targ

    def _mask_pick(self, action, observations):
        # Mask out the grasp if the object is already released.
        is_not_holding = 1 - observations[IsHoldingSensor.cls_uuid].view(-1)
        for i in torch.nonzero(is_not_holding):
            # Do not regrasp the object once it is released.
            action[i, self._ac_start + self._ac_len - 1] = -1.0
        return action

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.BoolTensor:
        # Is the agent not holding an object and is the end-effector at the
        # resting position?
        rel_resting_pos = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = rel_resting_pos < self._config.AT_RESTING_THRESHOLD
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
