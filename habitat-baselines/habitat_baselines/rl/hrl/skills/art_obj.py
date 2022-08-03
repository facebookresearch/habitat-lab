from typing import List, Tuple

import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    IsHoldingSensor,
    RelativeRestingPositionSensor,
)
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy


class ArtObjSkillPolicy(NnSkillPolicy):
    def on_enter(
        self,
        skill_arg: List[str],
        batch_idx: int,
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
        )
        self._did_leave_start_zone = torch.zeros(
            self._batch_size, device=prev_actions.device
        )
        self._episode_start_resting_pos = observations[
            RelativeRestingPositionSensor.cls_uuid
        ]

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.BoolTensor:

        cur_resting_pos = observations[RelativeRestingPositionSensor.cls_uuid]

        did_leave_start_zone = (
            torch.norm(
                cur_resting_pos - self._episode_start_resting_pos, dim=-1
            )
            > self._config.START_ZONE_RADIUS
        )
        self._did_leave_start_zone = torch.logical_or(
            self._did_leave_start_zone, did_leave_start_zone
        )

        cur_resting_dist = torch.norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = cur_resting_dist < self._config.AT_RESTING_THRESHOLD
        is_holding = (
            observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)
        )

        is_not_holding = ~is_holding
        return is_not_holding & is_within_thresh & self._did_leave_start_zone

    def _parse_skill_arg(self, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[-1].split("|")[1])
