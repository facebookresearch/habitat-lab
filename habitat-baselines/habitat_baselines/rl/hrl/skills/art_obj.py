# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = super().on_enter(
            skill_arg,
            batch_idxs,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )
        self._did_leave_start_zone = torch.zeros(
            self._batch_size, device=prev_actions.device
        )
        self._episode_start_resting_pos = observations[
            RelativeRestingPositionSensor.cls_uuid
        ]
        return result

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        cur_resting_pos = observations[RelativeRestingPositionSensor.cls_uuid]

        did_leave_start_zone = (
            torch.linalg.vector_norm(
                cur_resting_pos - self._episode_start_resting_pos, dim=-1
            )
            > self._config.start_zone_radius
        )
        self._did_leave_start_zone = torch.logical_or(
            self._did_leave_start_zone, did_leave_start_zone
        )

        cur_resting_dist = torch.linalg.vector_norm(
            observations[RelativeRestingPositionSensor.cls_uuid], dim=-1
        )
        is_within_thresh = cur_resting_dist < self._config.at_resting_threshold
        is_holding = (
            observations[IsHoldingSensor.cls_uuid].view(-1).type(torch.bool)
        )

        is_not_holding = ~is_holding
        return is_not_holding & is_within_thresh & self._did_leave_start_zone

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        self._internal_log(f"Parsing skill argument {skill_arg}")
        return int(skill_arg[1].split("|")[1])
