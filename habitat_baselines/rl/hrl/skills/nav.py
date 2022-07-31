from dataclasses import dataclass

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    TargetGoalGpsCompassSensor,
    TargetStartGpsCompassSensor,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    TargetOrGoalStartPointGoalSensor,
)
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy


class NavSkillPolicy(NnSkillPolicy):
    @dataclass(frozen=True)
    class NavArgs:
        obj_idx: int
        is_target: bool

    def __init__(
        self,
        wrap_policy,
        config,
        action_space: spaces.Space,
        filtered_obs_space: spaces.Space,
        filtered_action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
            should_keep_hold_state=True,
        )

    def _get_filtered_obs(self, observations, cur_batch_idx) -> TensorDict:
        ret_obs = super()._get_filtered_obs(observations, cur_batch_idx)

        if TargetOrGoalStartPointGoalSensor.cls_uuid in ret_obs:
            if self._cur_skill_args[cur_batch_idx].is_target:
                replace_sensor = TargetGoalGpsCompassSensor.cls_uuid
            else:
                replace_sensor = TargetStartGpsCompassSensor.cls_uuid
            ret_obs[TargetOrGoalStartPointGoalSensor.cls_uuid] = observations[
                replace_sensor
            ]
        return ret_obs

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        return self._cur_skill_args[batch_idx].obj_idx

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> torch.BoolTensor:
        filtered_prev_actions = prev_actions[
            :, self._ac_start : self._ac_start + self._ac_len
        ]

        lin_vel, ang_vel = (
            filtered_prev_actions[:, 0],
            filtered_prev_actions[:, 1],
        )
        should_stop = (
            torch.abs(lin_vel) < self._config.LIN_SPEED_STOP
            and torch.abs(ang_vel) < self._config.ANG_SPEED_STOP
        )
        return should_stop

    def _parse_skill_arg(self, skill_arg):
        targ_name, targ_idx = skill_arg[-1].split("|")
        return NavSkillPolicy.NavArgs(
            obj_idx=int(targ_idx), is_target=targ_name.startswith("TARGET")
        )
