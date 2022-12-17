# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions


class SkillPolicy(Policy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        """
        :param action_space: The overall action space of the entire task, not task specific.
        """
        self._config = config
        self._batch_size = batch_size

        self._cur_skill_step = torch.zeros(self._batch_size)
        self._should_keep_hold_state = should_keep_hold_state

        self._cur_skill_args: List[Any] = [
            None for _ in range(self._batch_size)
        ]
        self._raw_skill_args: List[Optional[str]] = [
            None for _ in range(self._batch_size)
        ]

        self._grip_ac_idx = 0
        found_grip = False
        for k, space in action_space.items():
            if k != "arm_action":
                self._grip_ac_idx += get_num_actions(space)
            else:
                # The last actioin in the arm action is the grip action.
                self._grip_ac_idx += get_num_actions(space) - 1
                found_grip = True
                break
        if not found_grip:
            raise ValueError(f"Could not find grip action in {action_space}")
        self._stop_action_idx, _ = find_action_range(
            action_space, "rearrange_stop"
        )

    def _internal_log(self, s, observations=None):
        baselines_logger.debug(
            f"Skill {self._config.skill_name} @ step {self._cur_skill_step}: {s}"
        )

    def _get_multi_sensor_index(self, batch_idx: List[int]) -> List[int]:
        """
        Gets the index to select the observation object index in `_select_obs`.
        Used when there are multiple possible goals in the scene, such as
        multiple objects to possibly rearrange.
        """
        return [self._cur_skill_args[i] for i in batch_idx]

    def _keep_holding_state(
        self, full_action: torch.Tensor, observations
    ) -> torch.Tensor:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """
        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.
        full_action[:, self._grip_ac_idx] = is_holding + (is_holding - 1.0)
        return full_action

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        is_skill_done = self._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, batch_idx
        )
        if is_skill_done.sum() > 0:
            self._internal_log(
                f"Requested skill termination {is_skill_done}",
                observations,
            )

        bad_terminate = torch.zeros(
            self._cur_skill_step.shape,
            device=self._cur_skill_step.device,
            dtype=torch.bool,
        )
        if self._config.max_skill_steps > 0:
            over_max_len = self._cur_skill_step > self._config.max_skill_steps
            if self._config.force_end_on_timeout:
                bad_terminate = over_max_len
            else:
                is_skill_done = is_skill_done | over_max_len

        if bad_terminate.sum() > 0:
            self._internal_log(
                f"Bad terminating due to timeout {self._cur_skill_step}, {bad_terminate}",
                observations,
            )

        return is_skill_done, bad_terminate

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes in the data at the current `batch_idx`
        :returns: The new hidden state and prev_actions ONLY at the batch_idx.
        """

        self._cur_skill_step[batch_idxs] = 0
        for i, batch_idx in enumerate(batch_idxs):
            self._raw_skill_args[batch_idx] = skill_arg[i]
            self._cur_skill_args[batch_idx] = self._parse_skill_arg(
                skill_arg[i]
            )

        return (
            rnn_hidden_states[batch_idxs] * 0.0,
            prev_actions[batch_idxs] * 0.0,
        )

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        return cls(config, action_space, batch_size)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        """
        :returns: Predicted action and next rnn hidden state.
        """
        self._cur_skill_step[cur_batch_idx] += 1
        action, hxs = self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        if self._should_keep_hold_state:
            action = self._keep_holding_state(action, observations)
        return action, hxs

    def to(self, device):
        self._cur_skill_step = self._cur_skill_step.to(device)

    def _select_obs(self, obs, cur_batch_idx):
        """
        Selects out the part of the observation that corresponds to the current goal of the skill.
        """
        for k in self._config.obs_skill_inputs:
            cur_multi_sensor_index = self._get_multi_sensor_index(
                cur_batch_idx
            )
            if k not in obs:
                raise ValueError(
                    f"Skill {self._config.skill_name}: Could not find {k} out of {obs.keys()}"
                )

            entity_positions = obs[k].view(
                len(cur_batch_idx),
                -1,
                self._config.get("obs_skill_input_dim", 3),
            )
            obs[k] = entity_positions[
                torch.arange(len(cur_batch_idx)), cur_multi_sensor_index
            ]
        return obs

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        return torch.zeros(observations.shape[0], dtype=torch.bool).to(
            masks.device
        )

    def _parse_skill_arg(self, skill_arg: str) -> Any:
        """
        Parses the skill argument string identifier and returns parsed skill argument information.
        """
        return skill_arg

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
