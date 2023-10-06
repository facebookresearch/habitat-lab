# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import torch

from habitat.core.simulator import Observations
from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
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
        self.should_ignore_grip = config.ignore_grip
        self._batch_size = batch_size
        self._apply_postconds = self._config.apply_postconds
        self._force_end_on_timeout = self._config.force_end_on_timeout
        self._max_skill_steps = self._config.max_skill_steps

        self._cur_skill_step = torch.zeros(self._batch_size)
        self._should_keep_hold_state = should_keep_hold_state

        self._cur_skill_args: List[Any] = [
            None for _ in range(self._batch_size)
        ]
        self._raw_skill_args: List[Optional[str]] = [
            None for _ in range(self._batch_size)
        ]
        self._full_ac_size = get_num_actions(action_space)

        # TODO: for some reason this doesnt work with "pddl_apply_action" in action_space
        # and needs to go through the keys argument
        if "pddl_apply_action" in list(action_space.keys()):
            self._pddl_ac_start, _ = find_action_range(
                action_space, "pddl_apply_action"
            )
        else:
            self._pddl_ac_start = None
        if self._apply_postconds and self._pddl_ac_start is None:
            raise ValueError(f"Could not find PDDL action in skill {self}")

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
        if not found_grip and not self.should_ignore_grip:
            raise ValueError(f"Could not find grip action in {action_space}")

        self._stop_action_idx, _ = find_action_range(
            action_space, "rearrange_stop"
        )

    def _internal_log(self, s):
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
        self, action_data: PolicyActionData, observations
    ) -> PolicyActionData:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """
        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.

        if not self.should_ignore_grip:
            action_data.write_action(
                self._grip_ac_idx, is_holding + (is_holding - 1.0)
            )
        return action_data

    def _apply_postcond(
        self,
        actions,
        log_info,
        skill_name,
        env_i,
        idx,
    ):
        """
        Modifies the actions according to the postconditions set in self._pddl_problem.actions[skill_name]
        """
        skill_args = self._raw_skill_args[env_i]
        action = self._pddl_problem.actions[skill_name]

        entities = [self._pddl_problem.get_entity(x) for x in skill_args]
        assert (
            self._pddl_ac_start is not None
        ), "Apply post cond not supported when pddl action not in action space"

        ac_idx = self._pddl_ac_start
        found = False
        for other_action in self._action_ordering:
            if other_action.name != action.name:
                ac_idx += other_action.n_args
            else:
                found = True
                break
        if not found:
            raise ValueError(f"Could not find action {action}")

        entity_idxs = [
            self._entities_list.index(entity) + 1 for entity in entities
        ]
        if len(entity_idxs) != action.n_args:
            raise ValueError(
                f"The skill was called with the wrong # of args {action.n_args} versus {entity_idxs} for {action} with {skill_args} and {entities}. Make sure the skill and PDDL definition match."
            )

        actions[idx, ac_idx : ac_idx + action.n_args] = torch.tensor(
            entity_idxs, dtype=actions.dtype, device=actions.device
        )
        apply_action = action.clone()
        apply_action.set_param_values(entities)

        log_info[env_i]["pddl_action"] = apply_action.compact_str
        return actions[idx]

    def should_terminate(
        self,
        observations: Observations,
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        hl_wants_skill_term: torch.BoolTensor,
        batch_idx: List[int],
        skill_name: List[str],
        log_info: List[Dict[str, Any]],
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
        """
        :returns: Both of the BoolTensor's will be on the CPU.
            - `is_skill_done`: Shape (batch_size,) size tensor where 1
              indicates the skill to return control to HL policy.
            - `bad_terminate`: Shape (batch_size,) size tensor where 1
              indicates the skill should immediately end the episode.
        """
        is_skill_done = self._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, batch_idx
        ).cpu()
        assert is_skill_done.shape == (
            len(batch_idx),
        ), f"Must return tensor of shape (batch_size,) but got tensor of shape {is_skill_done.shape}"

        cur_skill_step = self._cur_skill_step[batch_idx]

        bad_terminate = torch.zeros(
            cur_skill_step.shape,
            device=cur_skill_step.device,
            dtype=torch.bool,
        )
        if self._max_skill_steps > 0:
            over_max_len = cur_skill_step >= self._max_skill_steps
            if self._force_end_on_timeout:
                bad_terminate = over_max_len
            else:
                is_skill_done = is_skill_done | over_max_len

        # Apply the postconds based on the skill termination, not if the HL policy wanted to terminate.
        new_actions = torch.zeros_like(actions)
        for i, env_i in enumerate(batch_idx):
            if self._apply_postconds and is_skill_done[i]:
                new_actions[i] = self._apply_postcond(
                    new_actions, log_info, skill_name[i], env_i, i
                )
        # Also terminate the skill if the HL policy wanted termination.
        is_skill_done |= hl_wants_skill_term

        return is_skill_done, bad_terminate, new_actions

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes in the data at the current `batch_idx`
        :returns: The new hidden state and prev_actions ONLY at the batch_idx.
        """

        self._cur_skill_step[batch_idxs] = 0
        for i, batch_idx in enumerate(batch_idxs):
            self._raw_skill_args[batch_idx] = skill_arg[i]
            if baselines_logger.level >= logging.DEBUG:
                baselines_logger.debug(
                    f"Entering skill {self} with arguments {skill_arg[i]}"
                )
            self._cur_skill_args[batch_idx] = self._parse_skill_arg(
                skill_name[i], skill_arg[i]
            )

        return (
            rnn_hidden_states[batch_idxs] * 0.0,
            prev_actions[batch_idxs] * 0,
        )

    def set_pddl_problem(self, pddl_prob):
        self._pddl_problem = pddl_prob
        self._entities_list = self._pddl_problem.get_ordered_entities_list()
        self._action_ordering = self._pddl_problem.get_ordered_actions()

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
        action_data = self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            deterministic,
        )

        if self._should_keep_hold_state:
            action_data = self._keep_holding_state(action_data, observations)
        return action_data

    def to(self, device):
        pass

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
                len(cur_batch_idx), -1, self._config.obs_skill_input_dim
            )
            obs[k] = entity_positions[
                torch.arange(len(cur_batch_idx)), cur_multi_sensor_index
            ]
        return obs

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants
            to end and 0 if not where batch_size is potentially a subset of the
            overall num_environments as specified by `batch_idx`.
        """
        return torch.zeros(masks.shape[0], dtype=torch.bool).to(masks.device)

    def _parse_skill_arg(self, skill_name: str, skill_arg: str) -> Any:
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
    ) -> PolicyActionData:
        raise NotImplementedError()

    @property
    def required_obs_keys(self) -> List[str]:
        """
        Which keys from the observation dictionary this skill requires to
        compute actions and termination conditions.
        """
        if self._should_keep_hold_state:
            return [IsHoldingSensor.cls_uuid]
        else:
            return []
