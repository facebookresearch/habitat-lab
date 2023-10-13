# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from dataclasses import dataclass

import torch

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import (
    HasFinishedOracleNavSensor,
    IsHoldingSensor,
)
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData


class OracleNavPolicy(NnSkillPolicy):
    @dataclass
    class OracleNavActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        """

        action_idx: int

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
        task_config,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )

        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, "oracle_nav_action"
        )

    def set_pddl_problem(self, pddl_prob):
        super().set_pddl_problem(pddl_prob)
        self._all_entities = self._pddl_problem.get_ordered_entities_list()

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
        skill_name,
    ):
        ret = super().on_enter(
            skill_arg,
            batch_idx,
            observations,
            rnn_hidden_states,
            prev_actions,
            skill_name,
        )
        self._was_running_on_prev_step = False
        return ret

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        filtered_action_space = ActionSpace(
            {config.action_name: action_space[config.action_name]}
        )
        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size,
            full_config.habitat.task.pddl_domain_def,
            osp.join(
                full_config.habitat.task.task_spec_base_path,
                full_config.habitat.task.task_spec + ".yaml",
            ),
            full_config.habitat.task,
        )

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        ret = torch.zeros(masks.shape[0], dtype=torch.bool)

        finish_oracle_nav = observations[
            HasFinishedOracleNavSensor.cls_uuid
        ].cpu()
        ret = finish_oracle_nav.to(torch.bool)[:, 0]

        return ret

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        if len(skill_arg) == 2:
            search_target, _ = skill_arg
        elif len(skill_arg) == 3:
            _, search_target, _ = skill_arg
        else:
            raise ValueError(
                f"Unexpected number of skill arguments in {skill_arg}"
            )

        target = self._pddl_problem.get_entity(search_target)
        if target is None:
            raise ValueError(
                f"Cannot find matching entity for {search_target}"
            )
        match_i = self._all_entities.index(target)

        return OracleNavPolicy.OracleNavActionArgs(match_i)

    @property
    def required_obs_keys(self):
        ret = [HasFinishedOracleNavSensor.cls_uuid]
        if self._should_keep_hold_state:
            ret.append(IsHoldingSensor.cls_uuid)
        return ret

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx + 1 for i in cur_batch_idx]
        )

        full_action[:, self._oracle_nav_ac_idx] = action_idxs

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )


class OracleNavCoordPolicy(OracleNavPolicy):
    """The function produces a sequence of navigation targets and the oracle nav navigates to those targets"""

    @dataclass
    class OracleNavActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        """

        action_idx: int

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
        task_config,
    ):
        NnSkillPolicy.__init__(
            self,
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        # Random coordinate means that the navigation target is chosen randomly
        action_name = "oracle_nav_randcoord_action"
        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, action_name
        )

    def _parse_skill_arg(self, skill_arg):
        return OracleNavCoordPolicy.OracleNavActionArgs(skill_arg)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx for i in cur_batch_idx]
        )

        full_action[:, self._oracle_nav_ac_idx] = action_idxs

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )


class OracleNavHumanPolicy(OracleNavCoordPolicy):
    """
    Navigate to human's location using oracle nav
    """

    @dataclass
    class OracleNavActionArgs:
        """
        :property action_idx: The index of the oracle action we want to execute
        """

        action_idx: int

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
        task_config,
    ):
        NnSkillPolicy.__init__(
            self,
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )
        action_name = "oracle_nav_human_action"
        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, action_name
        )

    def _parse_skill_arg(self, skill_arg):
        return OracleNavHumanPolicy.OracleNavActionArgs(skill_arg)
