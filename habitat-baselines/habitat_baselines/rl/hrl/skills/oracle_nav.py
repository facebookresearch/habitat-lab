# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from dataclasses import dataclass

import torch

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.tasks.rearrange.rearrange_sensors import LocalizationSensor
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

        self._pddl_problem = PddlProblem(
            pddl_domain_path,
            pddl_task_path,
            task_config,
        )
        self._all_entities = self._pddl_problem.get_ordered_entities_list()
        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, "oracle_nav_action"
        )

        self._is_target_obj = None
        self._targ_obj_idx = None
        self._prev_pos = [None for _ in range(self._batch_size)]

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
    ):
        self._is_target_obj = None
        self._targ_obj_idx = None
        self._prev_angle = {}

        for i in batch_idx:
            self._prev_pos[i] = None

        ret = super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
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

        cur_pos = observations[LocalizationSensor.cls_uuid].cpu()

        for i, batch_i in enumerate(batch_idx):
            prev_pos = self._prev_pos[batch_i]
            if prev_pos is not None:
                movement = (prev_pos - cur_pos[i]).pow(2).sum().sqrt()
                ret[i] = movement < self._config.stop_thresh
            self._prev_pos[batch_i] = cur_pos[i]

        return ret

    def _parse_skill_arg(self, skill_arg):
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
