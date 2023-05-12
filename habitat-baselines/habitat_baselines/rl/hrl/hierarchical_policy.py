# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSuccess,
)
from habitat.tasks.rearrange.multi_task.pddl_domain import (
    PddlDomain,
    PddlProblem,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.hrl.hl import (  # noqa: F401.
    FixedHighLevelPolicy,
    HighLevelPolicy,
    NeuralHighLevelPolicy,
    PlannerHighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    NoopSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class HierarchicalPolicy(nn.Module, Policy):
    """
    :property _pddl: Stores the PDDL domain information. This allows
        accessing all the possible entities, actions, and predicates. Note that
        this is not the grounded PDDL problem with truth values assigned to the
        predicates basedon the current simulator state.
    """

    _pddl: PddlDomain

    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        num_envs: int,
        aux_loss_config=None,
        agent_name: Optional[str] = None,
    ):
        super().__init__()

        self._action_space = action_space
        self._num_envs: int = num_envs

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}
        # Can map multiple skills to the same underlying skill controller.
        self._skill_redirects: Dict[int, int] = {}

        self._pddl = self._create_pddl(full_config, config)
        self._create_skills(
            dict(config.hierarchical_policy.defined_skills),
            observation_space,
            action_space,
            full_config,
        )

        self._cur_skills: np.ndarray = np.full(
            (self._num_envs,), -1, dtype=np.int32
        )
        # Init with True so we always call the HL policy during the first step
        # it runs.
        self._cur_call_high_level: torch.BoolTensor = torch.ones(
            (self._num_envs,), dtype=torch.bool
        )

        high_level_cls = self._get_hl_policy_cls(config)
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config=config.hierarchical_policy.high_level_policy,
            pddl_problem=self._pddl,
            num_envs=num_envs,
            skill_name_to_idx=self._name_to_idx,
            observation_space=observation_space,
            action_space=action_space,
            aux_loss_config=aux_loss_config,
            agent_name=agent_name,
        )
        self._stop_action_idx, _ = find_action_range(
            action_space, "rearrange_stop"
        )

    def _create_skills(
        self, skills, observation_space, action_space, full_config
    ):
        skill_i = 0
        for (
            skill_name,
            skill_config,
        ) in skills.items():
            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config,
                observation_space,
                action_space,
                self._num_envs,
                full_config,
            )
            skill_policy.set_pddl_problem(self._pddl)
            if skill_config.pddl_action_names is None:
                action_names = [skill_name]
            else:
                action_names = skill_config.pddl_action_names
            for skill_id in action_names:
                self._name_to_idx[skill_id] = skill_i
                self._idx_to_name[skill_i] = skill_id
                self._skills[skill_i] = skill_policy
                skill_i += 1

        first_idx: Optional[int] = None
        for skill_i, skill in self._skills.items():
            if self._idx_to_name[skill_i] == "noop":
                continue
            if isinstance(skill, NoopSkillPolicy):
                if first_idx is None:
                    first_idx = skill_i
                else:
                    self._skill_redirects[skill_i] = first_idx

    def _get_hl_policy_cls(self, config):
        return eval(config.hierarchical_policy.high_level_policy.name)

    def _create_pddl(self, full_config, config) -> PddlDomain:
        """
        Creates the PDDL domain from the config.
        """
        task_spec_file = osp.join(
            full_config.habitat.task.task_spec_base_path,
            full_config.habitat.task.task_spec + ".yaml",
        )
        domain_file = full_config.habitat.task.pddl_domain_def

        return PddlProblem(
            domain_file,
            task_spec_file,
            config,
        )

    def eval(self):
        pass

    def get_policy_action_space(
        self, env_action_space: spaces.Space
    ) -> spaces.Space:
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """

        return self._high_level_policy.get_policy_action_space(
            env_action_space
        )

    def extract_policy_info(
        self, action_data, infos, dones
    ) -> List[Dict[str, float]]:
        ret_policy_infos = []
        for i, (info, policy_info) in enumerate(
            zip(infos, action_data.policy_info)
        ):
            cur_skill_idx = self._cur_skills[i]
            ret_policy_info: Dict[str, Any] = {
                "cur_skill": self._idx_to_name[cur_skill_idx],
                **policy_info,
            }

            did_skill_fail = dones[i] and not info[CompositeSuccess.cls_uuid]
            for skill_name, idx in self._name_to_idx.items():
                ret_policy_info[f"failed_skill_{skill_name}"] = (
                    did_skill_fail if idx == cur_skill_idx else 0.0
                )
            ret_policy_infos.append(ret_policy_info)

        return ret_policy_infos

    @property
    def num_recurrent_layers(self):
        if self._high_level_policy.num_recurrent_layers != 0:
            return self._high_level_policy.num_recurrent_layers
        else:
            return self._skills[
                list(self._skills.keys())[0]
            ].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return self._high_level_policy.should_load_agent_state

    def parameters(self):
        return self._high_level_policy.parameters()

    def to(self, device):
        self._high_level_policy.to(device)
        for skill in self._skills.values():
            skill.to(device)

    def _broadcast_skill_ids(
        self,
        skill_ids: torch.Tensor,
        sel_dat: Dict[str, Any],
        should_adds: Optional[torch.Tensor] = None,
    ) -> Dict[int, Tuple[List[int], Dict[str, Any]]]:
        """
        Groups the information per skill. Specifically, this will return a map
        from the skill ID to the indices of the batch and the observations at
        these indices the skill is currently running for. This is used to batch
        observations per skill.

        If an entry in `sel_dat` is `None`, then it is including in all groups.
        """

        skill_to_batch: Dict[int, List[int]] = defaultdict(list)
        if should_adds is None:
            should_adds = [True for _ in range(len(skill_ids))]
        for i, (cur_skill, should_add) in enumerate(
            zip(skill_ids, should_adds)
        ):
            if not should_add:
                continue
            if cur_skill in self._skill_redirects:
                cur_skill = self._skill_redirects[cur_skill]
            skill_to_batch[cur_skill].append(i)
        grouped_skills = {}
        for k, v in skill_to_batch.items():
            skill_dat = {}
            for dat_k, dat in sel_dat.items():
                if dat_k == "observations":
                    # Reduce the slicing required by only extracting what the
                    # skills will actually need.
                    dat = dat.slice_keys(*self._skills[k].required_obs_keys)
                skill_dat[dat_k] = dat[v]
            grouped_skills[k] = (v, skill_dat)
        return grouped_skills

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ):
        masks_cpu = masks.cpu()
        log_info: List[Dict[str, Any]] = [{} for _ in range(self._num_envs)]
        self._high_level_policy.apply_mask(masks_cpu)  # type: ignore[attr-defined]

        # Initialize empty action set based on the overall action space.
        actions = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)),
            device=masks.device,
        )

        # Always call high-level if the episode is over.
        self._cur_call_high_level |= (~masks_cpu).view(-1)

        hl_terminate_episode, hl_info = self._update_skills(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            actions,
            log_info,
            self._cur_call_high_level,
            deterministic,
        )
        did_choose_new_skill = self._cur_call_high_level.clone()

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
            },
        )
        for skill_id, (batch_ids, batch_dat) in grouped_skills.items():
            action_data = self._skills[skill_id].act(
                observations=batch_dat["observations"],
                rnn_hidden_states=batch_dat["rnn_hidden_states"],
                prev_actions=batch_dat["prev_actions"],
                masks=batch_dat["masks"],
                cur_batch_idx=batch_ids,
            )

            actions[batch_ids] += action_data.actions
            rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states

        # Skills should not be responsible for terminating the overall episode.
        actions[:, self._stop_action_idx] = 0.0

        (
            self._cur_call_high_level,
            bad_should_terminate,
            actions,
        ) = self._get_terminations(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            actions,
            log_info,
        )

        should_terminate_episode = bad_should_terminate | hl_terminate_episode
        if should_terminate_episode.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate_episode):
                actions[batch_idx, self._stop_action_idx] = 1.0

        action_kwargs = {
            "rnn_hidden_states": rnn_hidden_states,
            "actions": actions,
        }
        action_kwargs.update(hl_info)

        return PolicyActionData(
            take_actions=actions,
            policy_info=log_info,
            should_inserts=did_choose_new_skill.view(-1, 1),
            **action_kwargs,
        )

    def _update_skills(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        actions,
        log_info,
        should_choose_new_skill: torch.BoolTensor,
        deterministic: bool,
    ) -> Tuple[torch.BoolTensor, Dict[str, Any]]:
        """
        Will potentially update the set of running skills according to the HL
        policy. This updates the active skill indices in `self._cur_skills` in
        place. The HL policy may also want to terminate the entire episode and
        return additional logging information (such as which skills are
        selected).

        :returns: A tuple containing the following in order
        - A tensor of size (batch_size,) indicating whether the episode should
          terminate.
        - Logging metrics from the HL policy.
        """

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate_episode = torch.zeros(self._num_envs, dtype=torch.bool)
        hl_info: Dict[str, Any] = self._high_level_policy.create_hl_info()
        if should_choose_new_skill.sum() > 0:
            (
                new_skills,
                new_skill_args,
                hl_terminate_episode,
                hl_info,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                should_choose_new_skill,
                deterministic,
                log_info,
            )
            new_skills = new_skills.numpy()

            sel_grouped_skills = self._broadcast_skill_ids(
                new_skills,
                sel_dat={},
                should_adds=should_choose_new_skill,
            )

            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                if "rnn_hidden_states" in hl_info:
                    rnn_hidden_states[batch_ids] = hl_info[
                        "rnn_hidden_states"
                    ][batch_ids]
                    prev_actions[batch_ids] = hl_info["actions"][batch_ids]
                elif self._skills[skill_id].has_hidden_state:
                    raise ValueError(
                        f"The code does not currently support neural LL and neural HL skills. Skill={self._skills[skill_id]}, HL={self._high_level_policy}"
                    )
            hl_info["actions"] = prev_actions
            hl_info["rnn_hidden_states"] = rnn_hidden_states

            should_choose_new_skill = should_choose_new_skill.numpy()
            self._cur_skills = (
                (~should_choose_new_skill) * self._cur_skills
            ) + (should_choose_new_skill * new_skills)
        return hl_terminate_episode, hl_info

    def _get_terminations(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        actions,
        log_info: List[Dict[str, Any]],
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
        """
        Decides if the HL policy or the LL wants to terminate the current
        skill.

        :returns: A tuple containing the following (in order)
        - A tensor of shape (batch_size,) indicating if we should terminate the current skill.
        - A tensor of shape (batch_size,) indicating whether to terminate the entire episode.
        - An updated version of the input `actions`. This is needed if the skill wants
          to adjust the actions when terminating (like calling a PDDL
          condition).
        """

        hl_wants_skill_term = self._high_level_policy.get_termination(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
        )

        # Check if skills should terminate.
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )
        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
                "actions": actions,
                "hl_wants_skill_term": hl_wants_skill_term,
            },
        )
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            # TODO: either change name of the function or assign actions somewhere
            # else. Updating actions in should_terminate is counterintuitive
            (
                self._cur_call_high_level[batch_ids],
                bad_should_terminate[batch_ids],
                new_actions,
            ) = self._skills[skill_id].should_terminate(
                **dat,
                batch_idx=batch_ids,
                log_info=log_info,
                skill_name=[
                    self._idx_to_name[self._cur_skills[i]] for i in batch_ids
                ],
            )
            actions[batch_ids] += new_actions
        return self._cur_call_high_level, bad_should_terminate, actions

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        return self._high_level_policy.get_value(
            observations, rnn_hidden_states, prev_actions, masks
        )

    def _get_policy_components(self) -> List[nn.Module]:
        return self._high_level_policy.get_policy_components()

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        return self._high_level_policy.evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            rnn_build_seq_info,
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        agent_name=None,
        **kwargs,
    ):
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        return cls(
            config.habitat_baselines.rl.policy[agent_name],
            config,
            observation_space,
            orig_action_space,
            config.habitat_baselines.num_environments,
            config.habitat_baselines.rl.auxiliary_losses,
            agent_name,
        )
