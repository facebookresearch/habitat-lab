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
from habitat.tasks.rearrange.multi_task.pddl_domain import (
    PddlDomain,
    PddlProblem,
)
from habitat.tasks.rearrange.multi_task.pddl_sensors import PddlSuccess
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.hrl.hl import *  # noqa: F403,F401.
from habitat_baselines.rl.hrl.hl import HighLevelPolicy
from habitat_baselines.rl.hrl.skills import *  # noqa: F403,F401.
from habitat_baselines.rl.hrl.skills import NoopSkillPolicy, SkillPolicy
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
        orig_action_space: ActionSpace,
        num_envs: int,
        aux_loss_config,
        agent_name: Optional[str],
    ):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)
        self._num_envs: int = num_envs

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}
        # Can map multiple skills to the same underlying skill controller.
        self._skill_redirects: Dict[int, int] = {}

        if "rearrange_stop" not in orig_action_space.spaces:
            raise ValueError("Hierarchical policy requires the stop action")
        self._stop_action_idx, _ = find_action_range(
            orig_action_space, "rearrange_stop"
        )

        self._pddl = self._create_pddl(full_config, config)
        self._create_skills(
            {
                k: v
                for k, v in config.hierarchical_policy.defined_skills.items()
                if k not in config.hierarchical_policy.ignore_skills
            },
            observation_space,
            orig_action_space,
            full_config,
        )
        self._max_skill_rnn_layers = max(
            skill.num_recurrent_layers for skill in self._skills.values()
        )

        self._cur_skills: np.ndarray = np.full(
            (self._num_envs,), -1, dtype=np.int32
        )
        # Init with True so we always call the HL policy during the first step
        # it runs.
        self._cur_call_high_level: torch.BoolTensor = torch.ones(
            (self._num_envs,), dtype=torch.bool
        )

        self._active_envs: torch.BoolTensor = torch.ones(
            (self._num_envs,), dtype=torch.bool
        )

        high_level_cls = self._get_hl_policy_cls(config)
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config=config.hierarchical_policy.high_level_policy,
            pddl_problem=self._pddl,
            num_envs=num_envs,
            skill_name_to_idx=self._name_to_idx,
            observation_space=observation_space,
            action_space=orig_action_space,
            aux_loss_config=aux_loss_config,
            agent_name=agent_name,
        )
        first_idx: Optional[int] = None

        # Remap all the Noop skills to the same underlying skill so all the
        # calls to these are batched together.
        for skill_i, skill in self._skills.items():
            if isinstance(skill, NoopSkillPolicy):
                if first_idx is None:
                    first_idx = skill_i
                else:
                    self._skill_redirects[skill_i] = first_idx

        self._recurrent_hidden_size = (
            full_config.habitat_baselines.rl.ppo.hidden_size
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
            read_config=False,
        )

    def eval(self):
        pass

    @property
    def policy_action_space(self):
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """
        if self._has_ll_hidden_state or not self._has_hl_hidden_state:
            # The LL skill will take priority for the prev action.
            return super().policy_action_space
        else:
            return self._high_level_policy.policy_action_space

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

            did_skill_fail = dones[i] and not info[PddlSuccess.cls_uuid]
            for skill_name, idx in self._name_to_idx.items():
                ret_policy_info[f"failed_skill_{skill_name}"] = (
                    did_skill_fail if idx == cur_skill_idx else 0.0
                )
            ret_policy_infos.append(ret_policy_info)

        return ret_policy_infos

    @property
    def hidden_state_shape(self):
        return (
            self.num_recurrent_layers,
            self.recurrent_hidden_size,
        )

    @property
    def hidden_state_shape_lens(self):
        return [self.recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self) -> int:
        return self._recurrent_hidden_size

    @property
    def num_recurrent_layers(self):
        return (
            self._max_skill_rnn_layers
            + self._high_level_policy.num_recurrent_layers
        )

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
        batch_size = masks.shape[0]
        masks_cpu = masks.cpu()
        log_info: List[Dict[str, Any]] = [{} for _ in range(batch_size)]
        self._high_level_policy.apply_mask(masks_cpu)  # type: ignore[attr-defined]

        # Initialize empty action set based on the overall action space.
        actions = torch.zeros(
            (batch_size, get_num_actions(self._action_space)),
            device=masks.device,
        )
        hl_rnn_hidden_states, ll_rnn_hidden_states = self._split_hidden_states(
            rnn_hidden_states
        )

        # Always call high-level if the episode is over.
        self._cur_call_high_level |= (~masks_cpu).view(-1)

        hl_terminate_episode, hl_info = self._update_skills(
            observations,
            hl_rnn_hidden_states,
            ll_rnn_hidden_states,
            prev_actions,
            masks,
            actions,
            log_info,
            self._cur_call_high_level,
            deterministic,
        )
        did_choose_new_skill = self._cur_call_high_level.clone()
        if hl_info.rnn_hidden_states is not None and self._has_hl_hidden_state:
            # Update the HL hidden state.
            hl_rnn_hidden_states = hl_info.rnn_hidden_states

        if hl_info.policy_info is not None:
            # Merge the infos.
            for env_i, info in enumerate(hl_info.policy_info):
                log_info[env_i].update(info)

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": ll_rnn_hidden_states,
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

            if self._has_ll_hidden_state:
                # Update the LL hidden state.
                ll_rnn_hidden_states[batch_ids] = action_data.rnn_hidden_states

        # Skills should not be responsible for terminating the overall episode.
        actions[:, self._stop_action_idx] = 0.0

        (
            self._cur_call_high_level,
            bad_should_terminate,
            actions,
        ) = self._get_terminations(
            observations,
            hl_rnn_hidden_states,
            ll_rnn_hidden_states,
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

        rnn_hidden_states = self._combine_hidden_states(
            hl_rnn_hidden_states, ll_rnn_hidden_states
        )

        # This will update the prev action
        if (not self._has_hl_hidden_state) or self._has_ll_hidden_state:
            # The LL skill will take priority for the prev action
            use_action = actions
        else:
            use_action = hl_info.actions

        return PolicyActionData(
            take_actions=actions,
            policy_info=log_info,
            should_inserts=did_choose_new_skill.view(-1, 1),
            actions=use_action,
            values=hl_info.values,
            action_log_probs=hl_info.action_log_probs,
            rnn_hidden_states=rnn_hidden_states,
        )

    @property
    def _has_hl_hidden_state(self) -> bool:
        return self._high_level_policy.num_recurrent_layers != 0

    @property
    def _has_ll_hidden_state(self) -> bool:
        return self._max_skill_rnn_layers != 0

    def _split_hidden_states(
        self, rnn_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self._has_hl_hidden_state) or (not self._has_ll_hidden_state):
            # No need to split hidden states if both aren't being used.
            return rnn_hidden_states, rnn_hidden_states
        else:
            hl_num_layers = self._high_level_policy.num_recurrent_layers
            # Split the hidden state for HL and LL policies
            hl_rnn_hidden_states = rnn_hidden_states[:, :hl_num_layers]
            ll_rnn_hidden_states = rnn_hidden_states[:, hl_num_layers:]
            return hl_rnn_hidden_states, ll_rnn_hidden_states

    def _combine_hidden_states(
        self,
        hl_rnn_hidden_states: torch.Tensor,
        ll_rnn_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self._has_hl_hidden_state:
            return ll_rnn_hidden_states
        elif not self._has_ll_hidden_state:
            return hl_rnn_hidden_states
        else:
            # Stack the LL and HL hidden states.
            return torch.cat(
                [hl_rnn_hidden_states, ll_rnn_hidden_states], dim=1
            )

    def _update_skills(
        self,
        observations,
        hl_rnn_hidden_states,
        ll_rnn_hidden_states,
        prev_actions,
        masks,
        actions,
        log_info,
        should_choose_new_skill: torch.BoolTensor,
        deterministic: bool,
    ) -> Tuple[torch.BoolTensor, PolicyActionData]:
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
        batch_size = masks.shape[0]
        hl_terminate_episode = torch.zeros(batch_size, dtype=torch.bool)
        if should_choose_new_skill.sum() > 0:
            (
                new_skills,
                new_skill_args,
                hl_terminate_episode,
                hl_info,
            ) = self._high_level_policy.get_next_skill(
                observations,
                hl_rnn_hidden_states,
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
                (
                    ll_rnn_hidden_states_batched,
                    prev_actions_batched,
                ) = self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    ll_rnn_hidden_states,
                    prev_actions,
                    skill_name=[
                        self._idx_to_name[new_skills[i]] for i in batch_ids
                    ],
                )

                if self._has_ll_hidden_state:
                    ll_rnn_hidden_states = _write_tensor_batched(
                        ll_rnn_hidden_states,
                        ll_rnn_hidden_states_batched,
                        batch_ids,
                    )
                prev_actions = _write_tensor_batched(
                    prev_actions, prev_actions_batched, batch_ids
                )

                if (
                    hl_info.rnn_hidden_states is not None
                    and self._has_hl_hidden_state
                ):
                    # Only update the RNN hidden state for NEW skills.
                    hl_rnn_hidden_states = _update_tensor_batched(
                        hl_rnn_hidden_states,
                        hl_info.rnn_hidden_states,
                        batch_ids,
                    )

            # We made at least some decisions, so update the action info
            hl_info.rnn_hidden_states = hl_rnn_hidden_states

            should_choose_new_skill = should_choose_new_skill.numpy()
            self._cur_skills = (
                (~should_choose_new_skill) * self._cur_skills
            ) + (should_choose_new_skill * new_skills)
        else:
            # We made no decisions, so return an empty HL action info.
            hl_info = PolicyActionData()
        return hl_terminate_episode, hl_info

    def _get_terminations(
        self,
        observations,
        hl_rnn_hidden_states,
        ll_rnn_hidden_states,
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
            hl_rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
        )

        # Check if skills should terminate.
        batch_size = masks.shape[0]
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (batch_size,), dtype=torch.bool
        )
        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": ll_rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
                "actions": actions,
                "hl_wants_skill_term": hl_wants_skill_term,
            },
        )
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            (
                call_hl_batch,
                bad_should_terminate_batch,
                new_actions,
            ) = self._skills[skill_id].should_terminate(
                **dat,
                batch_idx=batch_ids,
                log_info=log_info,
                skill_name=[
                    self._idx_to_name[self._cur_skills[i]] for i in batch_ids
                ],
            )

            self._cur_call_high_level = _write_tensor_batched(
                self._cur_call_high_level, call_hl_batch, batch_ids
            )
            bad_should_terminate = _write_tensor_batched(
                bad_should_terminate, bad_should_terminate_batch, batch_ids
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

    def on_envs_pause(self, envs_to_pause):
        """
        Cleans up stateful variables of the policy so that they match with the
        active environments
        """

        if len(envs_to_pause) == 0:
            return
        # One hot of envs to pause
        all_envs_to_keep_active = self._active_envs.clone()
        all_envs_to_keep_active[envs_to_pause] = False

        # Filtering the new envs that we need to keep active
        curr_envs_to_keep_active = all_envs_to_keep_active[self._active_envs]

        self._cur_call_high_level = self._cur_call_high_level[
            curr_envs_to_keep_active
        ]
        self._cur_skills = self._cur_skills[curr_envs_to_keep_active]

        self._active_envs = all_envs_to_keep_active
        self._high_level_policy.filter_envs(curr_envs_to_keep_active)

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
            config=config.habitat_baselines.rl.policy[agent_name],
            full_config=config,
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=orig_action_space,
            num_envs=config.habitat_baselines.num_environments,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            agent_name=agent_name,
        )


def _write_tensor_batched(
    source_tensor: torch.Tensor,
    write_tensor: torch.Tensor,
    write_idxs: List[int],
) -> torch.Tensor:
    """
    This assumes that write_tensor has already been indexed into by
    `write_idxs` and only needs to be copied to `source_tensor`. Returns the
    updated `source_tensor`.
    """

    if source_tensor.shape[0] == len(write_idxs):
        source_tensor = write_tensor
    else:
        source_tensor[write_idxs] = write_tensor
    return source_tensor


def _update_tensor_batched(
    source_tensor: torch.Tensor,
    write_tensor: torch.Tensor,
    write_idxs: List[int],
) -> torch.Tensor:
    """
    Writes the indices of `write_idxs` from `write_tensor` into
    `source_tensor`. Returns the updated `source_tensor`.
    """

    if source_tensor.shape[0] == len(write_idxs):
        source_tensor = write_tensor[write_idxs]
    else:
        source_tensor[write_idxs] = write_tensor[write_idxs]
    return source_tensor
