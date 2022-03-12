import os.path as osp

import gym.spaces as spaces
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.high_level_policy import (  # noqa: F401.
    GtHighLevelPolicy,
    HighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    NavSkillPolicy,
    NnSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
)
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions


@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self, config, full_config, observation_space, action_space, num_envs
    ):
        super().__init__()

        self._action_space = action_space
        self._num_envs = num_envs

        # Maps (skill idx -> skill)
        self._skills = {}
        self._name_to_idx = {}

        for i, (skill_id, use_skill_name) in enumerate(
            config.USE_SKILLS.items()
        ):
            skill_config = config.DEFINED_SKILLS[use_skill_name]

            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config, observation_space, action_space, self._num_envs
            )
            self._skills[i] = skill_policy
            self._name_to_idx[skill_id] = i

        self._call_high_level = torch.ones(self._num_envs)
        self._cur_skills: torch.Tensor = torch.zeros(self._num_envs)

        high_level_cls = eval(config.high_level_policy.name)
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            osp.join(
                full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,
                full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",
            ),
            num_envs,
            self._name_to_idx,
        )

    def eval(self):
        pass

    @property
    def num_recurrent_layers(self):
        return self._skills[0].num_recurrent_layers

    def parameters(self):
        return self._skills[0].parameters()

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)
        self._call_high_level = self._call_high_level.to(device)
        self._cur_skills = self._cur_skills.to(device)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        self._high_level_policy.apply_mask(masks)

        batched_observations = [
            {k: v[batch_idx].unsqueeze(0) for k, v in observations.items()}
            for batch_idx in range(self._num_envs)
        ]
        batched_rnn_hidden_states = rnn_hidden_states.unsqueeze(1)
        batched_prev_actions = prev_actions.unsqueeze(1)
        batched_masks = masks.unsqueeze(1)

        # Check if skills should terminate.
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            should_terminate = self._skills[skill_idx.item()].should_terminate(
                batched_observations[batch_idx],
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
            )
            self._call_high_level[batch_idx] = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level = torch.clamp(
            self._call_high_level + (~masks).float().view(-1), 0.0, 1.0
        )

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        if self._call_high_level.sum() > 0:
            (
                new_skills,
                new_skill_args,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
            )

            for new_skill_batch_idx in torch.nonzero(self._call_high_level):
                skill_idx = new_skills[new_skill_batch_idx.item()]

                skill = self._skills[skill_idx.item()]

                (
                    batched_rnn_hidden_states[new_skill_batch_idx],
                    batched_prev_actions[new_skill_batch_idx],
                ) = skill.on_enter(
                    new_skill_args[new_skill_batch_idx],
                    new_skill_batch_idx,
                    batched_observations[new_skill_batch_idx],
                    batched_rnn_hidden_states[new_skill_batch_idx],
                    batched_prev_actions[new_skill_batch_idx],
                )
            self._cur_skills = (
                (1.0 - self._call_high_level) * self._cur_skills
            ) + (self._call_high_level * new_skills)

        actions = torch.zeros(
            self._num_envs, get_num_actions(self._action_space)
        )
        for batch_idx, skill_idx in enumerate(self._cur_skills):
            action, batched_rnn_hidden_states[batch_idx] = self._skills[
                skill_idx.item()
            ].act(
                batched_observations[batch_idx],
                batched_rnn_hidden_states[batch_idx],
                batched_prev_actions[batch_idx],
                batched_masks[batch_idx],
                batch_idx,
            )
            actions[batch_idx] = action

        return (
            None,
            actions,
            None,
            batched_rnn_hidden_states.view(rnn_hidden_states.shape),
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        return cls(
            config.RL.POLICY,
            config,
            observation_space,
            action_space,
            config.NUM_ENVIRONMENTS,
        )
