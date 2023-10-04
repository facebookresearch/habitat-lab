import logging
from itertools import chain
from typing import Any, List

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import (
    CriticHead,
    PolicyActionData,
    get_aux_modules,
)
from habitat_baselines.utils.common import CategoricalNet


class NeuralHighLevelPolicy(HighLevelPolicy):
    """
    A trained high-level policy that selects low-level skills and their skill
    inputs. Is limited to discrete skills and discrete skill inputs. The policy
    detects the available skills and their possible arguments via the PDDL
    problem.
    """

    def __init__(
        self,
        config,
        pddl_problem,
        num_envs,
        skill_name_to_idx,
        observation_space,
        action_space,
        aux_loss_config,
        agent_name,
    ):
        super().__init__(
            config,
            pddl_problem,
            num_envs,
            skill_name_to_idx,
            observation_space,
            action_space,
            aux_loss_config,
            agent_name,
        )
        self._all_actions = self._setup_actions()
        self._n_actions = len(self._all_actions)
        self._termination_obs_name = self._config.termination_obs_name

        use_obs_space = spaces.Dict(
            {
                k: self._obs_space.spaces[k]
                for k in self._config.policy_input_keys
            }
        )
        self._im_obs_space = spaces.Dict(
            {k: v for k, v in use_obs_space.items() if len(v.shape) == 3}
        )

        state_obs_space = {
            k: v for k, v in use_obs_space.items() if len(v.shape) == 1
        }
        self._state_obs_space = spaces.Dict(state_obs_space)

        rnn_input_size = sum(
            v.shape[0] for v in self._state_obs_space.values()
        )
        self._hidden_size = self._config.hidden_dim
        if len(self._im_obs_space) > 0 and self._config.backbone != "NONE":
            resnet_baseplanes = 32
            self._visual_encoder = ResNetEncoder(
                self._im_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, self._config.backbone),
            )
            self._visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self._visual_encoder.output_shape),
                    self._hidden_size,
                ),
                nn.ReLU(True),
            )
            rnn_input_size += self._hidden_size
        else:
            self._visual_encoder = nn.Sequential()
            self._visual_fc = nn.Sequential()

        self._state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=self._config.rnn_type,
            num_layers=self._config.num_rnn_layers,
        )
        self._policy = CategoricalNet(self._hidden_size, self._n_actions)
        self._critic = CriticHead(self._hidden_size)

        self.aux_modules = get_aux_modules(aux_loss_config, action_space, self)

    @property
    def should_load_agent_state(self):
        return True

    def get_termination(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_skills,
        log_info,
    ):
        if self._termination_obs_name is None:
            return super().get_termination(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                cur_skills,
                log_info,
            )
        return (observations[self._termination_obs_name] > 0.0).view(-1).cpu()

    @property
    def policy_action_space(self) -> spaces.Space:
        return spaces.Discrete(self._n_actions)

    @property
    def num_recurrent_layers(self):
        return self._state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    def parameters(self):
        return chain(
            self._visual_encoder.parameters(),
            self._visual_fc.parameters(),
            self._policy.parameters(),
            self._state_encoder.parameters(),
            self._critic.parameters(),
            self.aux_modules.parameters(),
        )

    def get_policy_components(self) -> List[nn.Module]:
        return [self]

    def forward(self, obs, rnn_hidden_states, masks, rnn_build_seq_info=None):
        hidden = []
        if len(self._im_obs_space) > 0:
            im_obs = {k: obs[k] for k in self._im_obs_space.keys()}
            visual_features = self._visual_encoder(im_obs)
            visual_features = self._visual_fc(visual_features)
            hidden.append(visual_features)

        if len(self._state_obs_space) > 0:
            hidden.extend([obs[k] for k in self._state_obs_space.keys()])
        hidden = torch.cat(hidden, -1)

        return self._state_encoder(
            hidden, rnn_hidden_states, masks, rnn_build_seq_info
        )

    def to(self, device):
        self._device = device
        return super().to(device)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        state, _ = self.forward(observations, rnn_hidden_states, masks)
        return self._critic(state)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        features, rnn_hidden_states = self.forward(
            observations, rnn_hidden_states, masks, rnn_build_seq_info
        )
        distribution = self._policy(features)
        value = self._critic(features)
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()
        aux_loss_res = {
            k: v(features, observations) for k, v in self.aux_modules.items()
        }

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        batch_size = plan_masks.shape[0]
        next_skill = torch.zeros(batch_size, dtype=torch.long)
        skill_args_data: List[Any] = [None for _ in range(batch_size)]
        immediate_end = torch.zeros(batch_size, dtype=torch.bool)

        state, rnn_hidden_states = self.forward(
            observations, rnn_hidden_states, masks
        )
        distrib = self._policy(state)
        values = self._critic(state)
        if deterministic:
            skill_sel = distrib.mode()
        else:
            skill_sel = distrib.sample()
        action_log_probs = distrib.log_probs(skill_sel)

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            use_ac = self._all_actions[skill_sel[batch_idx]]
            if baselines_logger.level >= logging.DEBUG:
                baselines_logger.debug(f"HL Policy selected skill {use_ac}")
            next_skill[batch_idx] = self._skill_name_to_idx[use_ac.name]
            skill_args_data[batch_idx] = [
                entity.name for entity in use_ac.param_values
            ]
            log_info[batch_idx]["nn_action"] = use_ac.compact_str

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(
                action_log_probs=action_log_probs,
                values=values,
                actions=skill_sel,
                rnn_hidden_states=rnn_hidden_states,
            ),
        )
