#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import gym.spaces as spaces

import habitat.gym.gym_wrapper as gym_wrapper
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import get_active_obs_transforms
from habitat_baselines.rl.multi_agent.multi_agent_access_mgr import (
    MultiAgentAccessMgr,
)
from habitat_baselines.rl.multi_agent.utils import (
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)
from habitat_hitl.environment.controllers.controller_abc import (
    BaselinesController,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat.core.environments import GymHabitatEnv


def clean_dict(d, remove_prefix):
    ret_d = {}
    for k, v in d.spaces.items():
        if k.startswith(remove_prefix):
            new_k = k[len(remove_prefix) :]
            if isinstance(v, spaces.Dict):
                ret_d[new_k] = clean_dict(v, remove_prefix)
            else:
                ret_d[new_k] = v
        elif not k.startswith("agent"):
            ret_d[k] = v
    return spaces.Dict(ret_d)


class SingleAgentBaselinesController(BaselinesController):
    """Controller for single baseline agent."""

    def __init__(
        self,
        agent_idx: int,
        is_multi_agent: bool,
        config: "DictConfig",
        gym_habitat_env: "GymHabitatEnv",
    ):
        self._agent_idx: int = agent_idx
        self._agent_name: str = config.habitat.simulator.agents_order[
            self._agent_idx
        ]

        self._agent_k: str
        if is_multi_agent:
            self._agent_k = f"agent_{self._agent_idx}_"
        else:
            self._agent_k = ""

        super().__init__(
            is_multi_agent,
            config,
            gym_habitat_env,
        )

    def _create_env_spec(self):
        # udjust the observation and action space to be agent specific (remove other agents)
        original_action_space = clean_dict(
            self._gym_habitat_env.original_action_space, self._agent_k
        )
        observation_space = clean_dict(
            self._gym_habitat_env.observation_space, self._agent_k
        )
        action_space = gym_wrapper.create_action_space(original_action_space)

        env_spec = EnvironmentSpec(
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=original_action_space,
        )

        return env_spec

    def _get_active_obs_transforms(self):
        return get_active_obs_transforms(self._config, self._agent_name)

    def _create_agent(self):
        agent = SingleAgentAccessMgr(
            agent_name=self._agent_name,
            config=self._config,
            env_spec=self._env_spec,
            num_envs=self._num_envs,
            is_distrib=False,
            device=self.device,
            percent_done_fn=lambda: 0,
        )

        return agent

    def _load_agent_state_dict(self, checkpoint):
        self._agent.load_state_dict(checkpoint[self._agent_idx])

    def _batch_and_apply_transforms(self, obs):
        batch = super()._batch_and_apply_transforms(obs)
        batch = update_dict_with_agent_prefix(batch, self._agent_idx)

        return batch


class MultiAgentBaselinesController(BaselinesController):
    """Controller for multiple baseline agents."""

    def _create_env_spec(self):
        observation_space = self._gym_habitat_env.observation_space
        action_space = self._gym_habitat_env.action_space
        original_action_space = self._gym_habitat_env.original_action_space

        env_spec = EnvironmentSpec(
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=original_action_space,
        )

        return env_spec

    def _get_active_obs_transforms(self):
        return get_active_obs_transforms(self._config)

    def _create_agent(self):
        agent = MultiAgentAccessMgr(
            config=self._config,
            env_spec=self._env_spec,
            num_envs=self._num_envs,
            is_distrib=False,
            device=self.device,
            percent_done_fn=lambda: 0,
        )

        return agent

    def _load_agent_state_dict(self, checkpoint):
        self._agent.load_state_dict(checkpoint)
