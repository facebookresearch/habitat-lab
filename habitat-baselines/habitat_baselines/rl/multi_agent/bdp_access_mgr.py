from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

import gym.spaces as spaces
import numpy as np
import torch

from habitat.gym.gym_wrapper import create_action_space
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.multi_agent.multi_agent_access_mgr import (
    MultiAgentAccessMgr,
)
from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiPolicy,
    MultiStorage,
    MultiUpdater,
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.multi_agent.self_play_wrappers import (
    SelfBatchedPolicy,
    SelfBatchedStorage,
    SelfBatchedUpdater,
)
from habitat_baselines.rl.multi_agent.utils import (
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

COORD_AGENT = 0
BEHAV_AGENT = 1
BEHAV_AGENT_NAME = "agent_1"

ROBOT_TYPE = 0
HUMAN_TYPE = 1

BEHAV_ID = "behav_latent"


@baseline_registry.register_agent_access_mgr
class BdpAgentAccessMgr(MultiAgentAccessMgr):
    def _sample_active_idxs(self):
        assert not self._pop_config.self_play_batched
        assert self._pop_config.num_agent_types == 2
        assert self._pop_config.num_active_agents_per_type == [1, 1]

        num_envs = self._agents[0]._num_envs
        device = self._agents[0]._device
        self._behav_latents = torch.zeros(
            (num_envs, self._pop_config.behavior_latent_dim), device=device
        )
        return np.array([COORD_AGENT, BEHAV_AGENT]), np.array(
            [ROBOT_TYPE, HUMAN_TYPE]
        )

    def _inject_behav_latent(self, obs, agent_idx):
        agent_obs = update_dict_with_agent_prefix(obs, agent_idx)
        if agent_idx == BEHAV_AGENT:
            agent_obs[BEHAV_ID] = self._behav_latents
        return agent_obs

    def _create_multi_components(self, config, env_spec, num_active_agents):
        multi_policy = MultiPolicy.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
            update_obs_with_agent_prefix_fn=self._inject_behav_latent,
        )
        multi_updater = MultiUpdater.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
        )

        multi_storage = MultiStorage.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
            update_obs_with_agent_prefix_fn=self._inject_behav_latent,
        )
        return multi_policy, multi_updater, multi_storage

    def _create_single_agent(
        self,
        config,
        agent_env_spec,
        is_distrib,
        device,
        use_resume_state,
        num_envs,
        percent_done_fn,
        lr_schedule_fn,
        agent_name,
    ):
        if agent_name == BEHAV_AGENT_NAME:
            # Inject the behavior latent into the observation spec
            agent_env_spec.observation_space[BEHAV_ID] = spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(self._pop_config.behavior_latent_dim,),
                dtype=np.float32,
            )
        return SingleAgentAccessMgr(
            config,
            agent_env_spec,
            is_distrib,
            device,
            use_resume_state,
            num_envs,
            percent_done_fn,
            lr_schedule_fn,
            agent_name,
        )
