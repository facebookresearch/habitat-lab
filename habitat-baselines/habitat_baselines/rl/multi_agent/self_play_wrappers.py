from typing import Dict, List

from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)
from habitat_baselines.rl.ppo.updater import Updater


class SelfBatchedPolicy(Policy):
    def __init__(self, agent: SingleAgentAccessMgr, n_agents):
        self._policy = agent.actor_critic
        self._n_agents = n_agents

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        raise NotImplementedError()

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        raise NotImplementedError()

    def get_extra(
        self, action_data: PolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        agent: SingleAgentAccessMgr,
        n_agents: int,
        **kwargs,
    ):
        return SelfBatchedPolicy(agent, n_agents)


class SelfBatchedStorage(Storage):
    def __init__(self, agent: SingleAgentAccessMgr, n_agents: int):
        self._storage = agent.rollouts
        self._n_agents = n_agents

    def insert(
        self,
        next_observations=None,
        rewards=None,
        buffer_index=0,
        next_masks=None,
        **kwargs,
    ):
        raise NotImplementedError()

    def to(self, device):
        raise NotImplementedError()

    def insert_first_observations(self, batch):
        raise NotImplementedError()

    def advance_rollout(self, buffer_index=0):
        raise NotImplementedError()

    def compute_returns(self, next_value, use_gae, gamma, tau):
        raise NotImplementedError()

    def after_update(self):
        raise NotImplementedError()

    def get_current_step(self, env_slice, buffer_index):
        raise NotImplementedError()

    def get_last_step(self):
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        agent: SingleAgentAccessMgr,
        n_agents: int,
        **kwargs,
    ):
        return SelfBatchedStorage(agent, n_agents)


class SelfBatchedUpdater(Updater):
    def __init__(self, agent: SingleAgentAccessMgr, n_agents: int):
        self._updater = agent.updater
        self.n_agents = n_agents

    def update(self, rollouts):
        raise NotImplementedError()

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        agent: SingleAgentAccessMgr,
        n_agents: int,
        **kwargs,
    ):
        return SelfBatchedUpdater(agent, n_agents)
