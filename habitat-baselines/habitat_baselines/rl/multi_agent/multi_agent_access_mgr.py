from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

import gym.spaces as spaces
import numpy as np

from habitat.gym.gym_wrapper import create_action_space
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
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
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_agent_access_mgr
class MultiAgentAccessMgr(AgentAccessMgr):
    """
    Maintains a population of agents. A subset of the overall of this
    population is maintained as active agents. The active agent population acts
    in a multi-agent environment. This is achieved by wrapping the active agent
    population in the multi-agent updaters/policies/storage wrappers under
    `habitat_baselines/rl/multi_agent/pop_play_wrappers.py`. The active agent
    pouplation is randomly re-sampled at a fixed interval.
    """

    def __init__(
        self,
        config: "DictConfig",
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done_fn: Callable[[], float],
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
    ):
        self._agents = []
        self._all_agent_idxs = []
        self._pop_config = config.habitat_baselines.rl.agent
        self._is_post_init = True

        for k in env_spec.orig_action_space:
            if not k.startswith("agent"):
                raise ValueError(
                    f"Multi-agent training requires splitting the action space between the agents. Shared actions are not supported yet {k}"
                )
        self._agents, self._all_agent_idxs = self._get_agents(
            config,
            env_spec,
            is_distrib,
            device,
            resume_state,
            num_envs,
            percent_done_fn,
            lr_schedule_fn,
        )

        if self._pop_config.self_play_batched:
            policy_cls: Type = SelfBatchedPolicy
            updater_cls: Type = SelfBatchedUpdater
            storage_cls: Type = SelfBatchedStorage
            self._active_agents = [0, 0]
        else:
            policy_cls = MultiPolicy
            updater_cls = MultiUpdater
            storage_cls = MultiStorage

        self._multi_policy = policy_cls.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=self._pop_config.num_active_agents,
        )
        self._multi_updater = updater_cls.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=self._pop_config.num_active_agents,
        )

        self._multi_storage = storage_cls.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=self._pop_config.num_active_agents,
        )

        if self.nbuffers != 1:
            raise ValueError(
                "Multi-agent training does not support double buffered sampling"
            )
        if config.habitat_baselines.evaluate:
            self._sample_active()

    def _get_agents(
        self,
        config: "DictConfig",
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done_fn: Callable[[], float],
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
    ):
        all_agent_idxs = []
        agents = []
        for agent_i in range(self._pop_config.num_total_agents):
            all_agent_idxs.append(agent_i)
            use_resume_state = None
            if resume_state is not None:
                use_resume_state = resume_state[str(agent_i)]

            agent_obs_space = spaces.Dict(
                update_dict_with_agent_prefix(
                    env_spec.observation_space, agent_i
                )
            )
            agent_orig_action_space = spaces.Dict(
                update_dict_with_agent_prefix(
                    env_spec.orig_action_space.spaces, agent_i
                )
            )
            agent_action_space = create_action_space(agent_orig_action_space)
            agent_env_spec = EnvironmentSpec(
                observation_space=agent_obs_space,
                action_space=agent_action_space,
                orig_action_space=agent_orig_action_space,
            )
            agents.append(
                SingleAgentAccessMgr(
                    config,
                    agent_env_spec,
                    is_distrib,
                    device,
                    use_resume_state,
                    num_envs,
                    percent_done_fn,
                    lr_schedule_fn,
                )
            )
        return agents, all_agent_idxs

    @property
    def nbuffers(self):
        return self._agents[0].nbuffers

    def _sample_active(self):
        """
        Samples the set of agents currently active in the episode.
        """
        assert not self._pop_config.self_play_batched

        # Random sample over which agents are active.
        self._active_agents = np.random.choice(
            self._all_agent_idxs,
            size=self._pop_config.num_active_agents,
            replace=self._pop_config.allow_self_play,
        )

        if not self._is_post_init:
            # If not post init then we are running in evaluation mode and
            # should only setup the policy
            self._multi_storage.set_active(
                [self._agents[i].rollouts for i in self._active_agents]
            )
            self._multi_updater.set_active(
                [self._agents[i].updater for i in self._active_agents]
            )
        self._multi_policy.set_active(
            [self._agents[i].actor_critic for i in self._active_agents]
        )

    def post_init(self, create_rollouts_fn=None):
        self._is_post_init = True
        for agent in self._agents:
            agent.post_init(create_rollouts_fn)

        self._num_updates = 0
        if not self._pop_config.self_play_batched:
            self._sample_active()

    def eval(self):
        for agent in self._agents:
            agent.eval()

    def train(self):
        for agent in self._agents:
            agent.train()

    @property
    def num_total_agents(self):
        return len(self._agents)

    def load_state_dict(self, state):
        for agent_i, agent in enumerate(self._agents):
            agent.load_state_dict(state[agent_i])

    def load_ckpt_state_dict(self, ckpt):
        for agent in self._agents:
            agent.load_ckpt_state_dict(ckpt)

    @property
    def masks_shape(self):
        return (
            sum(self._agents[i].masks_shape[0] for i in self._active_agents),
        )

    @property
    def hidden_state_shape(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        hidden_shapes = np.stack(
            [self._agents[i].hidden_state_shape for i in self._active_agents]
        )
        any_hidden_shape = hidden_shapes[0]
        # The hidden states will be concatenated over the last dimension.
        return [*any_hidden_shape[:-1], np.sum(hidden_shapes[:, -1])]

    def after_update(self):
        """
        Will resample the active agent population every `agent_sample_interval` calls to this method.
        """

        for agent in self._agents:
            agent.after_update()
        self._num_updates += 1
        if (
            self._num_updates % self._pop_config.agent_sample_interval == 0
            and self._pop_config.agent_sample_interval != -1
        ):
            self._sample_active()

    def pre_rollout(self):
        for agent in self._agents:
            agent.pre_rollout()

    def get_resume_state(self):
        return {
            str(agent_i): agent.get_resume_state()
            for agent_i, agent in enumerate(self._agents)
        }

    def get_save_state(self):
        return {
            agent_i: agent.get_save_state()
            for agent_i, agent in enumerate(self._agents)
        }

    @property
    def rollouts(self):
        return self._multi_storage

    @property
    def actor_critic(self):
        return self._multi_policy

    @property
    def updater(self):
        return self._multi_updater

    @property
    def policy_action_space(self):
        # TODO: Hack for discrete HL action spaces.
        return spaces.MultiDiscrete(
            tuple([agent.policy_action_space.n for agent in self._agents])
        )

    def update_hidden_state(self, rnn_hxs, prev_actions, action_data):
        """
        Update the hidden state of the agents in the population. Writes to the
        data in place.
        """
        n_agents = len(self._active_agents)
        hxs_dim = rnn_hxs.shape[-1] // n_agents
        ac_dim = prev_actions.shape[-1] // n_agents
        # Not very efficient, but update each agent's hidden state individually.
        for env_i, should_insert in enumerate(action_data.should_inserts):
            for policy_i, agent_should_insert in enumerate(should_insert):
                if agent_should_insert.item():
                    rnn_sel = slice(
                        policy_i * hxs_dim, (policy_i + 1) * hxs_dim
                    )
                    rnn_hxs[env_i, :, rnn_sel] = action_data.rnn_hidden_states[
                        env_i, :, rnn_sel
                    ]

                    ac_sel = slice(policy_i * ac_dim, (policy_i + 1) * ac_dim)
                    prev_actions[env_i, ac_sel].copy_(
                        action_data.actions[env_i, ac_sel]
                    )
