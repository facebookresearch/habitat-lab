from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import gym.spaces as spaces
import numpy as np

from habitat.gym.gym_wrapper import create_action_space
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiPolicy,
    MultiStorage,
    MultiUpdater,
    filter_agent_names,
)
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_agent
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

        for k in env_spec.orig_action_space:
            if not k.startswith("agent"):
                raise ValueError(
                    f"Multi-agent training requires splitting the action space between the agents. Shared actions are not supported yet {k}"
                )

        for agent_i in range(self._pop_config.num_total_agents):
            self._all_agent_idxs.append(agent_i)
            use_resume_state = None
            if resume_state is not None:
                use_resume_state = resume_state[agent_i]

            agent_obs_space = spaces.Dict(
                filter_agent_names(env_spec.observation_space, agent_i)
            )
            agent_orig_action_space = spaces.Dict(
                filter_agent_names(env_spec.orig_action_space.spaces, agent_i)
            )
            agent_action_space = create_action_space(agent_orig_action_space)
            agent_env_spec = EnvironmentSpec(
                observation_space=agent_obs_space,
                action_space=agent_action_space,
                orig_action_space=agent_orig_action_space,
            )
            self._agents.append(
                SingleAgentAccessMgr(
                    config,
                    agent_env_spec,
                    is_distrib,
                    device,
                    resume_state,
                    num_envs,
                    percent_done_fn,
                    lr_schedule_fn,
                )
            )
        self._multi_policy = MultiPolicy.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
        )
        self._multi_updater = MultiUpdater.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
        )

        self._multi_storage = MultiStorage.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
        )

        if self.nbuffers != 1:
            raise ValueError(
                "Multi-agent training does not support double buffered sampling"
            )

    @property
    def nbuffers(self):
        return self._agents[0].nbuffers

    def _sample_active(self):
        """
        Samples the set of agents currently active in the episode.
        """

        # Random sample over which agents are active.
        self._active_agents = np.random.choice(
            self._all_agent_idxs,
            size=2,
            replace=self._pop_config.allow_self_play,
        )

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
        for agent in self._agents:
            agent.post_init(create_rollouts_fn)

        self._num_updates = 0
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
        for agent in self._agents:
            agent.load_state_dict(state)

    def load_ckpt_state_dict(self, ckpt):
        for agent in self._agents:
            agent.load_ckpt_state_dict(ckpt)

    @property
    def hidden_state_shape(self):
        """
        Stack the hidden states of all the policies in the active population.
        """

        return np.sum(
            np.stack(
                [
                    self._agents[i].hidden_state_shape
                    for i in self._active_agents
                ]
            ),
            dim=0,
        )

    def after_update(self):
        """
        Will resample the active agent population every `agent_sample_interval` calls to this method.
        """

        for agent in self._agents:
            agent.after_update()
        self._num_updates += 1
        if self._num_updates % self._pop_config.agent_sample_interval == 0:
            self._sample_active()

    def pre_rollout(self):
        for agent in self._agents:
            agent.pre_rollout()

    def get_resume_state(self):
        return {
            agent_i: agent.get_resume_state()
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
        return self._agents[0].policy_action_space
