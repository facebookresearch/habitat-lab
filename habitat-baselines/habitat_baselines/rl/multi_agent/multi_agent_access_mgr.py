from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type

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
        self._agent_count_idxs = []
        self._pop_config = config.habitat_baselines.rl.agent
        self._rnd = np.random.RandomState(seed=42)

        # Tracks if the agent storage is setup.
        self._is_post_init = False

        for k in env_spec.orig_action_space:
            if not k.startswith("agent"):
                raise ValueError(
                    f"Multi-agent training requires splitting the action space between the agents. Shared actions are not supported yet {k}"
                )
        self._agents, self._agent_count_idxs = self._get_agents(
            config,
            env_spec,
            is_distrib,
            device,
            resume_state,
            num_envs,
            percent_done_fn,
            lr_schedule_fn,
        )

        num_active_agents = sum(self._pop_config.num_active_agents_per_type)
        (
            self._multi_policy,
            self._multi_updater,
            self._multi_storage,
        ) = self._create_multi_components(config, env_spec, num_active_agents)

        if self.nbuffers != 1:
            raise ValueError(
                "Multi-agent training does not support double buffered sampling"
            )
        if config.habitat_baselines.evaluate:
            self._sample_active()

    def init_distributed(self, find_unused_params: bool = True) -> None:
        for agent in self._agents:
            agent.init_distributed(find_unused_params)

    def _create_multi_components(self, config, env_spec, num_active_agents):
        if self._pop_config.self_play_batched:
            policy_cls: Type = SelfBatchedPolicy
            updater_cls: Type = SelfBatchedUpdater
            storage_cls: Type = SelfBatchedStorage
            self._active_agents = np.array([0, 0])
        else:
            policy_cls = MultiPolicy
            updater_cls = MultiUpdater
            storage_cls = MultiStorage

        # TODO(xavi to andrew): why do we call these functions? It seems they
        # just create an empty class

        multi_policy = policy_cls.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
        )
        multi_updater = updater_cls.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
        )

        multi_storage = storage_cls.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
        )
        return multi_policy, multi_updater, multi_storage

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
        agent_count_idxs = [0]
        agents = []
        for agent_i in range(self._pop_config.num_agent_types):
            num_agents_type = self._pop_config.num_pool_agents_per_type[
                agent_i
            ]
            agent_count_idxs.append(num_agents_type)

            for agent_type_i in range(num_agents_type):
                agent_ct = agent_i * agent_count_idxs[agent_i] + agent_type_i

                use_resume_state = None
                if resume_state is not None:
                    use_resume_state = resume_state[str(agent_ct)]

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
                agent_action_space = create_action_space(
                    agent_orig_action_space
                )
                agent_env_spec = EnvironmentSpec(
                    observation_space=agent_obs_space,
                    action_space=agent_action_space,
                    orig_action_space=agent_orig_action_space,
                )
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agents.append(
                    self._create_single_agent(
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
                )
        return agents, agent_count_idxs[1:]

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

    @property
    def nbuffers(self):
        return self._agents[0].nbuffers

    def _sample_active_idxs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns indices of active agents.
        """
        assert not self._pop_config.self_play_batched

        # Random sample over which agents are active.
        prev_num_agents = 0
        active_agents = []
        active_agent_types = []
        for agent_type_ind in range(self._pop_config.num_agent_types):
            if self._pop_config.num_active_agents_per_type[agent_type_ind] > 1:
                raise ValueError(
                    "The current code only supports sampling one agent of a given type at a time"
                )
            if (
                self._pop_config.force_partner_sample_idx >= 0
                and agent_type_ind > 0
            ):
                # We want to force the selection of an agent from the
                # population for the non-coordination agent (the coordination
                # agent is at index 0.
                active_agents_type = np.array(
                    [self._pop_config.force_partner_sample_idx]
                )
            else:
                active_agents_type = self._rnd.choice(
                    self._agent_count_idxs[agent_type_ind],
                    size=self._pop_config.num_active_agents_per_type[
                        agent_type_ind
                    ],
                )
            agent_cts = active_agents_type + prev_num_agents
            prev_num_agents += self._agent_count_idxs[agent_type_ind]
            active_agents.append(agent_cts)
            active_agent_types.append(
                np.ones(agent_cts.shape, dtype=np.int32) * agent_type_ind
            )
        return np.concatenate(active_agents), np.concatenate(
            active_agent_types
        )

    def _sample_active(self):
        """
        Samples the set of agents currently active in the episode.
        """

        self._active_agents, active_agent_types = self._sample_active_idxs()

        if self._is_post_init:
            # If not post init then we are running in evaluation mode and
            # should only setup the policy
            self._multi_storage.set_active(
                [self._agents[i].rollouts for i in self._active_agents],
                active_agent_types,
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
            if not agent.actor_critic.should_load_agent_state:
                continue
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
        # We do max because some policies may be non-neural
        # And will have a hidden state of [0, hidden_dim]
        max_hidden_shape = hidden_shapes.max(0)
        # The hidden states will be concatenated over the last dimension.
        return [*max_hidden_shape[:-1], np.sum(hidden_shapes[:, -1])]

    @property
    def hidden_state_shape_lens(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        hidden_indices = [
            self._agents[i].hidden_state_shape[-1] for i in self._active_agents
        ]
        return hidden_indices

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
            prev_rollouts = [
                self._agents[i].rollouts for i in self._active_agents
            ]
            self._sample_active()
            cur_rollouts = [
                self._agents[i].rollouts for i in self._active_agents
            ]

            # We just sampled new agents. We also need to reset the storage buffer current and starting state.
            for prev_rollout, cur_rollout in zip(prev_rollouts, cur_rollouts):
                # Need to call `insert_first` in case the rollout buffer has
                # some special setup logic (like in `HrlRolloutStorage` for
                # tracking the current step).
                cur_rollout.insert_first_observations(
                    prev_rollout.buffers["observations"][0]
                )
                cur_rollout.buffers[0] = prev_rollout.buffers[0]

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
        all_discrete = np.all(
            [
                isinstance(agent.policy_action_space, spaces.MultiDiscrete)
                for agent in self._agents
            ]
        )
        if all_discrete:
            return spaces.MultiDiscrete(
                tuple(
                    [
                        self._agents[agent_i].policy_action_space.n
                        for agent_i in self._active_agents
                    ]
                )
            )
        else:
            return spaces.Dict(
                {
                    agent_i: self._agents[agent_i].policy_action_space
                    for agent_i in self._active_agents
                }
            )

    @property
    def policy_action_space_shape_lens(self):
        lens = []
        for agent_i in self._active_agents:
            agent = self._agents[agent_i]
            if isinstance(agent.policy_action_space, spaces.Discrete):
                lens.append(1)
            elif isinstance(agent.policy_action_space, spaces.Box):
                lens.append(agent.policy_action_space.shape[0])
            else:
                raise ValueError(
                    f"Action distribution {agent.policy_action_space}"
                    "not supported."
                )
        return lens

    def update_hidden_state(self, rnn_hxs, prev_actions, action_data):
        # TODO: will not work with different hidden states
        n_agents = len(self._active_agents)
        hxs_dim = rnn_hxs.shape[-1] // n_agents
        ac_dim = prev_actions.shape[-1] // n_agents
        # Not very efficient, but update each policies's hidden state individually.
        for env_i, should_insert in enumerate(action_data.should_inserts):
            for policy_i, agent_should_insert in enumerate(should_insert):
                if not agent_should_insert.item():
                    continue
                rnn_sel = slice(policy_i * hxs_dim, (policy_i + 1) * hxs_dim)
                rnn_hxs[env_i, :, rnn_sel] = action_data.rnn_hidden_states[
                    env_i, :, rnn_sel
                ]

                ac_sel = slice(policy_i * ac_dim, (policy_i + 1) * ac_dim)
                prev_actions[env_i, ac_sel].copy_(
                    action_data.actions[env_i, ac_sel]
                )
