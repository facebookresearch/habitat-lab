from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type

import gym.spaces as spaces
import numpy as np
import torch

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
        num_envs: int,
        percent_done_fn: Callable[[], float],
        resume_state: Optional[Dict[str, Any]] = None,
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
        if self._pop_config.load_type1_pop_ckpts is not None:
            self._load_type1_ckpts(
                self._agents[self._agent_count_idxs[0] :],
                self._pop_config.load_type1_pop_ckpts,
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

    def _load_type1_ckpts(self, agents, ckpt_paths):
        """
        Only loads checkpoints for the type 1 agents.
        """

        if len(agents) != len(ckpt_paths):
            raise ValueError(
                f"{len(agents)} in population, doesn't match number of requested ckpts {len(ckpt_paths)}"
            )

        for agent, ckpt_path in zip(agents, ckpt_paths):
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            # Fetch the 1st agent from the type 1 population in the
            # checkpoint.
            agent.load_state_dict(ckpt_dict[1])

    def init_distributed(self, find_unused_params: bool = True) -> None:
        for agent in self._agents:
            agent.init_distributed(find_unused_params)

    def _create_multi_components(self, config, env_spec, num_active_agents):
        """
        Create the policy, updater, and storage components. These change if it
        is multi-agent training or self-play training.
        """

        if self._pop_config.self_play_batched:
            policy_cls: Type = SelfBatchedPolicy
            updater_cls: Type = SelfBatchedUpdater
            storage_cls: Type = SelfBatchedStorage
            self._active_agents = np.array([0, 0])
        else:
            policy_cls = MultiPolicy
            updater_cls = MultiUpdater
            storage_cls = MultiStorage

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
            config=config,
            env_spec=agent_env_spec,
            is_distrib=is_distrib,
            device=device,
            num_envs=num_envs,
            percent_done_fn=percent_done_fn,
            resume_state=use_resume_state,
            lr_schedule_fn=lr_schedule_fn,
            agent_name=agent_name,
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
            agent.load_state_dict(state[str(agent_i)])

    def load_ckpt_state_dict(self, ckpt):
        for agent in self._agents:
            agent.load_ckpt_state_dict(ckpt)

    @property
    def masks_shape(self) -> Tuple:
        return (
            sum(self._agents[i].masks_shape[0] for i in self._active_agents),
        )

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
