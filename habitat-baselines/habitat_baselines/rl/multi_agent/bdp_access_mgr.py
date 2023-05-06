import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.multi_agent.multi_agent_access_mgr import (
    MultiAgentAccessMgr,
)
from habitat_baselines.rl.multi_agent.pop_play_wrappers import (
    MultiPolicy,
    MultiStorage,
    MultiUpdater,
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.rl.ppo.single_agent_access_mgr import (
    SingleAgentAccessMgr,
)

# coordination agent is the agent trained to coordinate with a diverse set of partners
COORD_AGENT = 0
# behavior policy is a latent conditioned policy that generates diverse behaviors when conditioned on different latents
BEHAV_AGENT = 1
COORD_AGENT_NAME = "agent_0"
BEHAV_AGENT_NAME = "agent_1"

ROBOT_TYPE = 0
HUMAN_TYPE = 1

BEHAV_ID = "behav_latent"


@baseline_registry.register_agent_access_mgr
class BdpAgentAccessMgr(MultiAgentAccessMgr):
    """
    Behavioral Diversity Play implementation. A behavior policy is trained to
    generate diverse behaviors through a diversity reward bonus. A coordination
    policy is trained to coordinate with the behavior policy.
    """

    def _sample_active_idxs(self):
        if (
            self._pop_config.self_play_batched
            or self._pop_config.num_agent_types != 2
            or self._pop_config.num_active_agents_per_type != [1, 1]
        ):
            raise ValueError("BDP only supports pop play with 2 agents")

        num_envs = self._agents[0]._num_envs
        device = self._agents[0]._device

        if self._pop_config.force_all_agents:
            if num_envs != self._pop_config.behavior_latent_dim:
                raise ValueError(
                    f"Must have num_envs={num_envs} equal to behavior latent dim={self._pop_config.behavior_latent_dim}"
                )
            behav_ids = torch.arange(
                start=0,
                end=self._pop_config.behavior_latent_dim,
                device=device,
            )
        else:
            behav_ids = torch.randint(
                0,
                self._pop_config.behavior_latent_dim,
                (num_envs,),
                device=device,
            )
        self._behav_latents = F.one_hot(
            behav_ids, self._pop_config.behavior_latent_dim
        ).float()

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

        hl_policy = self._agents[BEHAV_AGENT].actor_critic._high_level_policy
        discrim = hl_policy.aux_modules["bdp_discrim"]
        multi_storage = BdpStorage.from_config(
            config,
            env_spec.observation_space,
            env_spec.action_space,
            orig_action_space=env_spec.orig_action_space,
            agent=self._agents[0],
            n_agents=num_active_agents,
            update_obs_with_agent_prefix_fn=self._inject_behav_latent,
            discrim_reward_weight=self._pop_config.discrim_reward_weight,
            hl_policy=hl_policy,
            discrim=discrim,
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
        elif agent_name == COORD_AGENT_NAME:
            # Remove the discriminator from this policy.
            config = config.copy()
            with read_write(config):
                del config.habitat_baselines.rl.auxiliary_losses["bdp_discrim"]
        else:
            raise ValueError(f"Unexpected agent name {agent_name}")

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


@baseline_registry.register_auxiliary_loss(name="bdp_discrim")
class BehavDiscrim(nn.Module):
    """
    Defines the discriminator network and the training objective for the
    discriminator. Through the Habitat Baselines auxiliary loss registry, this
    is automatically added to the policy class and the loss is computed in the
    policy update.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        net: Net,
        loss_scale,
        hidden_size,
        behavior_latent_dim,
    ):
        super().__init__()

        input_dim = net._hidden_size
        self.discrim = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, behavior_latent_dim),
        )
        self.loss_scale = loss_scale

    def pred_logits(self, policy_features, obs):
        return self.discrim(policy_features)

    def forward(self, policy_features, obs):
        # Don't backprop into the policy representation.
        policy_features = policy_features.detach()
        pred_logits = self.pred_logits(policy_features, obs)
        behav_ids = torch.argmax(obs[BEHAV_ID], -1)
        loss = F.cross_entropy(pred_logits, behav_ids)
        return {"loss": loss * self.loss_scale}


class BdpStorage(MultiStorage):
    """
    Overrides the storage so we can inject the diversity reward computation
    before computing return.
    """

    def __init__(
        self, *args, discrim, hl_policy, discrim_reward_weight, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._discrim = discrim
        self._hl_policy = hl_policy
        self._discrim_reward_weight = discrim_reward_weight

    def compute_returns(self, next_value, use_gae, gamma, tau):
        """
        Adds a weighted diversity reward to the task reward in the behavior
        agent's rollout buffer. This overrides the existing rewards in the
        buffer. The buffer of the coordination agent is unmodified.
        """

        behav_storage = self._active_storages[BEHAV_AGENT]
        with torch.no_grad():
            masks = behav_storage.buffers["masks"]
            obs = behav_storage.buffers["observations"]
            features, _ = self._hl_policy(
                obs.map(lambda x: x.flatten(0, 1)),
                behav_storage.buffers["recurrent_hidden_states"].flatten(0, 1),
                masks.flatten(0, 1),
            )
            features = features.view(*masks.shape[:2], -1)
            pred_logits = self._discrim.pred_logits(features, obs)
            behav_ids = torch.argmax(obs[BEHAV_ID], -1).long()
            scores = F.log_softmax(pred_logits, -1)
            log_prob = scores.gather(-1, behav_ids.view(*masks.shape[:2], 1))
            behav_storage.buffers["rewards"] += (
                self._discrim_reward_weight * log_prob
            )
        super().compute_returns(next_value, use_gae, gamma, tau)
