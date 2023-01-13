from typing import Any, Dict, Optional

import numpy as np
import torch
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.multi_agent.agent_data import AgentData
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import (
    get_num_actions,
    is_continuous_action_space,
)


@baseline_registry.register_agent_sampler()
class AgentSampler:
    def __init__(
        self,
        config,
        agent_sampler_cfg,
        obs_space,
        action_space,
        orig_action_space,
        is_distrib,
    ):
        self._is_distributed = is_distrib
        self.device = None
        self._agents = {}
        self.config = config
        self.action_space = action_space
        self.obs_space = obs_space
        is_static_encoder = not config.habitat_baselines.rl.ddppo.train_encoder
        ppo_cfg = self.config.habitat_baselines.rl.ppo
        for agent_name in agent_sampler_cfg.agents:
            actor_critic = self._create_actor_critic(
                config, obs_space, action_space, orig_action_space
            )
            self._load_weights(config, actor_critic)

            if is_static_encoder:
                for param in actor_critic.net.visual_encoder.parameters():
                    param.requires_grad_(False)

            if config.habitat_baselines.rl.ddppo.reset_critic:
                nn.init.orthogonal_(actor_critic.critic.fc.weight)
                nn.init.constant_(actor_critic.critic.fc.bias, 0)
            updater = (DDPPO if self._is_distributed else PPO).from_config(
                actor_critic, config.habitat_baselines.rl.ppo
            )
            self._agents[agent_name] = AgentData(
                actor_critic=actor_critic,
                updater=updater,
                is_static_encoder=is_static_encoder,
                ppo_cfg=ppo_cfg,
            )

    def init(
        self,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done: float,
    ) -> None:
        ppo_cfg = self.config.habitat_baselines.rl.ppo

        if is_continuous_action_space(self.action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(self.action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        for agent in self._agents.values():
            if resume_state is not None:
                agent.updater.load_state_dict(resume_state["state_dict"])
                agent.updater.optimizer.load_state_dict(
                    resume_state["optim_state"]
                )
            if self._is_distributed:
                agent.updater.init_distributed(find_unused_params=False)  # type: ignore

            logger.info(
                "agent number of parameters: {}".format(
                    sum(param.numel() for param in agent.updater.parameters())
                )
            )

            obs_space = self.obs_space
            if agent.is_static_encoder:
                obs_space = spaces.Dict(
                    {
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                            low=np.finfo(np.float32).min,
                            high=np.finfo(np.float32).max,
                            shape=agent.encoder.output_shape,
                            dtype=np.float32,
                        ),
                        **obs_space.spaces,
                    }
                )

            rollouts = RolloutStorage(
                ppo_cfg.num_steps,
                num_envs,
                obs_space,
                self.action_space,
                ppo_cfg.hidden_size,
                num_recurrent_layers=agent.actor_critic.num_recurrent_layers,
                is_double_buffered=ppo_cfg.use_double_buffered_sampler,
                action_shape=action_shape,
                discrete_actions=discrete_actions,
            )
            rollouts.to(self.device)
            lr_sched = None
            if agent.updater is not None:
                lr_sched = LambdaLR(
                    optimizer=agent.updater.optimizer,
                    lr_lambda=lambda x: 1 - percent_done,
                )
            agent.set_post_init_data(
                rollouts, discrete_actions, action_shape, lr_sched
            )

    def _load_weights(self, config, actor_critic):
        if (
            config.habitat_baselines.rl.ddppo.pretrained_encoder
            or config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if config.habitat_baselines.rl.ddppo.pretrained:
            actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

    def sample_agents(self):
        return self._agents

    def _create_actor_critic(
        self, config, obs_space, action_space, orig_action_space
    ):
        policy = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.name
        )
        actor_critic = policy.from_config(
            config,
            obs_space,
            action_space,
            orig_action_space=orig_action_space,
        )
        return actor_critic

    def to(self, device):
        self.device = device
        for agent in self._agents:
            agent.actor_critic.to(device)

    def get_resume_state(self) -> Dict[str, Any]:
        return dict(
            state_dict=[
                agent.actor_critic.state_dict() for agent in self._agents
            ],
            optim_state=[
                agent.updater.optimizer.state_dict()
                for agent in self._agents
                if agent.should_update
            ],
            lr_sched_state=[
                agent.lr_scheduler.state_dict()
                for agent in self._agents
                if agent.should_update
            ],
        )

    def get_save_state(self) -> Dict:
        return dict(
            state_dict=[
                agent.actor_critic.state_dict() for agent in self._agents
            ]
        )

    def load_ckpt_state_dict(self, ckpt: Dict) -> None:
        """
        Loads a state dict for evaluation. The difference from
        `load_state_dict` is that this will not load the policy state if the
        policy does not request it.
        """
        for agent in self._agents:
            if agent.actor_critic.should_load_agent_state:
                agent.actor_critic.load_state_dict(ckpt["state_dict"])

    def load_state_dict(self, state: Dict) -> None:
        updater_i = 0
        for agent_i, agent in enumerate(self._agents):
            agent.actor_critic.load_state_dict(state["state_dict"][agent_i])
            if agent.updater is not None:
                if "optim_state" in state:
                    agent.actor_critic.load_state_dict(
                        state["optim_state"][updater_i]
                    )
                if "lr_sched_state" in state:
                    agent.actor_critic.load_state_dict(
                        state["lr_sched_state"][updater_i]
                    )
                updater_i += 1

    def from_config(
        self,
        config,
        agent_sampler_cfg,
        obs_space,
        action_space,
        orig_action_space,
        is_distrib,
    ):
        return AgentSampler(
            config,
            agent_sampler_cfg,
            obs_space,
            action_space,
            orig_action_space,
            is_distrib,
        )
