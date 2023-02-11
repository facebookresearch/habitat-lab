from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import (  # noqa: F401.
    RolloutStorage,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.rl.ppo import PPO  # noqa: F401.
from habitat_baselines.rl.ppo.policy import Policy

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TrainerAgent:
    def __init__(
        self,
        config: "DictConfig",
        obs_space: spaces.Space,
        action_space: spaces.Space,
        orig_action_space: spaces.Space,
        is_distrib: bool,
        device,
    ):
        self._is_distributed = is_distrib
        self._is_static_encoder = (
            not config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._actor_critic = self._create_actor()
        self._actor_critic.to(device)
        self._device = device
        self._policy_action_space = self._actor_critic.get_policy_action_space(
            action_space
        )
        self._updater = self._create_updater()
        self.config = config
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        self._rollouts: Optional[RolloutStorage] = None
        self._lr_scheduler: Optional[LambdaLR] = None
        self.action_shape: Optional[Tuple[int]] = None

    def initialize(
        self,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done: float,
    ):
        if resume_state is not None:
            self._updater.load_state_dict(resume_state["state_dict"])
            self._updater.optimizer.load_state_dict(
                resume_state["optim_state"]
            )
        if self._is_distributed:
            self._updater.init_distributed(find_unused_params=False)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self._updater.parameters())
            )
        )

        if self._is_static_encoder:
            obs_space = spaces.Dict(
                {
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self.encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self.nbuffers = 2 if self._ppo_cfg.use_double_buffered_sampler else 1
        rollouts_cls = baseline_registry.get_storage(
            self.config.habitat_baselines.rollout_storage_name
        )
        self.rollouts = rollouts_cls(
            self._ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            self._ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.num_recurrent_layers,
            is_double_buffered=self._ppo_cfg.use_double_buffered_sampler,
        )
        self.rollouts.to(self.device)

        self._lr_sched = None
        self._lr_sched = LambdaLR(
            optimizer=self._updater.optimizer,
            lr_lambda=lambda x: 1 - percent_done,
        )
        if self._ppo_cfg.use_linear_clip_decay:
            self._pudater.clip_param = self._ppo_cfg.clip_param * (
                1 - percent_done
            )

    def _create_updater(self, config, actor_critic):
        if self._is_distributed:
            updater_cls = baseline_registry.get_updater(
                config.habitat_baselines.distrib_updater_name
            )
        else:
            updater_cls = baseline_registry.get_updater(
                config.habitat_baselines.updater_name
            )
        return updater_cls.from_config(
            actor_critic, config.habitat_baselines.rl.ppo
        )

    def _create_actor(
        self, config, obs_space, action_space, orig_action_space
    ) -> Policy:
        policy = baseline_registry.get_policy(
            config.habitat_baselines.rl.policy.name
        )
        actor_critic = policy.from_config(
            config,
            obs_space,
            action_space,
            orig_action_space=orig_action_space,
        )
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
        if self._is_static_encoder:
            for param in actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        return actor_critic

    @property
    def encoder(self):
        return self.actor_critic.net.visual_encoder

    @property
    def rollouts(self) -> RolloutStorage:
        if self._rollouts is not None:
            raise ValueError("Rollout storage is not set.")
        return self._rollouts

    @property
    def actor_critic(self) -> Policy:
        return self._actor_critic

    @property
    def lr_scheduler(self) -> LambdaLR:
        return self._lr_scheduler

    @property
    def should_update(self) -> bool:
        return self._updater is not None

    @property
    def is_static_encoder(self) -> bool:
        return self._is_static_encoder

    @property
    def updater(self) -> Optional[PPO]:
        """
        The updater to update this policy. If None, then the policy should not
        be updated.
        """

        return self._updater

    def post_step(self):
        if self.should_update and self._ppo_cfg.use_linear_lr_decay:
            self._lr_scheduler.step()  # type: ignore

    def from_config(
        self,
        config,
        agent_sampler_cfg,
        obs_space,
        action_space,
        orig_action_space,
        is_distrib,
    ):
        return TrainerAgent(
            config,
            agent_sampler_cfg,
            obs_space,
            action_space,
            orig_action_space,
            is_distrib,
        )

    def get_resume_state(self) -> Dict[str, Any]:
        return dict(
            state_dict=self._actor_critic.state_dict(),
            optim_state=self._updater.optimizer.state_dict(),
            lr_sched_state=self._lr_scheduler.state_dict(),
        )

    def get_save_state(self) -> Dict:
        return dict(state_dict=self.actor_critic.state_dict())

    def eval(self):
        self.actor_critic.eval()

    def train(self):
        self.actor_critic.train()
        if self.updater is not None:
            self.updater.train()

    def load_ckpt_state_dict(self, ckpt: Dict) -> None:
        """
        Loads a state dict for evaluation. The difference from
        `load_state_dict` is that this will not load the policy state if the
        policy does not request it.
        """
        self.actor_critic.load_state_dict(ckpt["state_dict"])

    def load_state_dict(self, state: Dict) -> None:
        self._actor_critic.load_state_dict(state["state_dict"])
        if self._updater is not None:
            if "optim_state" in state:
                self._actor_critic.load_state_dict(state["optim_state"])
            if "lr_sched_state" in state:
                self._actor_critic.load_state_dict(state["lr_sched_state"])
