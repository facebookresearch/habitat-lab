from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
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
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import inference_mode

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_agent
class TrainerAgent:
    """
    A `TrainerAgent` consists of:
    - Policy: How actions are selected from observations.
    - Data Storage: How data collected from the environment is stored.
    - Updater: How the Policy is updated.
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
        """
        :param percent_done_fn: Function that will return the percent of the
            way through training.
        :param lr_schedule_fn: For a learning rate schedule. ONLY used if
            specified in the config. Takes as input the current progress in
            training and returns the learning rate multiplier. The default behavior
            is to use `linear_lr_schedule`.
        """

        self._env_spec = env_spec
        self._config = config
        self._num_envs = num_envs
        self._device = device
        self._ppo_cfg = self._config.habitat_baselines.rl.ppo
        self._is_distributed = is_distrib
        self._is_static_encoder = (
            not config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._actor_critic = self._create_policy()
        self._actor_critic.to(self._device)
        self._policy_action_space = self._actor_critic.get_policy_action_space(
            env_spec.action_space
        )
        self._updater = self._create_updater(self._actor_critic)
        if resume_state is not None:
            self._updater.load_state_dict(resume_state["state_dict"])
            self._updater.optimizer.load_state_dict(
                resume_state["optim_state"]
            )

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self._updater.parameters())
            )
        )
        self.nbuffers = 2 if self._ppo_cfg.use_double_buffered_sampler else 1
        if lr_schedule_fn is not None:
            lr_schedule_fn = linear_lr_schedule
        self._percent_done_fn = percent_done_fn

        self._lr_scheduler = LambdaLR(
            optimizer=self._updater.optimizer,
            lr_lambda=lambda x: linear_lr_schedule(
                percent_done=percent_done_fn()
            ),
        )

    def post_init(self, create_rollouts_fn: Optional[Callable] = None) -> None:
        """
        Called after the constructor. Sets up the rollout storage.

        :param create_rollouts_fn: Override behavior for creating the
            rollout storage. Default behavior for this and the call signature is
            `default_create_rollouts`.
        """
        if create_rollouts_fn is None:
            create_rollouts_fn = default_create_rollouts
        self._rollouts = create_rollouts_fn(
            num_envs=self._num_envs,
            env_spec=self._env_spec,
            actor_critic=self.actor_critic,
            policy_action_space=self.policy_action_space,
            config=self._config,
            device=self._device,
        )

    def _create_updater(self, actor_critic):
        """
        Setup and initialize the policy updater.
        """
        if self._is_distributed:
            updater_cls = baseline_registry.get_updater(
                self._config.habitat_baselines.distrib_updater_name
            )
        else:
            updater_cls = baseline_registry.get_updater(
                self._config.habitat_baselines.updater_name
            )
        return updater_cls.from_config(actor_critic, self._ppo_cfg)

    @property
    def policy_action_space(self):
        """
        The action space the policy acts in. This can be different from the environment action space for hierarchical policies.
        """
        return self._policy_action_space

    def _create_policy(self) -> Policy:
        """
        Creates and initializes the policy. This should also load any model weights from checkpoints.
        """

        policy = baseline_registry.get_policy(
            self._config.habitat_baselines.rl.policy.name
        )
        actor_critic = policy.from_config(
            self._config,
            self._env_spec.observation_space,
            self._env_spec.action_space,
            orig_action_space=self._env_spec.orig_action_space,
        )
        if (
            self._config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self._config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self._config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if self._config.habitat_baselines.rl.ddppo.pretrained:
            actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self._config.habitat_baselines.rl.ddppo.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )
        if (
            self._config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self._config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self._config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
            )

        if self._config.habitat_baselines.rl.ddppo.pretrained:
            actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self._config.habitat_baselines.rl.ddppo.pretrained_encoder:
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
    def rollouts(self) -> RolloutStorage:
        return self._rollouts

    @property
    def actor_critic(self) -> Policy:
        return self._actor_critic

    @property
    def updater(self) -> PPO:
        return self._updater

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

    @property
    def hidden_state_shape(self):
        """
        The shape of the tensor to track the hidden state, such as the RNN hidden state.
        """

        return (
            self._agent.actor_critic.num_recurrent_layers,
            self._ppo_cfg.hidden_size,
        )

    def after_update(self):
        """
        Called after the updater has called `update` and the rollout `after_update` is called.
        """

        if self._ppo_cfg.use_linear_lr_decay:
            self._lr_scheduler.step()  # type: ignore

    def pre_rollout(self):
        """
        Called before a rollout is collected.
        """
        if self._ppo_cfg.use_linear_clip_decay:
            self._updater.clip_param = self._ppo_cfg.clip_param * (
                1 - self._percent_done_fn()
            )


def get_rollout_obs_space(obs_space, actor_critic, config):
    """
    Helper to get the observation space for the rollout storage when using a
    frozen visual encoder.
    """

    if not config.habitat_baselines.rl.ddppo.train_encoder:
        encoder = actor_critic.net.visual_encoder
        obs_space = spaces.Dict(
            {
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=encoder.output_shape,
                    dtype=np.float32,
                ),
                **obs_space.spaces,
            }
        )
    return obs_space


def default_create_rollouts(
    num_envs: int,
    env_spec: EnvironmentSpec,
    actor_critic: Policy,
    policy_action_space: spaces.Space,
    config: "DictConfig",
    device,
) -> RolloutStorage:
    """
    Default behavior for setting up and initializing the rollout storage.
    """

    obs_space = get_rollout_obs_space(
        env_spec.observation_space, actor_critic, config
    )
    ppo_cfg = config.habitat_baselines.rl.ppo
    return baseline_registry.get_storage(
        config.habitat_baselines.rollout_storage_name
    )(
        ppo_cfg.num_steps,
        num_envs,
        obs_space,
        policy_action_space,
        ppo_cfg.hidden_size,
        num_recurrent_layers=actor_critic.num_recurrent_layers,
        is_double_buffered=ppo_cfg.use_double_buffered_sampler,
    )


def linear_lr_schedule(percent_done: float) -> float:
    return 1 - percent_done
