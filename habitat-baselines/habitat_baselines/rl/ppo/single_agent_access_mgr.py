from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

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
from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ppo.updater import Updater

if TYPE_CHECKING:
    from omegaconf import DictConfig


def linear_lr_schedule(percent_done: float) -> float:
    return 1 - percent_done


@baseline_registry.register_agent_access_mgr
class SingleAgentAccessMgr(AgentAccessMgr):
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
        agent_name=None,
    ):
        """
        :param percent_done_fn: Function that will return the percent of the
            way through training.
        :param lr_schedule_fn: For a learning rate schedule. ONLY used if
            specified in the config. Takes as input the current progress in
            training and returns the learning rate multiplier. The default behavior
            is to use `linear_lr_schedule`.
        :param agent_name: the name of the agent for which we set the singleagentaccessmanager
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

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you should specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        self.agent_name = agent_name
        self._nbuffers = 2 if self._ppo_cfg.use_double_buffered_sampler else 1
        self._percent_done_fn = percent_done_fn
        if lr_schedule_fn is None:
            lr_schedule_fn = linear_lr_schedule
        self._init_policy_and_updater(lr_schedule_fn, resume_state)

    def _init_policy_and_updater(self, lr_schedule_fn, resume_state):
        self._actor_critic = self._create_policy()
        self._updater = self._create_updater(self._actor_critic)

        if self._updater.optimizer is None:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = LambdaLR(
                optimizer=self._updater.optimizer,
                lr_lambda=lambda _: lr_schedule_fn(self._percent_done_fn()),
            )
        if resume_state is not None:
            self._updater.load_state_dict(resume_state["state_dict"])
            self._updater.load_state_dict(
                {
                    "actor_critic." + k: v
                    for k, v, in resume_state["state_dict"].items()
                }
            )

    @property
    def masks_shape(self) -> Tuple:
        return (1,)

    @property
    def nbuffers(self):
        return self._nbuffers

    def _create_storage(
        self,
        num_envs: int,
        env_spec: EnvironmentSpec,
        actor_critic: NetPolicy,
        policy_action_space: spaces.Space,
        config: "DictConfig",
        device,
    ) -> Storage:
        """
        Default behavior for setting up and initializing the rollout storage.
        """

        obs_space = get_rollout_obs_space(
            env_spec.observation_space, actor_critic, config
        )
        ppo_cfg = config.habitat_baselines.rl.ppo
        rollouts = baseline_registry.get_storage(
            config.habitat_baselines.rollout_storage_name
        )(
            numsteps=ppo_cfg.num_steps,
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=policy_action_space,
            actor_critic=actor_critic,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
        )
        rollouts.to(device)
        return rollouts

    def post_init(self, create_rollouts_fn: Optional[Callable] = None) -> None:
        # Create the rollouts storage.
        if create_rollouts_fn is None:
            create_rollouts_fn = self._create_storage

        policy_action_space = self._actor_critic.policy_action_space
        self._rollouts = create_rollouts_fn(
            num_envs=self._num_envs,
            env_spec=self._env_spec,
            actor_critic=self._actor_critic,
            policy_action_space=policy_action_space,
            config=self._config,
            device=self._device,
        )

    def _create_updater(self, actor_critic) -> PPO:
        if self._is_distributed:
            updater_cls = baseline_registry.get_updater(
                self._config.habitat_baselines.distrib_updater_name
            )
        else:
            updater_cls = baseline_registry.get_updater(
                self._config.habitat_baselines.updater_name
            )

        updater = updater_cls.from_config(actor_critic, self._ppo_cfg)
        logger.info(
            "Agent number of parameters: {}".format(
                sum(param.numel() for param in updater.parameters())
            )
        )
        return updater

    def init_distributed(self, find_unused_params: bool = True) -> None:
        if len(list(self._updater.parameters())) > 0:
            self._updater.init_distributed(
                find_unused_params=find_unused_params
            )

    def _create_policy(self) -> NetPolicy:
        """
        Creates and initializes the policy. This should also load any model weights from checkpoints.
        """

        policy = baseline_registry.get_policy(
            self._config.habitat_baselines.rl.policy[self.agent_name].name
        )
        if policy is None:
            raise ValueError(
                f"Couldn't find policy {self._config.habitat_baselines.rl.policy[self.agent_name].name}"
            )
        actor_critic = policy.from_config(
            self._config,
            self._env_spec.observation_space,
            self._env_spec.action_space,
            orig_action_space=self._env_spec.orig_action_space,
            agent_name=self.agent_name,
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
            for param in actor_critic.visual_encoder.parameters():
                param.requires_grad_(False)

        if self._config.habitat_baselines.rl.ddppo.reset_critic:
            nn.init.orthogonal_(actor_critic.critic.fc.weight)
            nn.init.constant_(actor_critic.critic.fc.bias, 0)

        actor_critic.to(self._device)
        return actor_critic

    @property
    def rollouts(self) -> Storage:
        return self._rollouts

    @property
    def actor_critic(self) -> NetPolicy:
        return self._actor_critic

    @property
    def updater(self) -> Updater:
        return self._updater

    def get_resume_state(self) -> Dict[str, Any]:
        # If there is nothing to load, then we return the empty dict
        if self._updater.optimizer is None:
            return {"state_dict": {}, "optim_state": {}}
        ret = {
            "state_dict": self._actor_critic.state_dict(),
            **self._updater.get_resume_state(),
        }
        if self._lr_scheduler is not None:
            ret["lr_sched_state"] = self._lr_scheduler.state_dict()
        return ret

    def get_save_state(self):
        return {"state_dict": self._actor_critic.state_dict()}

    def eval(self):
        self._actor_critic.eval()

    def train(self):
        self._actor_critic.train()
        self._updater.train()

    def load_ckpt_state_dict(self, ckpt: Dict) -> None:
        self._actor_critic.load_state_dict(ckpt["state_dict"])

    def load_state_dict(self, state: Dict) -> None:
        self._actor_critic.load_state_dict(state["state_dict"])
        if self._updater is not None:
            self._updater.load_state_dict(state)
            if "lr_sched_state" in state:
                self._lr_scheduler.load_state_dict(state["lr_sched_state"])

    def after_update(self):
        if (
            self._ppo_cfg.use_linear_lr_decay
            and self._lr_scheduler is not None
        ):
            self._lr_scheduler.step()  # type: ignore
        self._updater.after_update()

    def pre_rollout(self):
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
        encoder = actor_critic.visual_encoder
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
