from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
    An agent consists of:
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
        percent_done: float,
    ):
        self._env_spec = env_spec
        self._config = config
        self._ppo_cfg = self._config.habitat_baselines.rl.ppo
        self._is_distributed = is_distrib
        self._is_static_encoder = (
            not config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._actor_critic = self._create_policy()
        self._actor_critic.to(device)
        self._policy_action_space = self._actor_critic.get_policy_action_space(
            env_spec.action_space
        )
        self._updater = self._create_updater(self._actor_critic)

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
        self.nbuffers = 2 if self._ppo_cfg.use_double_buffered_sampler else 1
        self._rollouts = self._create_rollouts(num_envs)
        self._rollouts.to(device)

        self._lr_sched = LambdaLR(
            optimizer=self._updater.optimizer,
            lr_lambda=lambda x: 1 - percent_done,
        )
        if self._ppo_cfg.use_linear_clip_decay:
            self._updater.clip_param = self._ppo_cfg.clip_param * (
                1 - percent_done
            )

    def _create_rollouts(self, num_envs: int) -> RolloutStorage:
        """
        Setup and initialize the rollout storage.
        """

        obs_space = self._env_spec.observation_space
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

        rollouts_cls = baseline_registry.get_storage(
            self._config.habitat_baselines.rollout_storage_name
        )
        return rollouts_cls(
            self._ppo_cfg.num_steps,
            num_envs,
            obs_space,
            self.policy_action_space,
            self._ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.num_recurrent_layers,
            is_double_buffered=self._ppo_cfg.use_double_buffered_sampler,
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
        """
        TODO: In the next PR accessing rollouts directly will be depricated.
        Right now it is still used to track the current observation, however,
        in the future, the trainer should be responsible for tracking the
        current step.
        """
        return self._rollouts

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done: float,
    ):
        return cls(
            config,
            env_spec,
            is_distrib,
            device,
            resume_state,
            num_envs,
            percent_done,
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

    def insert_first(self, batch) -> None:
        """
        Insert the first observation from the environment at the start of training into the rollout storage.
        """
        if self._is_static_encoder:
            self._add_visual_features(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        """
        Insert data into the rollout storage. By default this passes through to the insert operation from the `RolloutStorage`.
        """
        if next_observations is not None and self._is_static_encoder:
            self._add_visual_features(next_observations)

        self.rollouts.insert(
            next_observations=next_observations,
            next_recurrent_hidden_states=next_recurrent_hidden_states,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            next_masks=next_masks,
            buffer_index=buffer_index,
            **kwargs,
        )

    def update(self) -> Dict[str, float]:
        """
        Update the policy.
        """

        with inference_mode():
            step_batch = self._rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self._actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        self._agent.train()

        losses = self._agent.updater.update(self._agent.rollouts)
        self.rollouts.after_update()

        if self._ppo_cfg.use_linear_lr_decay:
            self._lr_scheduler.step()  # type: ignore
        return losses

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ) -> PolicyActionData:
        return self._actor_critic.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    @property
    def hidden_state_shape(self):
        """
        The shape of the tensor to track the hidden state, such as the RNN hidden state.
        """

        return (
            self._agent.actor_critic.num_recurrent_layers,
            self._ppo_cfg.hidden_size,
        )

    def get_extra(
        self, action_data: PolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        """
        Gets any extra information for logging from the policy.
        """

        return self._actor_critic.get_extra(action_data, infos, dones)

    def _add_visual_features(self, batch) -> None:
        """
        Modifies the observation batch to include the visual features in-place.
        """
        with inference_mode():
            batch[
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
            ] = self._encoder(batch)
