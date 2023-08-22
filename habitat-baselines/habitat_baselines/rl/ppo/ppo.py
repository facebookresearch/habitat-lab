#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.updater import Updater
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)
from habitat_baselines.utils.timing import g_timer

EPS_PPO = 1e-5


@baseline_registry.register_updater
class PPO(nn.Module, Updater):
    entropy_coef: Union[float, LagrangeInequalityCoefficient]

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
        )

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
    ) -> None:
        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = next(actor_critic.parameters()).device

        if (
            use_adaptive_entropy_pen
            and hasattr(self.actor_critic, "num_actions")
            and getattr(self.actor_critic, "action_distribution_type", None)
            == "gaussian"
        ):
            num_actions = self.actor_critic.num_actions

            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor) * num_actions,
                init_alpha=entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)

        self.use_normalized_advantage = use_normalized_advantage
        self.optimizer = self._create_optimizer(lr, eps)

        self.non_ac_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("actor_critic.")
        ]

    def _create_optimizer(self, lr, eps):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        logger.info(
            f"Number of params to train: {sum(param.numel() for param in params)}"
        )
        if len(params) > 0:
            optim_cls = optim.Adam
            optim_kwargs = dict(
                params=params,
                lr=lr,
                eps=eps,
            )
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            return optim_cls(**optim_kwargs)
        else:
            return None

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"]  # type: ignore
            - rollouts.buffers["value_preds"]
        )
        if not self.use_normalized_advantage:
            return advantages

        var, mean = self._compute_var_mean(
            advantages[torch.isfinite(advantages)]
        )

        advantages -= mean

        return advantages.mul_(torch.rsqrt(var + EPS_PPO))

    @staticmethod
    def _compute_var_mean(x):
        return torch.var_mean(x)

    def _set_grads_to_none(self):
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                p.grad = None

    @g_timer.avg_time("ppo.update_from_batch", level=1)
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        Performs a gradient update from the minibatch.
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()

        (
            values,
            action_log_probs,
            dist_entropy,
            _,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["recurrent_hidden_states"],
            batch["prev_actions"],
            batch["masks"],
            batch["actions"],
            batch.get("rnn_build_seq_info", None),
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )

        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
            mean_fn = lambda t: torch.mean(ver_is_coeffs * t)
        else:
            mean_fn = torch.mean

        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )

            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (
                        rollouts.current_policy_version
                        - batch["policy_version"]
                    ).float(),
                    "policy_version_difference",
                )

    def update(
        self,
        rollouts: RolloutStorage,
    ) -> Dict[str, float]:
        advantages = self.get_advantages(rollouts)

        learner_metrics: Dict[str, List[Any]] = collections.defaultdict(list)

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.data_generator(
                advantages, self.num_mini_batch
            )

            for _bid, batch in enumerate(data_generator):
                self._update_from_batch(
                    batch, epoch, rollouts, learner_metrics
                )

            profiling_wrapper.range_pop()  # PPO.update epoch

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    @g_timer.avg_time("ppo.eval_actions", level=1)
    def _evaluate_actions(self, *args, **kwargs):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(*args, **kwargs)

    def before_backward(self, loss: Tensor) -> Tensor:
        return loss

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> torch.Tensor:
        handles = []
        if torch.distributed.is_initialized():
            for p in self.non_ac_params:
                if p.grad is not None:
                    p.grad.data.detach().div_(
                        torch.distributed.get_world_size()
                    )
                    handles.append(
                        torch.distributed.all_reduce(
                            p.grad.data.detach(), async_op=True
                        )
                    )

        grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.policy_parameters(),
            self.max_grad_norm,
        )

        for v in self.actor_critic.aux_loss_parameters().values():
            nn.utils.clip_grad_norm_(v, self.max_grad_norm)

        [h.wait() for h in handles]

        return grad_norm

    def after_step(self) -> None:
        if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
            self.entropy_coef.project_into_bounds()

    def get_resume_state(self):
        return {
            "optim_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        if "optim_state" in state:
            self.optimizer.load_state_dict(state["optim_state"])
