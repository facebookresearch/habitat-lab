#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ddppo.algo.ddp_utils import rank0_only
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import cast_to_float_if_half

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
        fp16_autocast: bool = False,
        fp16_mixed: bool = False,
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

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

        self.grad_scaler = (
            torch.cuda.amp.GradScaler()
            if fp16_autocast or fp16_mixed
            else None
        )
        self._fp16_autocast = fp16_autocast
        self._fp16_mixed = fp16_mixed

        self._consecutive_steps_with_scale_reduce = 0
        self._prev_grad_scale = (
            self.grad_scaler.get_scale()
            if self.grad_scaler is not None
            else None
        )

        self.fp16_optim_params = (
            FP16OptimParamManager(self.optimizer) if fp16_mixed else None
        )

    def optim_state_dict(self):
        return dict(
            optimizer=self.optimizer.state_dict(),
            grad_scaler=self.grad_scaler.state_dict()
            if self.grad_scaler is not None
            else None,
        )

    def load_optim_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
            self._prev_grad_scale = self.grad_scaler.get_scale()

        if self.fp16_optim_params is not None:
            self.fp16_optim_params.set_fp32_params_from_optim()

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1]
            - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                with torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                    ) = self._evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                    )

                    if self._fp16_mixed:
                        values = values.float()
                        action_log_probs = action_log_probs.float()
                        dist_entropy = dist_entropy.float()

                    ratio = torch.exp(
                        action_log_probs - batch["action_log_probs"]
                    )
                    surr1 = ratio * batch["advantages"]
                    surr2 = (
                        torch.clamp(
                            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                        )
                        * batch["advantages"]
                    )
                    action_loss = -(torch.min(surr1, surr2).mean())

                    if self.use_clipped_value_loss:
                        value_pred_clipped = batch["value_preds"] + (
                            values - batch["value_preds"]
                        ).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - batch["returns"]).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - batch["returns"]
                        ).pow(2)
                        value_loss = 0.5 * torch.max(
                            value_losses, value_losses_clipped
                        )
                    else:
                        value_loss = 0.5 * (batch["returns"] - values).pow(2)

                    value_loss = value_loss.mean()
                    dist_entropy = dist_entropy.mean()

                    total_loss = (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - dist_entropy * self.entropy_coef
                    )

                self.before_backward(total_loss)
                if self.grad_scaler is not None:
                    total_loss = self.grad_scaler.scale(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                if self.grad_scaler is None:
                    self.optimizer.step()
                else:
                    self.grad_scaler.step(self.optimizer)
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritence
        """
        return self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        if self.fp16_optim_params is not None:
            self.fp16_optim_params.sync_grads()
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            (
                param
                for pg in self.optimizer.param_groups
                for param in pg["params"]
            ),
            self.max_grad_norm,
        )

    def after_step(self) -> None:
        if self.fp16_optim_params is not None:
            self.fp16_optim_params.sync_params()
            self.fp16_optim_params.clear_grads()
        else:
            self.optimizer.zero_grad()

        if self.grad_scaler is not None:
            self.grad_scaler.update()
            new_scale = self.grad_scaler.get_scale()
            if new_scale < self._prev_grad_scale:
                self._consecutive_steps_with_scale_reduce += 1
            else:
                self._consecutive_steps_with_scale_reduce = 0

            self._prev_grad_scale = new_scale
            if self._consecutive_steps_with_scale_reduce > 2 and rank0_only():
                logger.warn(
                    "Greater than 2 steps with scale reduction."
                    "  This typically indicates that fp16 training is unstable."
                    "  Consider switching from mixed to autocast or autocast to off"
                )
            if new_scale < 1.0 and rank0_only():
                logger.warn(
                    "Grad scale less than 1."
                    "  This typically indicates that fp16 training is unstable."
                    "  Consider switching from mixed to autocast or autocast to off"
                )


class FP16OptimParamManager:
    r"""Manages a second set of parameters in fp32 for the optimizer when the model
    is in fp16.  This works by taking an optimizer that currently has the model's fp16
    parameters and replacing them with fp32 versions.  It then has three calls: sync_grads()
    copies the gradients of the fp16 weights onto the fp32 copies (this should ideally be done
    before unscaling for maintaining the most amount of precisions), sync_params() copies the fp32
    params onto the fp16 params.  clear_grads() clears the gradients of both the fp16 and fp32 params.

    General usage:

    .. code:: py

        grad_scaler.scale(loss).backward()

        fp16_optim_params.sync_grads()

        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)

        fp16_optim_params.sync_params()
        fp16_optim_params.clear_grads()


    Note that after loading the optimizer's state dict with load_state_dict(),
    set_fp32_params_from_optim() must be called!
    """

    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer

        self._fp16_params = []
        self._fp32_params = []

        for pg in optimizer.param_groups:
            new_fp32_params = []
            self._fp16_params.append(pg["params"])
            for param in pg["params"]:
                fp32_param = cast_to_float_if_half(param)

                new_fp32_params.append(fp32_param)

            pg["params"] = new_fp32_params
            self._fp32_params.append(new_fp32_params)

    def _apply_fn(self, function):
        for fp32_pg, fp16_pg in zip(self._fp32_params, self._fp16_params):
            for fp32_p, fp16_p in zip(fp32_pg, fp16_pg):
                function(fp32_p, fp16_p)

    def sync_grads(self):
        def _sync_grad_fn(fp32_p, fp16_p):
            if fp16_p.grad is not None:
                fp32_p.grad = cast_to_float_if_half(fp16_p.grad)

        self._apply_fn(_sync_grad_fn)

    def sync_params(self):
        self._apply_fn(lambda fp32_p, fp16_p: fp16_p.data.copy_(fp32_p.data))

    def set_fp32_params_from_optim(self):
        self._fp32_params = [
            pg["params"] for pg in self.optimizer.param_groups
        ]
        self.sync_params()

    def clear_grads(self):
        def _set_none(fp32_p, fp16_p):
            fp32_p.grad = None
            fp16_p.grad = None

        self._apply_fn(_set_none)
