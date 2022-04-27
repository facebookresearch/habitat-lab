#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy

use_mixed_precision = False

from torch.cuda.amp import GradScaler, autocast


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


EPS_PPO = 1e-5


def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    norm_type: float = 2.0,
) -> float:
    r"""
    Calculates the norm of the gradient
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]
        ),
        norm_type,
    )
    return total_norm.item()


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

        for p in actor_critic.parameters():
            if not p.is_cuda:
                print("error: param with shape ", p.shape, " is not cuda")
                exit(0)

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

        if use_mixed_precision:
            self.grad_scaler = GradScaler()

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

    def update(
        self, rollouts: RolloutStorage
    ) -> Tuple[float, float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        grad_norm_epoch = 0.0

        for _e in range(self.ppo_epoch):
            # profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                precision_context = (
                    autocast if use_mixed_precision else dummy_context_mgr
                )
                with precision_context():
                    profiling_wrapper.range_push("PPO mini batch")
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                    ) = self._evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        torch.logical_and(
                            batch["not_done_mask_0"], batch["not_done_mask_1"]
                        ),
                        batch["actions"],
                    )

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
                    action_loss = -torch.min(surr1, surr2)

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

                    # Mask the loss. The mask corresponds to the transition right
                    # after a reset. This first observation needs to be ignored
                    # because its observation is actually part of the previous episode
                    action_loss = action_loss * batch["not_done_mask_0"]
                    value_loss = value_loss * batch["not_done_mask_0"]
                    dist_entropy = dist_entropy * batch["not_done_mask_0"]

                    action_loss = action_loss.mean()
                    value_loss = value_loss.mean()
                    dist_entropy = dist_entropy.mean()

                    self.optimizer.zero_grad()
                    total_loss = (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - dist_entropy * self.entropy_coef
                    )

                self.before_backward(total_loss)
                profiling_wrapper.range_push("backward")
                if use_mixed_precision:
                    self.grad_scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                profiling_wrapper.range_pop()
                self.after_backward(total_loss)

                self.before_step()

                grad_norm_epoch += get_grad_norm(
                    self.actor_critic.parameters()
                )

                profiling_wrapper.range_push("optimizer.step")
                if use_mixed_precision:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                profiling_wrapper.range_pop()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                profiling_wrapper.range_pop()

            # profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates

        return (
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            grad_norm_epoch,
        )

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
