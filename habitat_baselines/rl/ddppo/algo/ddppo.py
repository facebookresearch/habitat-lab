#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import distributed as distrib

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo import PPO

EPS_PPO = 1e-5


def distributed_mean_and_var(
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the mean and variances of a tensor over multiple workers.

    This method is equivalent to first collecting all versions of values and
    then computing the mean and variance locally over that

    :param values: (*,) shaped tensors to compute mean and variance over.  Assumed
                        to be solely the workers local copy of this tensor,
                        the resultant mean and variance will be computed
                        over _all_ workers version of this tensor.
    """
    assert distrib.is_initialized(), "Distributed must be initialized"

    world_size = distrib.get_world_size()

    mean = values.mean()
    distrib.all_reduce(mean)
    mean = mean / world_size

    var = (values - mean).pow(2).mean()
    distrib.all_reduce(var)
    var = var / world_size

    return mean, var


class _EvalActionsWrapper(torch.nn.Module):
    r"""Wrapper on evaluate_actions that allows that to be called from forward.
    This is needed to interface with DistributedDataParallel's forward call
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, *args, **kwargs):
        return self.actor_critic.evaluate_actions(*args, **kwargs)


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(
        self, rollouts: RolloutStorage
    ) -> torch.Tensor:
        advantages = (
            rollouts.buffers["returns"][: rollouts.current_rollout_step_idx]  # type: ignore
            - rollouts.buffers["value_preds"][
                : rollouts.current_rollout_step_idx
            ]
        )
        if not self.use_normalized_advantage:  # type: ignore
            return advantages

        mean, var = distributed_mean_and_var(advantages)

        return (advantages - mean) / (var.sqrt() + EPS_PPO)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:  # noqa: SIM119
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(  # type: ignore
                        model,
                        device_ids=[device],
                        output_device=device,
                        find_unused_parameters=find_unused_params,
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(  # type: ignore
                        model,
                        find_unused_parameters=find_unused_params,
                    )

        self._evaluate_actions_wrapper = Guard(_EvalActionsWrapper(self.actor_critic), self.device)  # type: ignore

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self._evaluate_actions_wrapper.ddp(
            observations, rnn_hidden_states, prev_actions, masks, action
        )


class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
