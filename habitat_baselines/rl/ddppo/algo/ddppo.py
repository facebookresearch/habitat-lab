#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.distributed as distrib

from habitat_baselines.rl.ppo import PPO

EPS_PPO = 1e-5


def distributed_mean_and_var(values):
    world_size = distrib.get_world_size()
    mean = values.mean()
    distrib.all_reduce(mean)
    mean /= world_size

    sq_diff = (values - mean).pow(2).mean()
    distrib.all_reduce(sq_diff)
    var = sq_diff / world_size

    return mean, var


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        mean, var = distributed_mean_and_var(advantages)

        return (advantages - mean) / (var.sqrt() + EPS_PPO)

    def init_distributed(self, find_unused_params=False):
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard(object):
            def __init__(self, model, device):
                self.ddp = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[device], output_device=device
                )

        self._ddp_hooks = Guard(self.actor_critic, self.device)
        self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        super().before_backward(loss)

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


# Mixin goes second that way the PPO __init__ will still be called
class DDPPO(PPO, DecentralizedDistributedMixin):
    pass
