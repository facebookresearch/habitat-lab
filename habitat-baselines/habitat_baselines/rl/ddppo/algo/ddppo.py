#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import numpy as np
import torch
from torch import distributed as distrib

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import PPO


def _recursive_apply(inp, fn):
    if isinstance(inp, dict):
        return type(inp)((k, _recursive_apply(v, fn)) for k, v in inp.items())
    elif isinstance(inp, (tuple, list)):
        return type(inp)(_recursive_apply(v, fn) for v in inp)
    else:
        return fn(inp)


def _convert_to_numpy_safe(t: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Moves any tensors on the CPU to numpy arrays. Leaves other values unaffected.
    """

    if t is not None and t.device.type == "cpu":
        return t.numpy()
    return t


def _cpu_to_numpy(inp):
    return _recursive_apply(inp, _convert_to_numpy_safe)


def _numpy_to_cpu_safe(
    t: Optional[Union[np.ndarray, torch.Tensor]]
) -> Optional[torch.Tensor]:
    """
    Moves numpy arrays to torch CPU tensors.
    """

    if t is not None and isinstance(t, np.ndarray):
        return torch.from_numpy(t)
    return t


def _numpy_to_cpu(inp):
    return _recursive_apply(
        inp,
        _numpy_to_cpu_safe,
    )


def distributed_var_mean(
    values: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor]:
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

    return var, mean


class _EvalActionsWrapper(torch.nn.Module):
    r"""Wrapper on evaluate_actions that allows that to be called from forward.
    This is needed to interface with DistributedDataParallel's forward call
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, *args, **kwargs):
        # We then convert numpy arrays back to a CPU tensor.
        # This is needed for older versions of pytorch that haven't deprecated
        # the single-process multi-device version of DDP
        return self.actor_critic.evaluate_actions(
            *_numpy_to_cpu(args), **_numpy_to_cpu(kwargs)
        )


class DecentralizedDistributedMixin:
    @staticmethod
    def _compute_var_mean(x):
        return distributed_var_mean(x)

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
                if device.type == "cuda":
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

    def _evaluate_actions(self, *args, **kwargs):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        # DistributedDataParallel moves all tensors to the device (or devices)
        # So we need to make anything that is on the CPU into a numpy array
        # This is needed for older versions of pytorch that haven't deprecated
        # the single-process multi-device version of DDP
        return self._evaluate_actions_wrapper.ddp(
            *_cpu_to_numpy(args), **_cpu_to_numpy(kwargs)
        )


@baseline_registry.register_updater
class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
