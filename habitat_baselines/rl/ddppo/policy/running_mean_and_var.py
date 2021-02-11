#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import Tensor
from torch import distributed as distrib
from torch import nn as nn


def _welford_update(mean, var, count, new_mean, new_var, new_count):
    r"""Running mean and var updated based on Welford's algorithm:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    m_a = var * (count - 1)
    m_b = new_var * new_count
    M2 = (
        m_a
        + m_b
        + (new_mean - mean).pow(2) * count * new_count / (count + new_count)
    )

    var = M2 / (count + new_count - 1)
    mean = (count * mean + new_count * new_mean) / (count + new_count)

    return mean, var, count + new_count


class RunningMeanAndVar(nn.Module):
    r"""Computes the running mean and variance of all samples
    seen during training and uses them to normalized inputs.
    Inputs are assumed to be images of (N, C, H, W)

    Usage:

    .. code:: py

        running_mean_var = RunningMeanAndVar(3)
        rgb = ...

        # inference
        running_mean_var.eval()
        normalized_rgb = running_mean_var(rgb)

        # training (this will update statistics)
        running_mean_var.train()
        normalized_rgb = running_mean_var(rgb)

    :param n_channels: Number of channels in the input
    :param initial_count: mean and var are initialized to 0 and 1,
        respectively.  This value is what count gets initialized to.
        Setting this to large values can help smooth the estimates
        at the start of training, thereby making it more stable
    """

    def __init__(self, n_channels: int, initial_count: float = 1.0) -> None:
        super().__init__()
        assert initial_count >= 1.0
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.ones(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.full((), initial_count))
        self._mean: torch.Tensor = self._mean
        self._var: torch.Tensor = self._var
        self._count: torch.Tensor = self._count

    def forward(self, x: Tensor) -> Tensor:
        r"""Update running statistics (if in train mode) and normalize
        x

        :param x: The input to use for udating statistics and
            to be normalized.  Shape: (N, C, H, W)

        :return: Normalized x
        """
        if self.training:
            n = x.size(0)
            # We will need to do reductions (mean) over the channel dimension,
            # so moving channels to the first dimension and then flattening
            # will make those faster.  Further, it makes things more numerically stable
            # for fp16 since it is done in a single reduction call instead of
            # multiple
            x_channels_first = (
                x.transpose(1, 0).contiguous().view(x.size(1), -1)
            )
            new_mean = x_channels_first.mean(-1, keepdim=True)
            new_count = torch.full_like(self._count, n, dtype=torch.float32)

            if distrib.is_initialized():
                new_mean = new_mean.float()
                msg = torch.cat([new_mean.view(-1), new_count.view(-1)])
                distrib.all_reduce(msg)

                new_count.copy_(msg[-1])
                new_mean.copy_(msg[0:-1].view_as(new_mean)).div_(
                    distrib.get_world_size()
                )

            new_var = (
                (x_channels_first - new_mean.type_as(x))
                .pow(2)
                .mean(-1, keepdim=True)
                .float()
            )

            if distrib.is_initialized():
                distrib.all_reduce(new_var)
                new_var /= distrib.get_world_size()

            new_mean = new_mean.view(1, -1, 1, 1)
            new_var = new_var.view(1, -1, 1, 1)

            self._mean, self._var, self._count = _welford_update(
                self._mean,
                self._var,
                self._count,
                new_mean,
                new_var,
                new_count,
            )

        inv_stdev = torch.rsqrt(torch.clamp(self._var, min=1e-2))
        # This is the same as
        # (x - self._mean) * inv_stdev but is faster since it can
        # make use of addcmul and is more numerically stable in fp16
        return torch.addcmul(
            (-self._mean * inv_stdev).type_as(x), x, inv_stdev.type_as(x)
        )
