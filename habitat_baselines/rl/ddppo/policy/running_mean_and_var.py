#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as distrib
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))

        self._distributed = distrib.is_initialized()

    def forward(self, x):
        if self.training:
            new_mean = F.adaptive_avg_pool2d(x, 1).sum(0, keepdim=True)
            new_count = torch.full_like(self._count, x.size(0))

            if self._distributed:
                distrib.all_reduce(new_mean)
                distrib.all_reduce(new_count)

            new_mean /= new_count

            new_var = F.adaptive_avg_pool2d((x - new_mean).pow(2), 1).sum(
                0, keepdim=True
            )

            if self._distributed:
                distrib.all_reduce(new_var)

            # No - 1 on all the variance as the number of pixels
            # seen over training is simply absurd, so it doesn't matter
            new_var /= new_count

            m_a = self._var * (self._count)
            m_b = new_var * (new_count)
            M2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )

            self._var = M2 / (self._count + new_count)
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )

            self._count += new_count

        stdev = torch.sqrt(
            torch.max(self._var, torch.full_like(self._var, 1e-2))
        )
        return (x - self._mean) / stdev
