#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import gym
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace, EmptySpace
from habitat_baselines.utils.common import iterate_action_space_recursively


class BoxActionEmbedding(nn.Module):
    r"""NeRF style sinusoidal embedding for continuous actions.

    Embeds continuous actions as [sin(x 2^t pi), cos(x 2^t pi)] where
    t is half the output dimensionality.

    x = (a - low) / (high - low). This assumes that the bounds
    in the action space are tight
    """

    # NeRF style sinusoidal embedding for continuous actions
    def __init__(self, action_space: gym.spaces.Box, dim_per_action: int = 32):
        super().__init__()

        self._ff_bands = dim_per_action // 2
        self._action_space_n_dim = len(action_space.shape)
        self.n_actions = int(np.prod(action_space.shape))
        self.register_buffer(
            "_action_low",
            torch.as_tensor(action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "_action_high",
            torch.as_tensor(action_space.high, dtype=torch.float32),
        )

        self.register_buffer(
            "_freqs",
            torch.logspace(
                start=0,
                end=self._ff_bands - 1,
                steps=self._ff_bands,
                base=2.0,
                dtype=torch.float32,
            )
            * math.pi,
        )

        self.output_size = self._ff_bands * 2 * self.n_actions

    def forward(self, action, masks=None):
        action = action.to(dtype=torch.float, copy=True)
        if masks is not None:
            action.masked_fill_(torch.logical_not(masks), 0)

        action = (
            action.sub_(self._action_low)
            .mul_(2 / (self._action_high - self._action_low))
            .add_(1)
        ).flatten(-self._action_space_n_dim)
        action = action.clamp_(-1, 1)

        action = (action.unsqueeze(-1) * self._freqs).flatten(-2)

        return torch.cat((action.sin(), action.cos()), dim=-1)


class DiscreteActionEmbedding(nn.Module):
    r"""Embeds discrete actions with an embedding table. Entry 0
    in this table functions as the start token.
    """

    def __init__(self, action_space: gym.spaces.Discrete, dim_per_action: int):
        super().__init__()
        self.n_actions = 1
        self.output_size = dim_per_action

        self.embedding = nn.Embedding(action_space.n + 1, dim_per_action)

    def forward(self, action, masks=None):
        action = action.long() + 1
        if masks is not None:
            action.masked_fill_(torch.logical_not(masks), 0)

        return self.embedding(action.squeeze(-1))


class ActionEmbedding(nn.Module):
    r"""Action embedding for a dictionary of action spaces."""

    _output_size: int

    def __init__(self, action_space: ActionSpace, dim_per_action: int = 32):
        super().__init__()

        self.embedding_modules = nn.ModuleList()
        self.embedding_slices = []

        all_spaces_empty = all(
            isinstance(space, EmptySpace)
            for space in iterate_action_space_recursively(action_space)
        )

        if all_spaces_empty:
            self.embedding_modules.append(
                DiscreteActionEmbedding(action_space, dim_per_action)
            )
            self.embedding_slices.append(slice(0, 1))
        else:
            ptr = 0
            for space in iterate_action_space_recursively(action_space):
                if isinstance(space, gym.spaces.Box):
                    e = BoxActionEmbedding(space, dim_per_action)
                elif isinstance(space, gym.spaces.Discrete):
                    e = DiscreteActionEmbedding(space, dim_per_action)
                else:
                    raise RuntimeError(f"Unknown space: {space}")

                self.embedding_modules.append(e)
                self.embedding_slices.append(slice(ptr, ptr + e.n_actions))

                ptr += e.n_actions

        self._output_size = sum(e.output_size for e in self.embedding_modules)

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, action, masks=None):
        output = []
        for _slice, emb_mod in zip(
            self.embedding_slices, self.embedding_modules
        ):
            output.append(emb_mod(action[..., _slice], masks))

        return torch.cat(output, -1)
