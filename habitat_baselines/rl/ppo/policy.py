#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from habitat_baselines.rl.ppo.utils import Flatten, CategoricalNet

import numpy as np


class Policy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__()
        self.dim_actions = action_space.n

        self.net = Net(
            observation_space=observation_space, hidden_size=hidden_size
        )

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, masks, deterministic=False):
        value, actor_features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, masks
        )
        distribution = self.action_distribution(actor_features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, masks):
        value, _, _ = self.net(observations, rnn_hidden_states, masks)
        return value

    def evaluate_actions(self, observations, rnn_hidden_states, masks, action):
        value, actor_features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, masks
        )
        distribution = self.action_distribution(actor_features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class Net(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        self._n_input_goal = observation_space.spaces["pointgoal"].shape[0]
        self._hidden_size = hidden_size

        self.cnn = self._init_perception_model(observation_space)

        if self.is_blind:
            self.rnn = nn.GRU(self._n_input_goal, self._hidden_size)
        else:
            self.rnn = nn.GRU(
                self.output_size + self._n_input_goal, self._hidden_size
            )

        self.critic_linear = nn.Linear(self._hidden_size, 1)

        self.layer_init()
        self.train()

    def _init_perception_model(self, observation_space):
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )

        if self.is_blind:
            return nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            return nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                Flatten(),
                nn.Linear(32 * cnn_dims[0] * cnn_dims[1], self._hidden_size),
                nn.ReLU(),
            )

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    @property
    def output_size(self):
        return self._hidden_size

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                nn.init.constant_(layer.bias, val=0)

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        nn.init.orthogonal_(self.critic_linear.weight, gain=1)
        nn.init.constant_(self.critic_linear.bias, val=0)

    def forward_rnn(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(0):
            x, hidden_states = self.rnn(
                x.unsqueeze(0), (hidden_states * masks).unsqueeze(0)
            )
            x = x.squeeze(0)
            hidden_states = hidden_states.squeeze(0)
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(0)
            t = int(x.size(0) / n)

            # unflatten
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)

            # steps in sequence which have zero for any agent. Assume t=0 has
            # a zero in it.
            has_zeros = (
                (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            )

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]  # handle scalar
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [t]

            hidden_states = hidden_states.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # process steps that don't have any zeros in masks together
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    hidden_states * masks[start_idx].view(1, -1, 1),
                )

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)  # flatten
            hidden_states = hidden_states.squeeze(0)

        return x, hidden_states

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward_perception_model(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        return self.cnn(cnn_input)

    def forward(self, observations, rnn_hidden_states, masks):
        x = observations["pointgoal"]

        if not self.is_blind:
            perception_embed = self.forward_perception_model(observations)
            x = torch.cat([perception_embed, x], dim=1)

        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)

        return self.critic_linear(x), x, rnn_hidden_states
