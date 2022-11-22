#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from gym import spaces

try:
    import torch
    import torch.distributed

    from habitat_baselines.rl.ddppo.policy.resnet import resnet18, resnet50
    from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder

    baseline_installed = True
except ImportError:
    baseline_installed = False


def _npobs_dict_to_tensorobs_dict(npobs_dict):
    result = {}
    for k, v in npobs_dict.items():
        result[k] = torch.as_tensor(v).unsqueeze(0)
    return result


@pytest.mark.skipif(
    not baseline_installed, reason="baseline sub-module not installed"
)
@pytest.mark.parametrize(
    "observation_space",
    [
        spaces.Dict(
            {
                "rgb_1": spaces.Box(low=0, high=1, shape=(62, 30, 3)),
                "rgb_2": spaces.Box(low=0, high=1, shape=(62, 30, 2)),
            },
        ),
        spaces.Dict(
            {
                "rgb_1": spaces.Box(low=0, high=1, shape=(63, 84, 1)),
                "depth_1": spaces.Box(low=0, high=1, shape=(63, 84, 2)),
            },
        ),
        spaces.Dict(
            {
                "rgb_1": spaces.Box(low=0, high=1, shape=(64, 128, 3)),
            },
        ),
        spaces.Dict(
            {
                "rgb_1": spaces.Box(low=0, high=1, shape=(65, 30, 3)),
                "rgb_2": spaces.Box(low=0, high=1, shape=(65, 30, 1)),
                "depth_1": spaces.Box(low=0, high=1, shape=(65, 30, 2)),
            },
        ),
        spaces.Dict(
            {
                "rgb_1": spaces.Box(low=0, high=1, shape=(66, 64, 3)),
                "depth_2": spaces.Box(low=0, high=1, shape=(66, 64, 2)),
            },
        ),
    ],
)
@pytest.mark.parametrize("backbone", [resnet18, resnet50])
def test_resnetencoder_initialization(observation_space, backbone):
    encoder = ResNetEncoder(observation_space, make_backbone=backbone)
    obs = observation_space.sample()
    t_obs = _npobs_dict_to_tensorobs_dict(obs)
    out = encoder.forward(t_obs)
    assert out.shape == (1, *encoder.output_shape)
