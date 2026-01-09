# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import attr
import gym.spaces as spaces


@attr.s(auto_attribs=True, slots=True)
class EnvironmentSpec:
    """
    Stores information about the spaces of an environment.

    :property obs_space: Observation space of the environment.
    :property action_space: The potentially flattened version of the environment action space.
    :property orig_action_space: The non-flattened version of the environment action space.
    """

    observation_space: spaces.Space
    action_space: spaces.Space
    orig_action_space: spaces.Space
