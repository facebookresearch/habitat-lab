#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.simulator import Observations


class Agent:
    """Abstract class for defining agents which act inside Env. This abstract
    class standardizes agents to allow seamless benchmarking. To implement an
    agent the user has to implement two methods:

        reset
        act
    """

    def reset(self) -> None:
        """Called before starting a new episode in environment.
        """
        raise NotImplementedError

    def act(self, observations: Observations) -> int:
        """

        Args:
            observations: observations coming in from environment to be used
                by agent to decide action.

        Returns:
            action to be taken inside the environment
        """
        raise NotImplementedError
