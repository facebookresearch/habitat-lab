#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod


class Controller(ABC):
    """Abstract controller."""

    def __init__(self, is_multi_agent):
        self._is_multi_agent = is_multi_agent

    @abstractmethod
    def act(self, obs, env):
        pass

    def on_environment_reset(self):
        pass


class GuiController(Controller):
    """Abstract controller for gui agents."""

    def __init__(self, agent_idx, is_multi_agent, gui_input):
        super().__init__(is_multi_agent)
        self._agent_idx = agent_idx
        self._gui_input = gui_input
