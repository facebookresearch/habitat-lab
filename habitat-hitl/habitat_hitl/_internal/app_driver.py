#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Callable, Optional

from habitat_hitl.core.gui_input import GuiInput
from habitat_hitl.core.hydra_utils import omegaconf_to_object
from habitat_hitl.core.text_drawer import AbstractTextDrawer
from habitat_sim.gfx import DebugLineRender


class AppDriver:
    @abc.abstractmethod
    def sim_update(self, dt: float):
        pass

    def __init__(
        self,
        config,
        gui_input: GuiInput,
        line_render: Optional[DebugLineRender],
        text_drawer: AbstractTextDrawer,
        create_app_state_lambda: Callable,
    ):
        """
        Base HITL application driver.
        """
        if "habitat_hitl" not in config:
            raise RuntimeError(
                "Required parameter 'habitat_hitl' not found in config. See hitl_defaults.yaml."
            )
        self._hitl_config = omegaconf_to_object(config.habitat_hitl)
        self._gui_input = gui_input
