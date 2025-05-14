#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Must call this before importing Habitat or Magnum.
# fmt: off
import ctypes
import sys

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
# fmt: on

import hydra
from state_machine import StateMachine

from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins


@hydra.main(version_base=None, config_path="./config", config_name="headless")
def main(config):
    hitl_main(
        config,
        lambda app_service: StateMachine(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
