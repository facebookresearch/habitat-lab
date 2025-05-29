#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Must call this before importing Habitat or Magnum.
# fmt: off
import torch # hack: must import early, before habitat or isaac
# make sure we restore these flags after import (habitat_sim needs RTLD_GLOBAL but that breaks Isaac)
import sys
original_flags = sys.getdlopenflags()
import magnum
import habitat_sim
sys.setdlopenflags(original_flags)
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
