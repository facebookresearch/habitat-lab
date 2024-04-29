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

# This registers collaboration episodes into this application.
import collaboration_episode_loader  # noqa: 401
import hydra
from app_state_main import AppStateMain

from habitat_hitl.core.hitl_main import hitl_main
from habitat_hitl.core.hydra_utils import register_hydra_plugins



@hydra.main(
    version_base=None, config_path="config", config_name="rearrange_v2"
)
def main(config):
    # We don't sync the server camera. Instead, we maintain one camera per user.
    assert config.habitat_hitl.networking.client_sync.server_camera == False

    hitl_main(
        config,
        lambda app_service: AppStateMain(app_service),
    )


if __name__ == "__main__":
    register_hydra_plugins()
    main()
