#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)


def register_hydra_plugins():
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    # todo: also register search paths for hitl framework's configs
