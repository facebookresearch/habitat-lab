#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from habitat.core.logging import HabitatLogger

logger = HabitatLogger(
    name="habitat_baselines",
    level=logging.INFO,
    format_str="%(asctime)-15s %(module)s: %(message)s",
)
