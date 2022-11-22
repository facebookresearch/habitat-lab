#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from habitat.core.logging import HabitatLogger

baselines_logger = HabitatLogger(
    name="habitat_baselines",
    level=int(os.environ.get("HABITAT_BASELINES_LOG", logging.ERROR)),
    format_str="[%(levelname)s,%(name)s] %(asctime)-15s %(filename)s:%(lineno)d %(message)s",
)
