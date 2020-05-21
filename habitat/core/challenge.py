#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from habitat.core.benchmark import Benchmark
from habitat.core.logging import logger


class Challenge(Benchmark):
    def __init__(self, eval_remote=False):
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        super().__init__(config_paths, eval_remote=eval_remote)

    def submit(self, agent):
        metrics = super().evaluate(agent)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))
