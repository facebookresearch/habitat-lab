#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat
from habitat.core.logging import logger


def test_habitat_install():
    r"""dummy test for testing installation"""
    logger.info(str(habitat))
