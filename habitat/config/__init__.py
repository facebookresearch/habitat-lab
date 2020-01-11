#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import yacs.config

from habitat.config.default import get_config

# Default config node
Config = lambda *args, **kwargs: yacs.config.CfgNode(
    *args, **kwargs, new_allowed=True
)


__all__ = ["Config", "get_config"]
