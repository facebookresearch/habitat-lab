#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat.config.default import DictConfig, get_config
from habitat.config.read_write import read_write

__all__ = ["get_config", "read_write", "default_structured_configs"]
