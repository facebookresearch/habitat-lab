#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

# Fixture for importing local files in 'examples/hitl/rearrange_v2'.
import sys

root_dir = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.insert(0, root_dir)
rearrange_v2_dir = os.path.join(root_dir, "examples/hitl/rearrange_v2")
sys.path.insert(0, rearrange_v2_dir)
