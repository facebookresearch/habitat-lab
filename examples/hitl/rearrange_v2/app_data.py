#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from habitat_hitl.core.types import ConnectionRecord


class AppData:
    """
    RearrangeV2 application data shared by all states.
    """

    def __init__(self, max_user_count: int):
        self.max_user_count = max_user_count
        self.connected_users: Dict[int, ConnectionRecord] = {}
