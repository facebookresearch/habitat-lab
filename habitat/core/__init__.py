#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import enum

class CHANGE_TASK_BEHAVIOUR(enum.IntEnum):
    FIXED = 0
    RANDOM = 1
    
class CHANGE_TASK_CYCLE_BEHAVIOUR(enum.IntEnum):
    ORDER = 0
    RANDOM = 1