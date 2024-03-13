#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

# multiprocessing.dummy uses multithreading instead of multiprocessing
use_dummy = False

# User code should import this file and use e.g. multiprocessing_config.Process. Thus
# user code doesn't have to know whether we're using multiprocessing.dummy or
# multiprocessing.

Process: Any
Queue: Any
Semaphore: Any

if use_dummy:
    from multiprocessing.dummy import Process, Queue, Semaphore  # noqa: 0
else:
    from multiprocessing import Process, Queue, Semaphore  # noqa: 0
