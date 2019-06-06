#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.eqa.eqa_task import EQATask
from habitat.tasks.nav.nav_task import NavigationTask


def make_task(id_task, **kwargs):
    logger.info("initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(
        id_task
    )

    return _task(**kwargs)
