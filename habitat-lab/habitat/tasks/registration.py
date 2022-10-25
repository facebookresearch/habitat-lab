#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.eqa import _try_register_eqa_task
from habitat.tasks.nav import _try_register_nav_task
from habitat.tasks.rearrange import _try_register_rearrange_task
from habitat.tasks.vln import _try_register_vln_task


def make_task(id_task, **kwargs):
    logger.info("Initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(
        id_task
    )

    return _task(**kwargs)


_try_register_eqa_task()
_try_register_nav_task()
_try_register_vln_task()
_try_register_rearrange_task()
