#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.logging import logger
from habitat.core.registry import Registry, Spec


class TaskSpec(Spec):
    def __init__(self, id_task, entry_point):
        super().__init__(id_task, entry_point)


class TaskRegistry(Registry):
    """Registry for maintaining tasks.

    Args:
        id_task: id for task being registered.
        kwargs: arguments to be passed to task constructor.
    """

    def register(self, id_task, **kwargs):
        if id_task in self.specs:
            raise ValueError(
                "Cannot re-register task specification with id: {}".format(
                    id_task
                )
            )
        self.specs[id_task] = TaskSpec(id_task, **kwargs)


task_registry = TaskRegistry()


def register_task(id_task, **kwargs):
    task_registry.register(id_task, **kwargs)


def make_task(id_task, **kwargs):
    logger.info("initializing task {}".format(id_task))
    return task_registry.make(id_task, **kwargs)


def get_spec_task(id_task):
    return task_registry.get_spec(id_task)
