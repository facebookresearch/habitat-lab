#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.tasks.registration import task_registry, register_task, make_task

register_task(id_task="EQA-v0", entry_point="habitat.tasks.eqa:EQATask")

register_task(id_task="Nav-v0", entry_point="habitat.tasks.nav:NavigationTask")

__all__ = ["task_registry", "register_task", "make_task"]
