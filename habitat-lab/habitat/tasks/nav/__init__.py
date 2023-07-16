#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def import_fallback(navtask_import_error):
    @registry.register_task(name="Nav-v0")
    class NavigationTaskImportError(EmbodiedTask):
        def __init__(self, *args, **kwargs):
            raise navtask_import_error


def _try_register_nav_task():
    try:
        from habitat.tasks.nav.nav import NavigationTask  # noqa: F401
    except ImportError as e:
        import_fallback(e)


def _try_register_languagenav_task():
    try:
        from habitat.tasks.nav.language_nav_task import (  # noqa: F401
            LanguageNavigationTask,
        )
    except ImportError as e:
        import_fallback(e)


def _try_register_goat_task():
    try:
        from habitat.tasks.nav.goat_task import GOATTask  # noqa: F401
    except ImportError as e:
        import_fallback(e)

    if not HabitatSimActions.has_action("goat_sub-task_stop"):
        HabitatSimActions.extend_action_space("goat_sub-task_stop")
