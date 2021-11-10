#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def _try_register_ant_v2_task():
    try:
        from habitat.tasks.ant_v2.ant_v2 import AntV2Task  # noqa
    except ImportError as e:
        anttask_import_error = e

        @registry.register_task(name="Ant-v2")
        class AntTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise anttask_import_error
    # Register actions
    import habitat.tasks.ant_v2.ant_v2

    if not HabitatSimActions.has_action("LEG_ACTION"):
        HabitatSimActions.extend_action_space("LEG_ACTION")