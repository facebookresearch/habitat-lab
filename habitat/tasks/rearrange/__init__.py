#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def _try_register_rearrange_task():
    try:
        import habitat.tasks.rearrange.rearrange_sensors
    except ImportError as e:
        print(e)

    try:
        import habitat.tasks.rearrange.rearrange_pick_task
        import habitat.tasks.rearrange.rearrange_task

    except ImportError as e:
        print(e)
        rearrangetask_import_error = e

        @registry.register_task(name="Rearrange-v0")
        class RearrangeTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise rearrangetask_import_error

    # Register actions
    import habitat.tasks.rearrange.actions

    if not HabitatSimActions.has_action("ARM_ACTION"):
        HabitatSimActions.extend_action_space("ARM_ACTION")
    if not HabitatSimActions.has_action("ARM_VEL"):
        HabitatSimActions.extend_action_space("ARM_VEL")
    if not HabitatSimActions.has_action("MAGIC_GRASP"):
        HabitatSimActions.extend_action_space("MAGIC_GRASP")
