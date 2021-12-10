#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def _try_register_rearrange_task():
    import habitat.tasks.rearrange.actions
    import habitat.tasks.rearrange.grip_actions
    import habitat.tasks.rearrange.rearrange_sensors
    import habitat.tasks.rearrange.rearrange_task
    import habitat.tasks.rearrange.sub_tasks.pick_sensors
    import habitat.tasks.rearrange.sub_tasks.pick_task

    if not HabitatSimActions.has_action("ARM_ACTION"):
        HabitatSimActions.extend_action_space("ARM_ACTION")
    if not HabitatSimActions.has_action("ARM_VEL"):
        HabitatSimActions.extend_action_space("ARM_VEL")
    if not HabitatSimActions.has_action("ARM_ABS_POS"):
        HabitatSimActions.extend_action_space("ARM_ABS_POS")
    if not HabitatSimActions.has_action("ARM_ABS_POS_KINEMATIC"):
        HabitatSimActions.extend_action_space("ARM_ABS_POS_KINEMATIC")
    if not HabitatSimActions.has_action("SUCTION_GRASP"):
        HabitatSimActions.extend_action_space("SUCTION_GRASP")
    if not HabitatSimActions.has_action("MAGIC_GRASP"):
        HabitatSimActions.extend_action_space("MAGIC_GRASP")
    if not HabitatSimActions.has_action("BASE_VELOCITY"):
        HabitatSimActions.extend_action_space("BASE_VELOCITY")
    if not HabitatSimActions.has_action("ARM_EE"):
        HabitatSimActions.extend_action_space("ARM_EE")
    if not HabitatSimActions.has_action("EMPTY"):
        HabitatSimActions.extend_action_space("EMPTY")
