#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def _try_register_rearrange_task():
    import habitat.tasks.rearrange.actions.actions
    import habitat.tasks.rearrange.actions.grip_actions
    import habitat.tasks.rearrange.actions.oracle_nav_action
    import habitat.tasks.rearrange.actions.pddl_actions
    import habitat.tasks.rearrange.multi_task.composite_sensors
    import habitat.tasks.rearrange.multi_task.composite_task
    import habitat.tasks.rearrange.rearrange_sensors
    import habitat.tasks.rearrange.rearrange_task
    import habitat.tasks.rearrange.sub_tasks.articulated_object_sensors
    import habitat.tasks.rearrange.sub_tasks.articulated_object_task
    import habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors
    import habitat.tasks.rearrange.sub_tasks.nav_to_obj_task
    import habitat.tasks.rearrange.sub_tasks.pick_sensors
    import habitat.tasks.rearrange.sub_tasks.pick_task
    import habitat.tasks.rearrange.sub_tasks.place_sensors
    import habitat.tasks.rearrange.sub_tasks.place_task
    import habitat.tasks.rearrange.sub_tasks.reach_sensors
    import habitat.tasks.rearrange.sub_tasks.reach_task

    if not HabitatSimActions.has_action("arm_action"):
        HabitatSimActions.extend_action_space("arm_action")
    if not HabitatSimActions.has_action("arm_vel"):
        HabitatSimActions.extend_action_space("arm_vel")
    if not HabitatSimActions.has_action("arm_abs_pos"):
        HabitatSimActions.extend_action_space("arm_abs_pos")
    if not HabitatSimActions.has_action("arm_abs_pos_kinematic"):
        HabitatSimActions.extend_action_space("arm_abs_pos_kinematic")
    if not HabitatSimActions.has_action("suction_grasp"):
        HabitatSimActions.extend_action_space("suction_grasp")
    if not HabitatSimActions.has_action("magic_grasp"):
        HabitatSimActions.extend_action_space("magic_grasp")
    if not HabitatSimActions.has_action("base_velocity"):
        HabitatSimActions.extend_action_space("base_velocity")
    if not HabitatSimActions.has_action("arm_ee"):
        HabitatSimActions.extend_action_space("arm_ee")
    if not HabitatSimActions.has_action("empty"):
        HabitatSimActions.extend_action_space("empty")
    if not HabitatSimActions.has_action("rearrange_stop"):
        HabitatSimActions.extend_action_space("rearrange_stop")
