#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_rearrange_task():
    import habitat.tasks.rearrange.actions.actions
    import habitat.tasks.rearrange.actions.grip_actions
    import habitat.tasks.rearrange.actions.humanoid_actions
    import habitat.tasks.rearrange.actions.oracle_nav_action
    import habitat.tasks.rearrange.actions.pddl_actions
    import habitat.tasks.rearrange.multi_agent_sensors
    import habitat.tasks.rearrange.multi_task.pddl_sensors
    import habitat.tasks.rearrange.multi_task.pddl_task
    import habitat.tasks.rearrange.rearrange_sensors
    import habitat.tasks.rearrange.rearrange_task
    import habitat.tasks.rearrange.robot_specific_sensors
    import habitat.tasks.rearrange.social_nav.oracle_social_nav_actions
    import habitat.tasks.rearrange.social_nav.social_nav_sensors
    import habitat.tasks.rearrange.social_nav.social_nav_task
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
