#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from typing import List

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.dynamic_task_utils import (
    load_task_object,
)
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    Action,
    search_for_id,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_collision

DYN_NAV_TASK_NAME = "RearrangeNavToObjTask-v0"


@registry.register_task(name=DYN_NAV_TASK_NAME)
class DynNavRLEnv(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.force_obj_to_idx = None
        self.force_recep_to_name = None
        self._prev_measure = 1.0

        data_path = dataset.config.DATA_PATH.format(split=dataset.config.SPLIT)
        fname = data_path.split("/")[-1].split(".")[0]
        save_dir = osp.dirname(data_path)
        self.cache = CacheHelper(
            osp.join(save_dir, f"{fname}_{config.TYPE}_start.pickle"),
            def_val={},
            verbose=False,
        )
        self.start_states = self.cache.load()
        self.domain = None
        self._nav_target_pos = mn.Vector3(0.0, 0.0, 0.0)
        self._nav_target_angle = 0.0

    @property
    def nav_target_pos(self):
        return self._nav_target_pos

    @property
    def nav_target_angle(self):
        return self._nav_target_angle

    def set_args(self, obj_to, **kwargs):
        if "recep_to" in kwargs:
            self.force_recep_to_name = kwargs["orig_applied_args"]["recep_to"]
        self.force_obj_to_idx = obj_to
        self.force_obj_to_name = kwargs["orig_applied_args"]["obj_to"]
        self.force_kwargs = kwargs

    def _get_agent_pos(self):
        sim = self._env._sim
        current_pos = sim.get_robot_transform().translation
        return sim.safe_snap_point(current_pos)

    def _get_cur_geo_dist(self):
        sim = self._env._sim
        distance_to_target = sim.geodesic_distance(
            self._get_agent_pos(),
            [self._nav_target_pos],
            None,
        )
        return distance_to_target

    def _get_allowed_tasks(self) -> List[Action]:
        cur_preds = self.domain.get_true_predicates()
        # Get all actions which can be actively applied.
        allowed_tasks = [
            action
            for action in self.domain.actions.values()
            if action.task != DYN_NAV_TASK_NAME
            and action.are_preconditions_true(cur_preds)
            and (
                len(self._config.FILTER_NAV_TO_TASKS) == 0
                or action.name in self._config.FILTER_NAV_TO_TASKS
            )
        ]
        return allowed_tasks

    def _determine_nav_pos(self, episode):
        allowed_tasks = self._get_allowed_tasks()

        use_task = random.choice(allowed_tasks)
        task_name = use_task.name

        nav_point, targ_angle = self._get_nav_targ(
            task_name, {"obj": 0}, episode
        )

        return nav_point, targ_angle

    def _internal_log(self, s):
        logger.info(f"NavToObjTask: {s}")

    def _get_nav_targ(self, task_name: str, task_args, episode):
        self._internal_log(f"Getting nav target for {task_name}")
        # Get the config for this task
        action = self.domain.get_task_match_for_name(task_name)

        orig_state = self._sim.capture_state(with_robot_js=True)
        load_task_object(
            action.task,
            action.task_def,
            self._config.clone(),
            self,
            self._dataset,
            True,
            task_args,
            episode,
            action.config_task_args,
        )
        robo_pos = self._sim.robot.base_pos
        heading_angle = self._sim.robot.base_rot

        self._sim.set_state(orig_state, set_hold=True)

        return robo_pos, heading_angle

    def _generate_nav_start_goal(self, episode):
        targ_pos, targ_angle = self._determine_nav_pos(episode)
        self._nav_target_pos = np.array(self._sim.safe_snap_point(targ_pos))

        start_pos, start_rot = get_robo_start_pos(
            self._sim, self._nav_target_pos
        )

        return (
            self._nav_target_pos,
            float(targ_angle),
            start_pos,
            float(start_rot),
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode)

        if self.domain is None:
            self.domain = PddlDomain(
                self._config.PDDL_DOMAIN_DEF,
                self._dataset,
                self._config,
                self._sim,
            )
        else:
            self.domain.reset()

        episode_id = sim.ep_info["episode_id"]

        if self.force_obj_to_idx is not None:
            full_key = (
                f"{episode_id}_{self.force_obj_to_idx}_{self.force_kwargs}"
            )
            if (
                full_key in self.start_states
                and not self._config.FORCE_REGENERATE
            ):
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                ) = self.start_states[full_key]
            else:
                self._internal_log(
                    f"Navigation getting target for {self.force_obj_to_idx} with task arguments {self.force_kwargs}"
                )
                name_to_id = self.domain.get_name_to_id_mapping()

                if self.force_recep_to_name is not None:
                    _, entity_type = search_for_id(
                        self.force_recep_to_name, name_to_id
                    )
                    use_name = self.force_recep_to_name

                else:
                    _, entity_type = search_for_id(
                        self.force_obj_to_name, name_to_id
                    )
                    use_name = self.force_obj_to_name

                matching_skills = self.domain.get_matching_skills(
                    entity_type, use_name
                )

                allowed_tasks = self._get_allowed_tasks()
                allowed_tasks = [
                    task
                    for task in allowed_tasks
                    if task.name in matching_skills
                ]
                self._internal_log(f"Got allowed tasks {allowed_tasks}")
                task_args = {"obj": self.force_obj_to_idx}

                if len(allowed_tasks) != 1:
                    raise ValueError(
                        f"Got multiple possible tasks {allowed_tasks}"
                    )
                nav_to_task = allowed_tasks[0]
                self._internal_log(
                    f"Navigating to {nav_to_task.name} with arguments {task_args}"
                )

                targ_pos, self._nav_target_angle = self._get_nav_targ(
                    nav_to_task.name, task_args, episode
                )
                self._nav_target_pos = np.array(
                    self._sim.safe_snap_point(targ_pos)
                )
                self._nav_target_angle = float(self._nav_target_angle)

                self.start_states[full_key] = (
                    self._nav_target_pos,
                    self._nav_target_angle,
                )
                self.cache.save(self.start_states)
            start_pos, start_rot = get_robo_start_pos(
                sim, self._nav_target_pos
            )
        else:
            if (
                episode_id in self.start_states
                and not self._config.FORCE_REGENERATE
            ):
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                ) = self.start_states[episode_id]

                sim.robot.base_pos = mn.Vector3(
                    start_pos[0],
                    sim.robot.base_pos[1],
                    start_pos[2],
                )
                sim.robot.base_rot = start_rot
            else:
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                ) = self._generate_nav_start_goal(episode)
                self.start_states[episode_id] = (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                )
                self.cache.save(self.start_states)

            targ_idxs, goal_pos = sim.get_targets()

        if not sim.pathfinder.is_navigable(self._nav_target_pos):
            print("Goal is not navigable")

        if self._sim.habitat_config.DEBUG_RENDER:
            # Visualize the position the agent is navigating to.
            sim.viz_ids["nav_targ_pos"] = sim.visualize_position(
                self._nav_target_pos,
                sim.viz_ids["nav_targ_pos"],
                r=0.2,
            )

        return self._get_observations(episode)


def get_robo_start_pos(sim, nav_targ_pos):
    timeout_len = 1000
    orig_state = sim.capture_state()

    # Find a valid navigable point between the start and goal.
    i = 0
    while i < timeout_len:
        start_pos = sim.pathfinder.get_random_navigable_point()
        start_rot = np.random.uniform(0, 2 * np.pi)
        sim.robot.base_pos = start_pos
        sim.robot.base_rot = start_rot
        start_island_radius = sim.pathfinder.island_radius(start_pos)
        goal_island_radius = sim.pathfinder.island_radius(nav_targ_pos)

        current_position = sim.robot.base_pos
        # This should only snap the height
        current_position = sim.safe_snap_point(current_position)
        distance_to_target = sim.geodesic_distance(
            current_position, [nav_targ_pos], None
        )
        is_valid_nav = (
            start_island_radius == goal_island_radius
            and distance_to_target != np.inf
        )
        if not is_valid_nav:
            continue

        # no collision check
        for _ in range(5):
            sim.internal_step(-1)
            did_collide, details = rearrange_collision(
                sim,
                True,
                ignore_base=False,
            )
        if not did_collide:
            break
        i += 1
    if i == timeout_len - 1:
        if not is_valid_nav:
            print("Goal and start position are not navigable.")
        else:
            print("Could not get nav start without collision")
    # Reset everything except for the robot state.
    orig_state["robot_T"] = None
    sim.set_state(orig_state)
    return start_pos, start_rot
