#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlAction,
    RearrangeObjectTypes,
    search_for_id,
)
from habitat.tasks.rearrange.multi_task.task_creator_utils import (
    create_task_object,
)
from habitat.tasks.rearrange.rearrange_task import ADD_CACHE_KEY, RearrangeTask
from habitat.tasks.rearrange.utils import (
    CacheHelper,
    rearrange_collision,
    rearrange_logger,
)

DYN_NAV_TASK_NAME = "RearrangeNavToObjTask-v0"


@registry.register_task(name=DYN_NAV_TASK_NAME)
class DynNavRLEnv(RearrangeTask):
    """
    :property _nav_to_task_name: The name of the task we are navigating to.
    """

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
        self._nav_to_task_name: Optional[str] = None
        self._nav_to_obj_type: RearrangeObjectTypes = (
            RearrangeObjectTypes.RIGID_OBJECT
        )

    @property
    def nav_to_obj_type(self):
        return self._nav_to_obj_type

    @property
    def nav_to_task_name(self):
        return self._nav_to_task_name

    @property
    def nav_target_pos(self):
        return self._nav_target_pos

    @property
    def nav_target_angle(self):
        return self._nav_target_angle

    def set_args(self, obj, **kwargs):
        if "marker" in kwargs:
            self.force_recep_to_name = kwargs["orig_applied_args"]["marker"]
        self.force_obj_to_idx = obj
        self.force_obj_to_name = kwargs["orig_applied_args"]["obj"]
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

    def _get_allowed_tasks(
        self, filter_actions: Optional[List[str]] = None
    ) -> Dict[str, List[PddlAction]]:
        """
        :returns: Mapping the action name to the grounded instances of the action that are possible in the current state.
        """
        cur_preds = self.domain.get_true_predicates()
        # Get all actions which can be actively applied.
        rearrange_logger.debug(f"Current true predicates {cur_preds}")

        allowed_actions = defaultdict(list)
        for action in self.domain.actions.values():
            if (
                filter_actions is not None
                and action.name not in filter_actions
            ):
                continue
            if action.task == DYN_NAV_TASK_NAME or (
                len(self._config.FILTER_NAV_TO_TASKS) != 0
                and action.name not in self._config.FILTER_NAV_TO_TASKS
            ):
                continue

            consistent_actions = action.get_possible_actions(
                cur_preds, self.domain.get_name_to_id_mapping()
            )
            rearrange_logger.debug(
                f"For {action.name} got consistent actions:"
            )
            for action in consistent_actions:
                rearrange_logger.debug(f"- {action}")
                allowed_actions[action.name].append(action)

        return allowed_actions

    def _determine_nav_pos(
        self, episode: Episode
    ) -> Tuple[mn.Vector3, float, str, RearrangeObjectTypes]:

        # Only change the scene if this skill is not running as a sub-task
        if random.random() < self._config.OBJECT_IN_HAND_SAMPLE_PROB:
            # Snap the target object to the robot hand.
            target_idxs, _ = self._sim.get_targets()
            abs_targ_idx = self._sim.scene_obj_ids[target_idxs[0]]
            self._sim.grasp_mgr.snap_to_obj(abs_targ_idx, force=True)

        allowed_tasks = self._get_allowed_tasks()

        task_name = random.choice(list(allowed_tasks.keys()))
        task = random.choice(allowed_tasks[task_name])

        nav_point, targ_angle, obj_type = self._get_nav_targ(
            task_name,
            {
                **task.task_args,
                ADD_CACHE_KEY: "nav",
            },
            episode,
        )

        return nav_point, targ_angle, task_name, obj_type

    def _get_nav_targ(
        self, task_name: str, task_args: Dict[str, Any], episode: Episode
    ) -> Tuple[mn.Vector3, float, RearrangeObjectTypes]:
        rearrange_logger.debug(
            f"Getting nav target for {task_name} with arguments {task_args}"
        )
        # Get the config for this task
        action = self.domain.get_task_match_for_name(task_name)
        rearrange_logger.debug(
            f"Corresponding action with task={action.task}, task_def={action.task_def}, config_task_args={action.config_task_args}"
        )

        orig_state = self._sim.capture_state(with_robot_js=True)
        create_task_object(
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

        _, obj_to_type = search_for_id(
            task_args["orig_obj"], self.domain.get_name_to_id_mapping()
        )

        return robo_pos, heading_angle, obj_to_type

    def _generate_nav_start_goal(self, episode):
        (
            targ_pos,
            targ_angle,
            nav_to_task_name,
            obj_type,
        ) = self._determine_nav_pos(episode)
        self._nav_target_pos = np.array(self._sim.safe_snap_point(targ_pos))

        start_pos, start_rot = get_robo_start_pos(
            self._sim, self._nav_target_pos
        )

        return (
            self._nav_target_pos,
            float(targ_angle),
            start_pos,
            float(start_rot),
            nav_to_task_name,
            obj_type,
        )

    def _get_force_nav_start_info(
        self, episode: Episode
    ) -> Tuple[np.ndarray, float, str, RearrangeObjectTypes]:
        """
        :returns: The target position and the target angle.
        """
        rearrange_logger.debug(
            f"Navigation getting target for {self.force_obj_to_idx} with task arguments {self.force_kwargs}"
        )
        name_to_id = self.domain.get_name_to_id_mapping()

        if self.force_recep_to_name is not None:
            rearrange_logger.debug(
                f"Forcing receptacle {self.force_recep_to_name}"
            )
            _, entity_type = search_for_id(
                self.force_recep_to_name, name_to_id
            )
            use_name = self.force_recep_to_name
        else:
            _, entity_type = search_for_id(self.force_obj_to_name, name_to_id)
            use_name = self.force_obj_to_name
            rearrange_logger.debug(
                f"Search object name {use_name} with type {entity_type}"
            )

        matching_skills = self.domain.get_matching_skills(
            entity_type, use_name
        )

        allowed_tasks = self._get_allowed_tasks(matching_skills)
        if len(allowed_tasks) == 0:
            raise ValueError(
                f"Got no allowed tasks {allowed_tasks} from {matching_skills}, {entity_type}, {use_name}"
            )

        filtered_allowed_tasks = []
        orig_args = self.force_kwargs["orig_applied_args"]
        for sub_allowed_tasks in allowed_tasks.values():
            for task in sub_allowed_tasks:
                assigned_args = {
                    k: v
                    for k, v in zip(
                        task.parameters, task.orig_applied_func_args
                    )
                }
                # Check that `orig_args` is a SUBSET of `assigned_args` meaning
                # the keys and values match something in assigned args.
                is_orig_args_subset = all(
                    [
                        (k in assigned_args) and (assigned_args[k] == v)
                        for k, v in orig_args.items()
                    ]
                )
                if is_orig_args_subset:
                    filtered_allowed_tasks.append(task)
        rearrange_logger.debug(f"Got allowed tasks {filtered_allowed_tasks}")

        if len(filtered_allowed_tasks) == 0:
            raise ValueError(
                f"Got no tasks out of {allowed_tasks} with entity_type={entity_type}, use_name={use_name}"
            )
        nav_to_task = filtered_allowed_tasks[0]

        rearrange_logger.debug(
            f"Navigating to {nav_to_task.name} with arguments {nav_to_task.task_args}"
        )

        targ_pos, self._nav_target_angle, obj_type = self._get_nav_targ(
            nav_to_task.name, nav_to_task.task_args, episode
        )
        return (
            np.array(self._sim.safe_snap_point(targ_pos)),
            float(self._nav_target_angle),
            nav_to_task.name,
            obj_type,
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode)
        rearrange_logger.debug("Resetting navigation task")

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
                    self._nav_to_task_name,
                    self._nav_to_obj_type,
                ) = self.start_states[full_key]
                rearrange_logger.debug(
                    f"Forcing episode, loaded `{full_key}` from cache {self.cache.cache_id}."
                )
            else:
                (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    self._nav_to_task_name,
                    self._nav_to_obj_type,
                ) = self._get_force_nav_start_info(episode)

                self.start_states[full_key] = (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    self._nav_to_task_name,
                    self._nav_to_obj_type,
                )
                if self._config.SHOULD_SAVE_TO_CACHE:
                    self.cache.save(self.start_states)
                    rearrange_logger.debug(
                        f"Forcing episode, saved key `{full_key}` to cache {self.cache.cache_id}."
                    )
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
                    self._nav_to_task_name,
                    self._nav_to_obj_type,
                ) = self.start_states[episode_id]
                rearrange_logger.debug(
                    f"Loaded episode from cache {self.cache.cache_id}."
                )

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
                    self._nav_to_task_name,
                    self._nav_to_obj_type,
                ) = self._generate_nav_start_goal(episode)
                self.start_states[episode_id] = (
                    self._nav_target_pos,
                    self._nav_target_angle,
                    start_pos,
                    start_rot,
                    self._nav_to_task_name,
                    self._nav_to_obj_type,
                )
                if self._config.SHOULD_SAVE_TO_CACHE:
                    self.cache.save(self.start_states)
                    rearrange_logger.debug(
                        f"Saved episode to cache {self.cache.cache_id}."
                    )

            targ_idxs, goal_pos = sim.get_targets()

        rearrange_logger.debug(
            f"Got nav target position {self._nav_target_pos}"
        )

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
