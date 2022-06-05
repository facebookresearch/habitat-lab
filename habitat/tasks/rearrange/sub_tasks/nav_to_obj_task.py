#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import (
    PddlDomain,
    PddlProblem,
)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlAction,
    PddlEntity,
    search_for_id,
)
from habitat.tasks.rearrange.multi_task.task_creator_utils import (
    create_task_object,
)
from habitat.tasks.rearrange.rearrange_task import ADD_CACHE_KEY, RearrangeTask
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_logger

DYN_NAV_TASK_NAME = "RearrangeNavToObjTask-v0"


@dataclass
class NavToInfo:
    """
    :property nav_target_pos: Where the robot should navigate to.
    :property nav_target_angle: What angle the robot should be at when at the goal.
    :property nav_to_task_name: The name of the sub-task we are navigating to.
    """

    nav_target_pos: mn.Vector3
    nav_target_angle: float
    nav_to_task_name: str
    nav_to_entity_name: str
    start_hold_obj_idx: Optional[bool] = None
    start_base_pos: Optional[mn.Vector3] = None
    start_base_rot: Optional[float] = None


@registry.register_task(name=DYN_NAV_TASK_NAME)
class DynNavRLEnv(RearrangeTask):
    """
    :_nav_to_info: Information about the next skill we are navigating to.
    """

    pddl_problem: PddlProblem

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)
        self.force_obj_to_idx = None
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

        task_spec_path = osp.join(
            self._config.TASK_SPEC_BASE_PATH, self._config.TASK_SPEC + ".yaml"
        )

        self.pddl_problem = PddlProblem(
            PddlDomain(self._config.PDDL_DOMAIN_DEF, self._config),
            task_spec_path,
        )
        self._nav_to_info: Optional[NavToInfo] = None

    @property
    def nav_to_task_name(self):
        return self._nav_to_info.nav_to_task_name

    @property
    def nav_to_entity_name(self) -> str:
        return self._nav_to_info.nav_to_entity_name

    @property
    def nav_target_pos(self):
        return self._nav_to_info.nav_target_pos

    @property
    def nav_target_angle(self):
        return self._nav_to_info.nav_target_angle

    def set_args(self, obj, **kwargs):
        self.force_obj_to_idx = obj
        self.force_kwargs = kwargs

    def _get_allowed_tasks(
        self, filter_entities: Optional[List[PddlEntity]] = None
    ) -> Dict[str, List[PddlAction]]:
        """
        :returns: Mapping the action name to the grounded instances of the action that are possible in the current state.
        """
        cur_preds = self.pddl_problem.get_true_predicates()

        allowed_actions = None
        if len(self._config.FILTER_NAV_TO_TASKS) != 0:
            allowed_actions = self._config.FILTER_NAV_TO_TASKS

        poss_actions = self.pddl_problem.get_possible_actions(
            filter_entities=filter_entities,
            restricted_action_names=[DYN_NAV_TASK_NAME],
            allowed_action_names=allowed_actions,
        )

        grouped_poss_actions = defaultdict(list)
        for action in poss_actions:
            grouped_poss_actions[action.name].append(action)

        return dict(grouped_poss_actions)

    def _get_nav_targ(
        self,
        task_name: str,
        add_task_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mn.Vector3, float, str]:
        rearrange_logger.debug(
            f"Getting nav target for {task_name} with added arguments {add_task_kwargs}"
        )
        # Get the config for this task
        action = self.pddl_problem.actions[task_name]
        rearrange_logger.debug(f"Corresponding action {action}")

        orig_state = self._sim.capture_state(with_robot_js=True)
        action.init_task(
            self.pddl_domain.sim_info,
            should_reset=False,
            add_task_kwargs=add_task_kwargs,
        )
        robo_pos = self._sim.robot.base_pos
        heading_angle = self._sim.robot.base_rot

        self._sim.set_state(orig_state, set_hold=True)
        action.task_kwargs["orig_applied_args"]
        nav_to_entity = action.get_arg_value("obj")

        goal_type = self.pddl_problem.expr_types["goal_type"]
        rigid_type = self.pddl_problem.expr_types["rigid_obj_type"]
        if not (
            nav_to_entity.is_match(goal_type)
            or nav_to_entity.is_match(rigid_type)
        ):
            raise ValueError(
                f"Cannot navigate to non obj_type {nav_to_entity}"
            )
        if nav_to_entity:
            raise ValueError(f"`obj` argument is necessary in action {action}")

        return robo_pos, heading_angle, nav_to_entity.name

    def _generate_snap_to_obj(self) -> int:
        # Snap the target object to the robot hand.
        target_idxs, _ = self._sim.get_targets()
        return self._sim.scene_obj_ids[target_idxs[0]]

    def _generate_nav_start_goal(self, episode) -> NavToInfo:
        start_hold_obj_idx: Optional[int] = None

        # Only change the scene if this skill is not running as a sub-task
        if random.random() < self._config.OBJECT_IN_HAND_SAMPLE_PROB:
            start_hold_obj_idx = self._generate_snap_to_obj()

        allowed_tasks = self._get_allowed_tasks()

        nav_to_task_name = random.choice(list(allowed_tasks.keys()))
        task = random.choice(allowed_tasks[nav_to_task_name])

        target_pos, target_angle, nav_to_entity_name = self._get_nav_targ(
            nav_to_task_name,
            {
                ADD_CACHE_KEY: "nav",
            },
        )

        rearrange_logger.debug(f"Got nav to skill {nav_to_task_name}")
        target_pos = np.array(self._sim.safe_snap_point(target_pos))

        start_pos, start_rot = get_robo_start_pos(self._sim, target_pos)
        return NavToInfo(
            nav_target_pos=target_pos,
            nav_target_angle=float(target_angle),
            nav_to_task_name=nav_to_task_name,
            nav_to_entity_name=nav_to_entity_name,
            start_hold_obj_idx=start_hold_obj_idx,
            start_base_pos=start_pos,
            start_base_rot=start_rot,
        )

    def _get_force_nav_start_info(self, episode: Episode) -> NavToInfo:
        rearrange_logger.debug(
            f"Navigation getting target for {self.force_obj_to_idx} with task arguments {self.force_kwargs}"
        )
        name_to_id = self.pddl_problem.get_name_to_id_mapping()

        if self.force_recep_to_name is not None:
            use_entity = self.pddl_problem.get_entity(self.force_recep_to_name)
            rearrange_logger.debug(f"Forcing receptacle {use_entity}")
        else:
            use_entity = self.pddl_problem.get_entity(self.force_obj_to_name)
            rearrange_logger.debug(f"Search object name {use_entity}")

        must_include_entities = [
            self.pddl_problem.get_entity(entity_name)
            for entity_name in self.force_kwargs["orig_applied_args"]
        ]

        allowed_tasks = self._get_allowed_tasks(must_include_entities)
        if len(allowed_tasks) == 0:
            raise ValueError(f"Got no allowed tasks.")

        nav_to_task = allowed_tasks[0]

        rearrange_logger.debug(f"Navigating to {nav_to_task}")

        targ_pos, nav_target_angle, nav_to_entity_name = self._get_nav_targ(
            nav_to_task.name
        )
        return NavToInfo(
            nav_target_pos=np.array(self._sim.safe_snap_point(targ_pos)),
            nav_target_angle=float(nav_target_angle),
            nav_to_entity_name=nav_to_entity_name,
            nav_to_task_name=nav_to_task.name,
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode, fetch_observations=False)
        rearrange_logger.debug("Resetting navigation task")

        self.pddl_problem.bind_to_instance(
            self._sim, self._dataset, self, episode
        )

        episode_id = sim.ep_info["episode_id"]

        if self.force_obj_to_idx is not None:
            full_key = (
                f"{episode_id}_{self.force_obj_to_idx}_{self.force_kwargs}"
            )
            if (
                full_key in self.start_states
                and not self._config.FORCE_REGENERATE
            ):
                self._nav_to_info = self.start_states[full_key]
                rearrange_logger.debug(
                    f"Forcing episode, loaded `{full_key}` from cache {self.cache.cache_id}."
                )
                if not isinstance(self._nav_to_info, NavToInfo):
                    rearrange_logger.warning(
                        f"Incorrect cache saved to file {self._nav_to_info}. Regenerating now."
                    )
                    self._nav_to_info = None

            if self._nav_to_info is None:
                self._nav_to_info = self._get_force_nav_start_info(episode)

                self.start_states[full_key] = self._nav_to_info
                if self._config.SHOULD_SAVE_TO_CACHE:
                    self.cache.save(self.start_states)
                    rearrange_logger.debug(
                        f"Forcing episode, saved key `{full_key}` to cache {self.cache.cache_id}."
                    )
        else:
            if (
                episode_id in self.start_states
                and not self._config.FORCE_REGENERATE
            ):
                self._nav_to_info = self.start_states[episode_id]

                if (
                    not isinstance(self._nav_to_info, NavToInfo)
                    or self._nav_to_info.start_base_pos is None
                    or self._nav_to_info.start_base_rot is None
                ):
                    rearrange_logger.warning(
                        f"Incorrect cache saved to file {self._nav_to_info}. Regenerating now."
                    )
                    self._nav_to_info = None
                else:
                    rearrange_logger.debug(
                        f"Loaded episode from cache {self.cache.cache_id}."
                    )

                if (
                    self._nav_to_info is not None
                    and self._nav_to_info.start_hold_obj_idx is not None
                ):
                    # The object to hold was generated from stale object IDs.
                    # Reselect a new object to hold.
                    self._nav_to_info.start_hold_obj_idx = (
                        self._generate_snap_to_obj()
                    )

            if self._nav_to_info is None:
                self._nav_to_info = self._generate_nav_start_goal(episode)
                self.start_states[episode_id] = self._nav_to_info
                if self._config.SHOULD_SAVE_TO_CACHE:
                    self.cache.save(self.start_states)
                    rearrange_logger.debug(
                        f"Saved episode to cache {self.cache.cache_id}."
                    )
            sim.robot.base_pos = self._nav_to_info.start_base_pos
            sim.robot.base_rot = self._nav_to_info.start_base_rot
            if self._nav_to_info.start_hold_obj_idx is not None:
                if self._sim.grasp_mgr.is_grasped:
                    raise ValueError(
                        f"Attempting to grasp {self._nav_to_info.start_hold_obj_idx} even though object is already grasped"
                    )
                rearrange_logger.debug(
                    f"Forcing to grasp object {self._nav_to_info.start_hold_obj_idx}"
                )
                self._sim.grasp_mgr.snap_to_obj(
                    self._nav_to_info.start_hold_obj_idx, force=True
                )

        rearrange_logger.debug(f"Got nav to info {self._nav_to_info}")

        if not sim.pathfinder.is_navigable(self._nav_to_info.nav_target_pos):
            rearrange_logger.error("Goal is not navigable")

        if self._sim.habitat_config.DEBUG_RENDER:
            # Visualize the position the agent is navigating to.
            sim.viz_ids["nav_targ_pos"] = sim.visualize_position(
                self._nav_to_info.nav_target_pos,
                sim.viz_ids["nav_targ_pos"],
                r=0.2,
            )

        return self._get_observations(episode)


def get_robo_start_pos(
    sim, nav_targ_pos: mn.Vector3
) -> Tuple[np.ndarray, float]:
    orig_state = sim.capture_state()

    start_pos, start_rot = sim.set_robot_base_to_random_point(
        max_attempts=1000
    )

    # Reset everything except for the robot state.
    orig_state["robot_T"] = None
    sim.set_state(orig_state)
    return start_pos, start_rot
