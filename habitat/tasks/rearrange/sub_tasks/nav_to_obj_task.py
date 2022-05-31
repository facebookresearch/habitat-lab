#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import random
from collections import defaultdict
from dataclasses import dataclass
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
from habitat.tasks.rearrange.utils import CacheHelper, rearrange_logger

DYN_NAV_TASK_NAME = "RearrangeNavToObjTask-v0"


@dataclass
class NavToInfo:
    """
    :property nav_target_pos: Where the robot should navigate to.
    :property nav_target_angle: What angle the robot should be at when at the goal.
    :property nav_to_task_name: The name of the sub-task we are navigating to.
    :property nav_to_obj_type: All sub-tasks are assumed to be interacting with
        some object. This is the object the sub-task we are navigating to is
        defined relative to.
    """

    nav_target_pos: mn.Vector3
    nav_target_angle: float
    nav_to_task_name: str
    nav_to_obj_type: RearrangeObjectTypes
    start_hold_obj_idx: Optional[bool] = None
    start_base_pos: Optional[mn.Vector3] = None
    start_base_rot: Optional[float] = None


@registry.register_task(name=DYN_NAV_TASK_NAME)
class DynNavRLEnv(RearrangeTask):
    """
    :_nav_to_info: Information about the next skill we are navigating to.
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
        self._nav_to_info: Optional[NavToInfo] = None

    @property
    def nav_to_obj_type(self):
        return self._nav_to_info.nav_to_obj_type

    @property
    def nav_to_task_name(self):
        return self._nav_to_info.nav_to_task_name

    @property
    def nav_target_pos(self):
        return self._nav_to_info.nav_target_pos

    @property
    def nav_target_angle(self):
        return self._nav_to_info.nav_target_angle

    def set_args(self, obj, **kwargs):
        if "marker" in kwargs:
            self.force_recep_to_name = kwargs["orig_applied_args"]["marker"]
        self.force_obj_to_idx = obj
        self.force_obj_to_name = kwargs["orig_applied_args"]["obj"]
        self.force_kwargs = kwargs

    def _get_allowed_tasks(
        self, filter_actions: Optional[List[str]] = None
    ) -> Dict[str, List[PddlAction]]:
        """
        :returns: Mapping the action name to the grounded instances of the action that are possible in the current state.
        """
        cur_preds = self.domain.get_true_predicates()

        # Get all actions which can be actively applied.
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
            False,
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

        target_pos, target_angle, obj_type = self._get_nav_targ(
            nav_to_task_name,
            {
                **task.task_args,
                ADD_CACHE_KEY: "nav",
            },
            episode,
        )

        rearrange_logger.debug(f"Got nav to skill {nav_to_task_name}")
        target_pos = np.array(self._sim.safe_snap_point(target_pos))

        start_pos, start_rot = get_robo_start_pos(self._sim, target_pos)
        return NavToInfo(
            nav_target_pos=target_pos,
            nav_target_angle=float(target_angle),
            nav_to_task_name=nav_to_task_name,
            nav_to_obj_type=obj_type,
            start_hold_obj_idx=start_hold_obj_idx,
            start_base_pos=start_pos,
            start_base_rot=start_rot,
        )

    def _get_force_nav_start_info(self, episode: Episode) -> NavToInfo:
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
                assigned_args = task.task_args

                # Check that `orig_args` is a SUBSET of `assigned_args` meaning
                # the keys and values match something in assigned args.
                is_orig_args_subset = all(
                    [
                        assigned_args.get(k, None) == v
                        or assigned_args.get(f"orig_{k}", None) == v
                        for k, v in orig_args.items()
                    ]
                )
                if is_orig_args_subset:
                    filtered_allowed_tasks.append(task)

        rearrange_logger.debug(f"Got allowed tasks {filtered_allowed_tasks}")

        if len(filtered_allowed_tasks) == 0:
            allowed_tasks_str = (
                "".join(["\n   - " + x for x in allowed_tasks]) + "\n"
            )
            raise ValueError(
                f"Got no tasks out of {allowed_tasks_str}. With entity_type={entity_type}, use_name={use_name} force kwargs={self.force_kwargs}"
            )
        nav_to_task = filtered_allowed_tasks[0]

        rearrange_logger.debug(
            f"Navigating to {nav_to_task.name} with arguments {nav_to_task.task_args}"
        )

        targ_pos, nav_target_angle, obj_type = self._get_nav_targ(
            nav_to_task.name, nav_to_task.task_args, episode
        )
        return NavToInfo(
            nav_target_pos=np.array(self._sim.safe_snap_point(targ_pos)),
            nav_target_angle=float(nav_target_angle),
            nav_to_task_name=nav_to_task.name,
            nav_to_obj_type=obj_type,
        )

    def reset(self, episode: Episode):
        sim = self._sim
        super().reset(episode, fetch_observations=False)
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
