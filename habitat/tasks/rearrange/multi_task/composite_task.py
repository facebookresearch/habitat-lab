#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlAction,
    PddlSetState,
    Predicate,
    RearrangeObjectTypes,
    parse_func,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger


@registry.register_task(name="RearrangeCompositeTask-v0")
class CompositeTask(RearrangeTask):
    """
    All tasks using a combination of sub-tasks (skills) should utilize this task.
    """

    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)

        task_spec_path = osp.join(
            self._config.TASK_SPEC_BASE_PATH, self._config.TASK_SPEC + ".yaml"
        )

        with open(task_spec_path, "r") as f:
            task_def = yaml.safe_load(f)

        # Stores configuration for the task.
        self.task_def: Dict[str, Any] = task_def

        self.start_state = PddlSetState(task_def["start"]["state"])

        self._cur_node_idx: int = -1
        self._cur_task: RearrangeTask = None
        self._cached_tasks: Dict[str, RearrangeTask] = {}
        self._cur_state = None

        # None until loaded.
        self.domain: Optional[PddlDomain] = None
        self._stage_goals: Optional[Dict[str, List[Predicate]]] = {}
        self._goal_state: Optional[List[Predicate]] = None
        self._solution: Optional[List[PddlAction]] = None

        # Based on the current environment state, we can infer which subtask
        # from the solution list the agent is currently executing.
        self._inferred_cur_node_idx: int = -1
        self._inferred_cur_task: Optional[RearrangeTask] = None

        if self._config.SINGLE_EVAL_NODE >= 0:
            self._cur_node_idx = self._config.SINGLE_EVAL_NODE

    @property
    def stage_goals(self) -> Dict[str, List[Predicate]]:
        return self._stage_goals

    def _parse_precond_list(
        self, predicate_strs: List[str]
    ) -> List[Predicate]:
        preds = []
        for pred_s in predicate_strs:
            pred = copy.deepcopy(self.domain.predicate_lookup(pred_s))
            _, effect_arg = parse_func(pred_s)
            effect_arg = effect_arg.split(",")
            if effect_arg[0] == "":
                effect_arg = []
            pred.bind(effect_arg)
            preds.append(pred)
        return preds

    def load_solution(self, solution_d: Dict[str, Any]) -> List[PddlAction]:
        """
        Loads the solution definition from the PDDL file and converts it to a
        list of executable actions.
        """
        solution = []
        for i, action in enumerate(solution_d):
            name, args = parse_func(action)
            args = args.split(",")

            ac_instance = self.domain.actions[name].copy_new()

            ac_instance.bind(
                args, self.task_def.get("add_args", {}).get(i, {})
            )
            solution.append(ac_instance)
        return solution

    def jump_to_node(
        self, node_idx: int, episode: Episode, is_full_task: bool = False
    ) -> None:
        """
        Sequentially applies all solution actions before `node_idx`. But NOT
        including the solution action at index `node_idx`.

        :param node_idx: An integer in [0, len(self._solution)).
        :param is_full_task: If true, then calling reset will always the task to this solution node.
        """

        rearrange_logger.debug(
            "Jumping to node {node_idx}, is_full_task={is_full_task}"
        )
        # We don't want to reset to this node if we are in full task mode.
        if not is_full_task:
            self._cur_node_idx = node_idx

        for i in range(node_idx):
            self._solution[i].apply(
                self.domain.get_name_to_id_mapping(), self._sim
            )

        if node_idx not in self._cached_tasks:
            task = self._solution[node_idx].init_task(self, episode)
            self._cached_tasks[node_idx] = task
        else:
            self._cached_tasks[node_idx].reset(episode)

    def reset(self, episode: Episode):
        super().reset(episode, fetch_observations=False)
        if self.domain is None:
            self.domain = PddlDomain(
                self._config.PDDL_DOMAIN_DEF,
                self._dataset,
                self._config,
                self._sim,
            )
        else:
            self.domain.reset()

        self._solution = self.load_solution(self.task_def["solution"])
        self._goal_state = self._parse_precond_list(self.task_def["goal"])
        self._cur_state = self._parse_precond_list(
            self.task_def["start"]["precondition"]
        )

        for k, preconds in self.task_def["stage_goals"].items():
            self._stage_goals[k] = self._parse_precond_list(preconds)

        self.start_state.set_state(
            self.domain.get_name_to_id_mapping(), self._sim
        )

        if self._config.DEBUG_SKIP_TO_NODE != -1:
            self.jump_to_node(
                self._config.DEBUG_SKIP_TO_NODE, episode, is_full_task=True
            )

        if self._cur_node_idx >= 0:
            self.jump_to_node(self._cur_node_idx, episode)

        self._inferred_cur_node_idx = 0
        self._inferred_cur_task = None
        if self._config.USING_SUBTASKS:
            self._increment_solution_subtask(episode)
        self._cached_tasks.clear()
        return self._get_observations(episode)

    def get_inferred_node_idx(self) -> int:
        if not self._config.USING_SUBTASKS:
            raise ValueError(
                "Cannot get inferred sub-task when task is not configured to use sub-tasks. See `TASK.USING_SUBTASKS` key."
            )
        return self._inferred_cur_node_idx

    def get_inferrred_node_task(self) -> RearrangeTask:
        if not self._config.USING_SUBTASKS:
            raise ValueError(
                "Cannot get inferred sub-task when task is not configured to use sub-tasks. See `TASK.USING_SUBTASKS` key."
            )
        return self._inferred_cur_task

    def increment_inferred_solution_idx(self, episode: Episode) -> None:
        """
        Increment to the next index in the solution list. If the solution is
        exhausted then stay at the last index. This will update both
        `inferred_node_idx` and `inferrred_cur_task`.
        """
        prev_inf_cur_node = self._inferred_cur_node_idx
        self._inferred_cur_node_idx += 1
        if not self._increment_solution_subtask(episode):
            self._inferred_cur_node_idx = prev_inf_cur_node

    def _increment_solution_subtask(self, episode: Episode) -> bool:
        """
        Gets the next inferred sub-task in the solution list. Returns False if
        there are no remaining sub-tasks in the solution list.
        """
        task_solution = self.solution
        if self._inferred_cur_node_idx >= len(task_solution):
            return False
        while (
            task_solution[self._inferred_cur_node_idx].name
            in self._config.SKIP_NODES
        ):
            self._inferred_cur_node_idx += 1
            if self._inferred_cur_node_idx >= len(task_solution):
                return False

        prev_state = self._sim.capture_state(with_robot_js=True)
        if self._inferred_cur_node_idx in self._cached_tasks:
            self._inferred_cur_task = self._cached_tasks[
                self._inferred_cur_node_idx
            ]
            self._inferred_cur_task.reset(episode)
            rearrange_logger.debug(
                f"Incrementing solution to {self._inferred_cur_node_idx}. Loading next task from cached"
            )
        else:
            rearrange_logger.debug(
                f"Incrementing solution to {self._inferred_cur_node_idx}. Loading next task."
            )
            task = task_solution[self._inferred_cur_node_idx].init_task(
                self, episode, should_reset=False
            )
            self._cached_tasks[self._inferred_cur_node_idx] = task
            self._inferred_cur_task = task
        self._sim.set_state(prev_state)

        return True

    @property
    def forced_node_task(self) -> RearrangeTask:
        """
        The current sub-task from the solution list the agent is forced to be
        in. This must be programmatically. Unlike the inferred_node, this will
        not automatically increment.
        """
        if self._cur_node_idx >= 0:
            return self._cached_tasks[self._cur_node_idx]
        else:
            return None

    @property
    def forced_node_task_idx(self) -> int:
        """
        The index of the current sub-task in the solution list the agent is at.
        """
        return self._cur_node_idx

    @property
    def num_solution_subtasks(self) -> int:
        """
        Get the number of sub-tasks in the solution.
        """
        return len(self._solution)

    @property
    def solution(self) -> List[PddlAction]:
        """
        Hard-coded solution defined in the task PDDL config.
        """
        return self._solution

    def are_predicates_satisfied(self, preds: List[Predicate]) -> bool:
        """ """
        return all(self.domain.is_pred_true(pred) for pred in reversed(preds))

    def is_goal_state_satisfied(self) -> bool:
        return self.are_predicates_satisfied(self._goal_state)

    def _try_get_subtask_prop(self, prop_name: str, def_val: Any) -> Any:
        """
        Try to get a property from the current inferred subtask. If the subtask
        is not valid, then return the supplied default value.
        """
        if self.forced_node_task is not None and hasattr(
            self.cur_task, prop_name
        ):
            return getattr(self.cur_task, prop_name)

        elif self._inferred_cur_task is not None and hasattr(
            self._inferred_cur_task, prop_name
        ):
            return getattr(self._inferred_cur_task, prop_name)
        return def_val

    #########################################################################
    # START Sub-task property overrides
    # These will emulate properties from sub-tasks needed to compute sub-task
    # sensors and measurements.
    #########################################################################

    @property
    def targ_idx(self) -> int:
        return self._try_get_subtask_prop("targ_idx", self._targ_idx)

    @property
    def abs_targ_idx(self) -> int:
        if self._targ_idx is None:
            abs_targ_idx = None
        else:
            abs_targ_idx = self._sim.get_targets()[0][self._targ_idx]

        return self._try_get_subtask_prop("abs_targ_idx", abs_targ_idx)

    @property
    def nav_to_task_name(self) -> str:
        return self._try_get_subtask_prop("nav_to_task_name", None)

    @property
    def nav_to_obj_type(self) -> RearrangeObjectTypes:
        return self._try_get_subtask_prop(
            "nav_to_obj_type", RearrangeObjectTypes.RIGID_OBJECT
        )

    @property
    def nav_target_pos(self) -> np.ndarray:
        return self._try_get_subtask_prop("nav_target_pos", np.zeros((3,)))

    @property
    def nav_target_angle(self) -> float:
        return self._try_get_subtask_prop("nav_target_angle", 0.0)

    @property
    def success_js_state(self) -> float:
        return 0.0

    @property
    def use_marker_name(self) -> str:
        subtask_marker_name = self._try_get_subtask_prop(
            "use_marker_name", None
        )
        if subtask_marker_name is not None:
            return subtask_marker_name
        else:
            all_markers = self._sim.get_all_markers()
            return list(all_markers.keys())[0]

    def get_use_marker(self) -> MarkerInfo:
        return self._sim.get_marker(self.use_marker_name)

    #########################################################################
    # END Sub-task property overrides
    #########################################################################
