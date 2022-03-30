#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from typing import Any, Dict, List, Optional

import magnum as mn
import numpy as np
import yaml

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    Action,
    Predicate,
    RearrangeObjectTypes,
    SetState,
    parse_func,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import logger


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

        self.start_state = SetState(task_def["start"]["state"])

        self._cur_node: int = -1
        self._cur_task: RearrangeTask = None
        self.cached_tasks: Dict[str, RearrangeTask] = {}
        self._cur_state = None

        # None until loaded.
        self.domain: Optional[PddlDomain] = None
        self._stage_goals: Optional[Dict[str, List[Predicate]]] = {}
        self._goal_state: Optional[List[Predicate]] = None
        self._solution: Optional[List[Action]] = None

        # Based on the current environment state, we can infer which subtask
        # from the solution list the agent is currently executing.
        self._inferred_cur_node_idx: int = -1
        self._inferred_cur_task: Optional[RearrangeTask] = None

        assert isinstance(self._config.SINGLE_EVAL_NODE, int)
        if self._config.SINGLE_EVAL_NODE >= 0:
            self._cur_node = self._config.SINGLE_EVAL_NODE

    def get_stage_goals(self) -> Dict[str, List[Predicate]]:
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

    def query(self, pred_s):
        pred = self.domain.predicate_lookup(pred_s)
        _, search_args = parse_func(pred_s)
        search_args = search_args.split(",")
        for pred in self.cur_state:
            if pred.name != pred.name:
                continue
            if pred.set_args is None:
                raise ValueError("unbound predicate in the current state")
            if len(pred.set_args) != len(search_args):
                raise ValueError("Predicate has wrong # of args")
            all_match = True
            for k1, k2 in zip(pred.set_args, search_args):
                if k2 == "*":
                    continue
                if k1 != k2:
                    all_match = False
                    break
            if all_match:
                return pred
        return None

    def load_solution(self, solution_d: Dict[str, Any]) -> List[Action]:
        """
        Loads the solution definition from the PDDL file and converts it to a
        list of executable actions.
        """
        solution = []
        for i, action in enumerate(solution_d):
            if (
                self._config.LIMIT_TASK_NODE != -1
                and i > self._config.LIMIT_TASK_NODE
            ):
                break
            name, args = parse_func(action)
            args = args.split(",")

            ac_instance = self.domain.actions[name].copy_new()

            ac_instance.bind(
                args, self.task_def.get("add_args", {}).get(i, {})
            )
            solution.append(ac_instance)
        return solution

    def _jump_to_node(
        self, node_idx: int, episode: Episode, is_full_task: bool = False
    ) -> None:
        """
        Sequentially applies all solution actions before `node_idx`. But NOT
        including the solution action at index `node_idx`.
        """

        logger.info("Jumping to node {node_idx}, is_full_task={is_full_task}")
        # We don't want to reset to this node if we are in full task mode.
        if not is_full_task:
            self._cur_node = node_idx

        for i in range(node_idx):
            self._solution[i].apply(
                self.domain.get_name_to_id_mapping(), self._sim
            )

        if node_idx in self.cached_tasks:
            self._cur_task = self.cached_tasks[node_idx]
            self._cur_task.reset()
        else:
            task = self._solution[node_idx].init_task(self, episode)
            self.cached_tasks[node_idx] = task
            self._cur_task = task

    def reset(self, episode: Episode):
        result = super().reset(episode)
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
            self._jump_to_node(
                self._config.DEBUG_SKIP_TO_NODE, episode, is_full_task=True
            )

        if self._cur_node >= 0:
            self._jump_to_node(self._cur_node, episode)

        self._inferred_cur_node_idx = 0
        self._inferred_cur_task = None
        self._increment_solution_subtask(episode)
        self.cached_tasks = {}
        return self._get_observations(episode)

    def get_inferred_node_idx(self) -> int:
        return self._inferred_cur_node_idx

    def get_inferrred_node_task(self) -> RearrangeTask:
        return self._inferred_cur_task

    def increment_inferred_solution_idx(self, episode: Episode) -> None:
        """
        Increment to the next index in the solution list. If the solution is
        exhausted then stay at the last index.
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
        task_solution = self.get_solution()
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
        if self._inferred_cur_node_idx in self.cached_tasks:
            self._inferred_cur_task = self.cached_tasks[
                self._inferred_cur_node_idx
            ]
            self._inferred_cur_task.reset(episode)
        else:
            task = task_solution[self._inferred_cur_node_idx].init_task(
                self, episode, should_reset=False
            )
            self.cached_tasks[self._inferred_cur_node_idx] = task
            self._inferred_cur_task = task
        self._sim.set_state(prev_state)

        return True

    def get_cur_task(self) -> RearrangeTask:
        return self._cur_task

    def get_cur_node(self) -> int:
        return self._cur_node

    def get_num_nodes(self) -> int:
        return len(self._solution)

    def get_solution(self) -> List[Action]:
        return self._solution

    def is_pred_list_sat(self, preds: List[Predicate]) -> bool:
        return all(self.domain.is_pred_true(pred) for pred in reversed(preds))

    def is_goal_state_satisfied(self) -> bool:
        return self.is_pred_list_sat(self._goal_state)

    def _try_get_subtask_prop(self, prop_name: str, def_val: Any) -> Any:
        """
        Try to get a property from the current inferred subtask. If the subtask
        is not valid, then return the supplied default value.
        """
        if self._cur_task is not None and hasattr(self._cur_node, prop_name):
            return getattr(self._cur_node, prop_name)
        elif self._inferred_cur_task is not None and hasattr(
            self._inferred_cur_task, prop_name
        ):
            return getattr(self._inferred_cur_task, prop_name)
        return def_val

    ###############################
    # Sub-task property overrides
    # These will emulate properties from sub-tasks needed to compute sub-task
    # sensors and measurements.
    ###############################
    @property
    def targ_idx(self):
        return self._try_get_subtask_prop("targ_idx", self._targ_idx)

    @property
    def nav_to_task_name(self):
        return self._try_get_subtask_prop("nav_to_task_name", None)

    @property
    def nav_to_obj_type(self) -> RearrangeObjectTypes:
        return self._try_get_subtask_prop(
            "nav_to_obj_type", RearrangeObjectTypes.RIGID_OBJECT
        )

    @property
    def nav_target_pos(self):
        return self._try_get_subtask_prop("nav_target_pos", np.zeros((3,)))

    @property
    def nav_target_angle(self):
        return self._try_get_subtask_prop("nav_target_angle", 0.0)
