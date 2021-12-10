#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from typing import Dict

import magnum as mn
import yaml

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    SetState,
    parse_func,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


@registry.register_task(name="RearrangeCompositeTask-v0")
class CompositeTask(RearrangeTask):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(config=config, *args, dataset=dataset, **kwargs)

        task_spec_path = osp.join(
            self._config.TASK_SPEC_BASE_PATH, self._config.TASK_SPEC + ".yaml"
        )

        with open(task_spec_path, "r") as f:
            task_def = yaml.safe_load(f)
        self.task_def = task_def

        start_d = task_def["start"]
        self.start_state = SetState(start_d["state"])

        self._cur_node: int = -1
        self._cur_task: RearrangeTask = None
        self.cached_tasks: Dict[str, RearrangeTask] = {}
        self.domain = None
        self._cur_state = None
        self._stage_goals = {}
        self._goal_state = None
        self._solution = None

        self._inf_cur_node = -1
        self._inf_cur_task = None

        assert isinstance(self._config.SINGLE_EVAL_NODE, int)
        if self._config.SINGLE_EVAL_NODE >= 0:
            self._cur_node = self._config.SINGLE_EVAL_NODE

    def _load_stage_preconds(self, stage_goals):
        self._stage_goals = {}
        for k, preconds in stage_goals.items():
            self._stage_goals[k] = self._parse_precond_list(preconds)

    def get_stage_goals(self):
        return self._stage_goals

    def _parse_precond_list(self, d):
        preds = []
        for pred_s in d:
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

    def load_solution(self, solution_d):
        solution = []
        for i, action in enumerate(solution_d):
            if (
                self._config.LIMIT_TASK_NODE != -1
                and i > self._config.LIMIT_TASK_NODE
            ):
                break
            name, args = parse_func(action)
            args = args.split(",")
            ac_instance = copy.deepcopy(self.domain.actions[name])

            ac_instance.bind(
                args, self.task_def.get("add_args", {}).get(i, {})
            )
            solution.append(ac_instance)
        return solution

    def _jump_to_node(self, node_idx, episode, is_full_task=False):
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
        super().reset(episode)
        if self.domain is None:
            start_d = self.task_def["start"]
            self.domain = PddlDomain(
                self._config.PDDL_DOMAIN_DEF,
                self._dataset,
                self._config,
                self._sim,
            )

            self._solution = self.load_solution(self.task_def["solution"])
            self._goal_state = self._parse_precond_list(self.task_def["goal"])
            self._cur_state = self._parse_precond_list(start_d["precondition"])

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

        self._inf_cur_node = 0
        self._inf_cur_task = None
        self._get_next_inf_sol(episode)
        return super().reset(episode)

    def get_inf_cur_node(self):
        return self._inf_cur_node

    def get_inf_cur_task(self):
        return self._inf_cur_task

    def increment_inf_sol(self, episode):
        prev_inf_cur_node = self._inf_cur_node
        self._inf_cur_node += 1
        if not self._get_next_inf_sol(episode):
            self._inf_cur_node = prev_inf_cur_node

    def _get_next_inf_sol(self, episode):
        # Never give reward from these nodes, skip to the next node instead.
        # Returns False if there is no next subtask in the solution
        task_solution = self.get_solution()
        if self._inf_cur_node >= len(task_solution):
            return False
        while (
            task_solution[self._inf_cur_node].name in self._config.SKIP_NODES
        ):
            self._inf_cur_node += 1
            if self._inf_cur_node >= len(task_solution):
                return False

        prev_state = self._sim.capture_state(with_robot_js=True)
        if self._inf_cur_node in self.cached_tasks:
            self._inf_cur_task = self.cached_tasks[self._inf_cur_node]
            self._inf_cur_task.reset(episode)
        else:
            task = task_solution[self._inf_cur_node].init_task(
                self, episode, should_reset=False
            )
            self.cached_tasks[self._inf_cur_node] = task
            self._inf_cur_task = task
        self._sim.set_state(prev_state)

        return True

    def get_cur_task(self):
        return self._cur_task

    def get_cur_node(self):
        return self._cur_node

    def get_num_nodes(self):
        return len(self._solution)

    def get_solution(self):
        return self._solution

    def is_pred_list_sat(self, preds):
        return all(self.domain.is_pred_true(pred) for pred in reversed(preds))

    def is_goal_state_satisfied(self):
        return self.is_pred_list_sat(self._goal_state)

    def _try_get_subtask_prop(self, prop_name, def_val):
        if self._cur_task is not None and hasattr(self._cur_node, prop_name):
            return getattr(self._cur_node, prop_name)
        elif self._inf_cur_task is not None and hasattr(
            self._inf_cur_task, prop_name
        ):
            return getattr(self._inf_cur_task, prop_name)
        return def_val

    ###############################
    # Sub-task property overrides
    ###############################
    @property
    def targ_idx(self):
        return self._try_get_subtask_prop("targ_idx", self._targ_idx)

    @property
    def nav_target_pos(self):
        return self._try_get_subtask_prop(
            "nav_target_pos", mn.Vector3(0.0, 0.0, 0.0)
        )

    @property
    def nav_target_angle(self):
        return self._try_get_subtask_prop("nav_target_angle", 0.0)
