#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from typing import Dict

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
        self._inf_cur_node: int = 0
        self._cur_task: RearrangeTask = None
        self.cached_tasks: Dict[str, RearrangeTask] = {}
        self.domain = None
        self._cur_state = None
        self._stage_goals = {}
        self._goal_state = None

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
            self._solution[i].apply(self.name_to_id, self._sim)

        if node_idx in self.cached_tasks:
            self._cur_task = self.cached_tasks[node_idx]
            self._cur_task.reset()
        else:
            task = self._solution[node_idx].init_task(self, episode)
            self.cached_tasks[node_idx] = task
            self._cur_task = task
        self._set_force_limit()

    def __getattr__(self, attr):
        pass

    def _set_force_limit(self):
        if self._cur_task is not None:
            self.use_max_accum_force = self._cur_task._config.MAX_ACCUM_FORCE
        is_subtask = self._config.SINGLE_EVAL_NODE >= 0
        if not is_subtask:
            if self.tcfg.MAX_ACCUM_FORCE != -1.0:
                self.use_max_accum_force = (
                    len(self._solution) * self.cur_task._config.MAX_ACCUM_FORCE
                )
            else:
                self.use_max_accum_force = -1.0
            if self._config.LIMIT_TASK_NODE != -1:
                self._max_episode_steps = (
                    self._config.LIMIT_TASK_LEN_SCALING
                    * (self.tcfg.LIMIT_TASK_NODE + 1)
                )

            self._cur_task = None
        else:
            self._max_episode_steps = self._cur_task._max_episode_steps
            self.use_max_accum_force = self._cur_task.use_max_accum_force

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

            self.load_solution(self.task_def["solution"])
            self._goal_state = self._parse_precond_list(
                self._config["GOAL_PRECOND"]
            )
            self._cur_state = self._parse_precond_list(start_d["precondition"])

            for k, preconds in self.task_def["stage_goals"].items():
                self._stage_goals[k] = self._parse_precond_list(preconds)

        self.start_state.set_state(self.name_to_id, self._sim)

        if self._config.DEBUG_SKIP_TO_NODE != -1:
            self._jump_to_node(
                self._config.DEBUG_SKIP_TO_NODE, episode, is_full_task=True
            )

        if self._cur_node >= 0:
            self._jump_to_node(self._cur_node, episode)

        self._set_force_limit()

        self._get_next_inf_sol()
        return super().reset(episode)

        return self.get_task_obs()

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
