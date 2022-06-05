#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Dict, List, Optional, Union

import yaml

from habitat import Config
from habitat.core.dataset import Episode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_action import (
    ActionTaskInfo,
    PddlAction,
)
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr,
    LogicalExprType,
)
from habitat.tasks.rearrange.multi_task.pddl_set_state import (
    ArtSampler,
    PddlRobotState,
    PddlSetState,
)
from habitat.tasks.rearrange.multi_task.predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    PddlEntity,
    PddlSimInfo,
    parse_func,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


class PddlDomain:
    """
    Manages the information from the PDDL domain and task definition.
    """

    def __init__(
        self,
        domain_file_path: str,
        cur_task_config: Optional[Config] = None,
    ):
        self._sim_info: Optional[PddlSimInfo] = None
        self._config = cur_task_config
        with open(domain_file_path, "r") as f:
            domain_def = yaml.safe_load(f)

        self.expr_types: Dict[str, ExprType] = {}
        self._leaf_exprs = []
        in_parent = []
        for parent_type, sub_types in domain_def["types"]:
            if parent_type not in self.expr_types:
                self.expr_types[parent_type] = ExprType(parent_type, None)
            in_parent.append(parent_type)
            for sub_type in sub_types:
                self.expr_types[sub_type] = ExprType(
                    sub_type, self.expr_types[parent_type]
                )
        self._leaf_exprs = [
            expr_type
            for expr_type in self.expr_types.values()
            if expr_type.name not in in_parent
        ]

        self._constants: Dict[str, PddlEntity] = {}
        for c in domain_def["_constants"]:
            self._constants[c["name"]] = PddlEntity(
                c["name"],
                self.expr_types[c["expr_type"]],
            )

        self.predicates: Dict[str, Predicate] = {}
        for pred_d in domain_def["predicates"]:
            arg_entities = [
                PddlEntity(arg.name, self.expr_types[arg.expr_type])
                for arg in pred_d["args"]
            ]
            pred_entities = {e.name: e for e in arg_entities}
            art_states = pred_d["set_state"].get("art_states", {})
            obj_states = pred_d["set_state"].get("obj_states", {})
            robot_states = pred_d["set_state"].get("robot_states", {})

            all_entites = {**self._constants, **pred_entities}

            art_states = {
                all_entites[k]: ArtSampler(
                    **v, thresh=self._config.ART_SUCC_THRESH
                )
                for k, v in art_states
            }
            obj_states = {all_entites[k]: v for k, v in obj_states}
            robot_states = {
                all_entites[k]: PddlRobotState(**v) for k, v in robot_states
            }

            set_state = PddlSetState(art_states, obj_states, robot_states)

            pred = Predicate(pred_d["name"], set_state, arg_entities)
            self.predicates[pred.name] = pred

        self.actions: Dict[str, PddlAction] = {}
        for action_d in self.domain_def["actions"]:
            parameters = [
                PddlEntity(p["name"], self.expr_types[p["expr_type"]])
                for p in action_d["parameters"]
            ]
            name_to_param = {p.name: p for p in parameters}

            pre_cond = self.parse_logical_expr(
                action_d["precondition"], name_to_param
            )
            post_cond = [
                self.parse_predicate(p, name_to_param)
                for p in action_d["postcondition"]
            ]
            task_info_d = action_d["task_info"]
            add_task_args = {
                k: self._constants[v]
                for k, v in task_info_d["add_task_args"].items()
            }

            task_info = ActionTaskInfo(
                task_config=self._config,
                task=task_info_d["task"],
                task_def=task_info_d["task_def"],
                config_args=task_info_d["config_args"],
                add_task_args=add_task_args,
            )
            action = PddlAction(
                action_d["name"], parameters, pre_cond, post_cond, task_info
            )
            self.actions[action.name] = action

    @property
    def leaf_expr_types(self) -> List[ExprType]:
        return self._leaf_exprs

    def parse_predicate(
        self, pred_str: str, existing_entities: Dict[str, PddlEntity]
    ) -> Predicate:
        func_name, func_args = parse_func(pred_str)
        pred = self.predicates[func_name].clone()
        arg_values = []
        for func_arg in func_args:
            if func_arg in self._constants:
                v = self._constants[func_arg]
            elif func_arg in existing_entities:
                v = existing_entities[func_arg]
            else:
                raise ValueError(f"Could not find entity {func_arg}")
            arg_values.append(v)
        pred.set_param_values(arg_values)
        return pred

    def parse_logical_expr(
        self, load_d, existing_entities: Dict[str, PddlEntity]
    ) -> Union[LogicalExpr, Predicate]:
        if isinstance(load_d, str):
            # This can be assumed to just be a predicate
            return self.parse_predicate(load_d, existing_entities)

        expr_type = LogicalExprType[load_d[["expr_type"]]]
        sub_exprs = [
            self.parse_logical_expr(sub_expr, existing_entities)
            for sub_expr in load_d["sub_exprs"]
        ]
        return LogicalExpr(expr_type, sub_exprs)

    def bind_to_instance(
        self,
        sim: RearrangeSim,
        dataset: RearrangeDatasetV0,
        env: RearrangeTask,
        episode: Episode,
    ) -> None:
        id_to_name = {}
        for k, i in self._sim.ref_handle_to_rigid_obj_id.items():
            id_to_name[i] = k

        self._sim_info = PddlSimInfo(
            sim=sim,
            dataset=dataset,
            env=env,
            episode=episode,
            obj_thresh=self._config.OBJ_SUCC_THRESH,
            art_thresh=self._config.ART_SUCC_THRESH,
            expr_types=self.expr_types,
            obj_ids=self._sim.ref_handle_to_rigid_obj_id,
            target_ids={
                f"TARGET_{id_to_name[idx]}": idx
                for idx in self._sim.get_targets()[0]
            },
            art_handles={k: i for i, k in enumerate(self._sim.art_objs)},
            marker_handles=self._sim.get_all_markers(),
            robot_ids={
                f"ROBOT_{robot_id}": robot_id
                for robot_id in range(self._sim.num_robots)
            },
        )

    @property
    def sim_info(self) -> PddlSimInfo:
        if self._sim_info is None:
            raise ValueError("Need to first bind to simulator instance.")
        return self._sim_info

    def set_pred_states(self, preds: List[Predicate]) -> None:
        for pred in preds:
            pred.set_state(self.sim_info)

    def apply_action(self, action: PddlAction) -> None:
        action.apply(self.sim_info)

    def is_expr_true(self, expr: LogicalExpr) -> bool:
        return expr.is_true(self.sim_info)

    def get_true_predicates(self) -> List[Predicate]:
        all_entities = self.all_entities
        true_preds: List[Predicate] = []
        for pred in self.predicates.values():
            for entity_input in itertools.combinations(
                all_entities, pred.n_args
            ):
                if not pred.are_args_compatible(entity_input):
                    continue

                pred = pred.clone()
                pred.set_param_values(entity_input)

                if pred.is_true(self.sim_info):
                    true_preds.append(pred)
        return true_preds

    def get_possible_actions(
        self,
        filter_entities: Optional[List[PddlEntity]] = None,
        allowed_action_names: Optional[List[str]] = None,
        restricted_action_names: Optional[List[str]] = None,
        true_preds: Optional[List[Predicate]] = None,
    ) -> List[PddlAction]:
        if true_preds is None:
            true_preds = self.get_true_predicates()
        if filter_entities is None:
            filter_entities = []
        if restricted_action_names is None:
            restricted_action_names = []

        all_entities = self.all_entities
        matching_actions = []
        for action in self.actions.values():
            if (
                allowed_action_names is not None
                and action.name not in allowed_action_names
            ):
                continue

            if action.name in restricted_action_names:
                continue

            for entity_input in itertools.combinations(
                all_entities, action.n_args
            ):
                # Check that all the filter_entities are in entity_input
                matches_filter = all(
                    filter_entity in entity_input
                    for filter_entity in filter_entities
                )
                if not matches_filter:
                    continue

                if not action.are_args_compatible(entity_input):
                    continue
                new_action = action.clone()
                new_action.set_param_values(entity_input)
                matching_actions.append(new_action)
        return matching_actions

    @property
    def all_entities(self) -> Dict[str, PddlEntity]:
        return self._constants


class PddlProblem:
    stage_goals: Dict[str, LogicalExpr]
    init: List[Predicate]
    goal: LogicalExpr

    def __init__(self, pddl_domain: PddlDomain, problem_file_path: str):
        with open(problem_file_path, "r") as f:
            problem_def = yaml.safe_load(f)
        self._objects = {
            o["name"]: PddlEntity(o["name"], pddl_domain.expr_types[o["type"]])
            for o in problem_def["_objects"]
        }
        self.init = [
            pddl_domain.parse_predicate(p, self._objects)
            for p in problem_def["init"]
        ]
        self.goal = pddl_domain.parse_logical_expr(problem_def["goal"])
        self.stage_goals = {}
        for stage_name, cond in problem_def["stage_goals"].items():
            self.stage_goals[stage_name] = pddl_domain.parse_logical_expr(
                cond, self.all_entities
            )

        self._solution: Optional[List[PddlAction]] = None
        if "solution" in problem_def:
            self._solution = []
            for sol in problem_def["solution"]:
                action_name, action_args = parse_func(sol)
                action = self.actions[action_name].clone()
                arg_values = []
                for action_arg in action_args:
                    if action_arg in self.all_entities:
                        v = self.all_entities[action_arg]
                    elif action_arg in action.name_to_param:
                        v = action.name_to_param[action_arg]
                    else:
                        raise ValueError(f"Could not find entity {action_arg}")
                    arg_values.append(v)

                action.set_param_values(arg_values)

                self._solution.append(action)

    def solution(self):
        if self._solution is None:
            raise ValueError("Solution is not supported by this PDDL")
        return self._solution

    @property
    def all_entities(self) -> Dict[str, PddlEntity]:
        return {**self._objects, **self._constants}

    def get_entity(self, k: str) -> PddlEntity:
        return self.all_entities[k]
