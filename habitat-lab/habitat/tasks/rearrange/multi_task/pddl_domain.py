#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os.path as osp
from typing import TYPE_CHECKING, Dict, List, Optional, Union, cast

import yaml  # type: ignore[import]

from habitat.config.default import get_full_habitat_config_path
from habitat.core.dataset import Episode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.pddl_action import (
    ActionTaskInfo,
    PddlAction,
)
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr,
    LogicalExprType,
    LogicalQuantifierType,
)
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.pddl_sim_state import (
    ArtSampler,
    PddlRobotState,
    PddlSimState,
)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    PddlEntity,
    PddlSimInfo,
    parse_func,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PddlDomain:
    """
    Manages the information from the PDDL domain and task definition.
    """

    def __init__(
        self,
        domain_file_path: str,
        cur_task_config: Optional["DictConfig"] = None,
    ):
        """
        :param domain_file_path: Either an absolute path or a path relative to `habitat/task/rearrange/multi_task/domain_configs/`.
        :param cur_task_config: The task config (`habitat.task`). This is
            used when the action initializes a task via `PddlAction::init_task`. If
            this is not used, `cur_task_config` can be None.
        """
        self._sim_info: Optional[PddlSimInfo] = None
        self._config = cur_task_config

        if not osp.isabs(domain_file_path):
            parent_dir = osp.dirname(__file__)
            domain_file_path = osp.join(
                parent_dir, "domain_configs", domain_file_path
            )

        if "." not in domain_file_path:
            domain_file_path += ".yaml"

        with open(get_full_habitat_config_path(domain_file_path), "r") as f:
            domain_def = yaml.safe_load(f)

        self._parse_expr_types(domain_def)
        self._parse_constants(domain_def)
        self._parse_predicates(domain_def)
        self._parse_actions(domain_def)

    def _parse_actions(self, domain_def) -> None:
        """
        Fetches the PDDL actions into `self.actions`
        """

        self.actions: Dict[str, PddlAction] = {}
        for action_d in domain_def["actions"]:
            parameters = [
                PddlEntity(p["name"], self.expr_types[p["expr_type"]])
                for p in action_d["parameters"]
            ]
            name_to_param = {p.name: p for p in parameters}

            pre_cond = self._parse_only_logical_expr(
                action_d["precondition"], name_to_param
            )
            post_cond = [
                self.parse_predicate(p, name_to_param)
                for p in action_d["postcondition"]
            ]
            task_info_d = action_d["task_info"]
            full_entities = {**self._constants, **name_to_param}
            add_task_args = {
                k: full_entities[v]
                for k, v in task_info_d.get("add_task_args", {}).items()
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

    def _parse_predicates(self, domain_def) -> None:
        """
        Fetches the PDDL predicates into `self.predicates`.
        """

        self.predicates: Dict[str, Predicate] = {}
        for pred_d in domain_def["predicates"]:
            arg_entities = [
                PddlEntity(arg["name"], self.expr_types[arg["expr_type"]])
                for arg in pred_d["args"]
            ]
            pred_entities = {e.name: e for e in arg_entities}
            art_states = pred_d["set_state"].get("art_states", {})
            obj_states = pred_d["set_state"].get("obj_states", {})
            robot_states = pred_d["set_state"].get("robot_states", {})

            all_entities = {**self._constants, **pred_entities}

            art_states = {
                all_entities[k]: ArtSampler(**v) for k, v in art_states.items()
            }
            obj_states = {
                all_entities[k]: all_entities[v] for k, v in obj_states.items()
            }

            use_robot_states = {}
            for k, v in robot_states.items():
                use_k = all_entities[k]
                robot_pos = v.get("pos", None)
                holding = v.get("holding", None)

                use_robot_states[use_k] = PddlRobotState(
                    holding=all_entities.get(holding, holding),
                    should_drop=v.get("should_drop", False),
                    pos=all_entities.get(robot_pos, robot_pos),
                )

            set_state = PddlSimState(art_states, obj_states, use_robot_states)

            pred = Predicate(pred_d["name"], set_state, arg_entities)
            self.predicates[pred.name] = pred

    def _parse_constants(self, domain_def) -> None:
        """
        Fetches the constants into `self._constants`.
        """

        self._constants: Dict[str, PddlEntity] = {}
        for c in domain_def["constants"]:
            self._constants[c["name"]] = PddlEntity(
                c["name"],
                self.expr_types[c["expr_type"]],
            )

    def _parse_expr_types(self, domain_def):
        """
        Fetches the types from the domain into `self.expr_types`.
        """

        self.expr_types: Dict[str, ExprType] = {}
        self._leaf_exprs = []
        in_parent = []
        for parent_type, sub_types in domain_def["types"].items():
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

    @property
    def leaf_expr_types(self) -> List[ExprType]:
        return self._leaf_exprs

    def parse_predicate(
        self, pred_str: str, existing_entities: Dict[str, PddlEntity]
    ) -> Predicate:
        """
        Instantiates a predicate from call in string such as "in(X,Y)".
        :param pred_str: The string to parse such as "in(X,Y)".
        :param existing_entities: The valid entities for arguments in the predicate.
        """

        func_name, func_args = parse_func(pred_str)
        pred = self.predicates[func_name].clone()
        arg_values = []
        for func_arg in func_args:
            if func_arg in self._constants:
                v = self._constants[func_arg]
            elif func_arg in existing_entities:
                v = existing_entities[func_arg]
            else:
                raise ValueError(
                    f"Could not find entity {func_arg} in predicate `{pred_str}` (args={func_args} name={func_name})"
                )
            arg_values.append(v)
        try:
            pred.set_param_values(arg_values)
        except Exception as e:
            raise ValueError(
                f"Problem setting predicate values {pred} with {arg_values}"
            ) from e
        return pred

    def _parse_only_logical_expr(
        self, load_d, existing_entities: Dict[str, PddlEntity]
    ) -> LogicalExpr:
        ret = self._parse_logical_expr(load_d, existing_entities)
        if not isinstance(ret, LogicalExpr):
            raise ValueError(f"Expected logical expr, got {ret}")
        return ret

    def _parse_logical_expr(
        self, load_d, existing_entities: Dict[str, PddlEntity]
    ) -> Union[LogicalExpr, Predicate]:
        """
        Similar to `self.parse_predicate` for logical expressions. If `load_d`
        is a string, it will be parsed as a predicate.
        """

        if load_d is None:
            return LogicalExpr(LogicalExprType.AND, [], [], None)

        if isinstance(load_d, str):
            # This can be assumed to just be a predicate
            return self.parse_predicate(load_d, existing_entities)
        if isinstance(load_d, list):
            raise TypeError(
                f"Could not parse logical expr {load_d}. You likely need to nest the predicate list in a logical expression"
            )

        try:
            expr_type = LogicalExprType[load_d["expr_type"]]
        except Exception as e:
            raise ValueError(f"Could not load expr_type from {load_d}") from e

        inputs = load_d.get("inputs", [])
        inputs = [
            PddlEntity(x["name"], self.expr_types[x["expr_type"]])
            for x in inputs
        ]

        sub_exprs = [
            self._parse_logical_expr(
                sub_expr, {**existing_entities, **{x.name: x for x in inputs}}
            )
            for sub_expr in load_d["sub_exprs"]
        ]
        quantifier = load_d.get("quantifier", None)
        if quantifier is not None:
            quantifier = LogicalQuantifierType[quantifier]
        return LogicalExpr(expr_type, sub_exprs, inputs, quantifier)

    def bind_to_instance(
        self,
        sim: RearrangeSim,
        dataset: RearrangeDatasetV0,
        env: RearrangeTask,
        episode: Episode,
    ) -> None:
        """
        Attach the domain to the simulator. This does not bind any entity
        values, but creates `self._sim_info` which is needed to check simulator
        backed values (like truth values of predicates).
        """

        id_to_name = {}
        for k, i in sim.ref_handle_to_rigid_obj_id.items():
            id_to_name[i] = k

        self._sim_info = PddlSimInfo(
            sim=sim,
            dataset=dataset,
            env=env,
            episode=episode,
            obj_thresh=self._config.obj_succ_thresh,
            art_thresh=self._config.art_succ_thresh,
            robot_at_thresh=self._config.robot_at_thresh,
            expr_types=self.expr_types,
            obj_ids=sim.ref_handle_to_rigid_obj_id,
            target_ids={
                f"TARGET_{id_to_name[idx]}": idx
                for idx in sim.get_targets()[0]
            },
            art_handles={k.handle: i for i, k in enumerate(sim.art_objs)},
            marker_handles=sim.get_all_markers(),
            robot_ids={
                f"robot_{robot_id}": robot_id
                for robot_id in range(sim.num_robots)
            },
            all_entities=self.all_entities,
            predicates=self.predicates,
        )
        # Ensure that all objects are accounted for.
        for entity in self.all_entities.values():
            self._sim_info.search_for_entity_any(entity)

    @property
    def sim_info(self) -> PddlSimInfo:
        """
        Info from the simulator instance needed to interface the planning
        domain with the simulator. This property is used for all calls in the
        PDDL code that must interface with the simulator.
        """

        if self._sim_info is None:
            raise ValueError("Need to first bind to simulator instance.")
        return self._sim_info

    def apply_action(self, action: PddlAction) -> None:
        """
        Helper to apply an action with the simulator info.
        """

        action.apply(self.sim_info)

    def is_expr_true(self, expr: LogicalExpr) -> bool:
        """
        Helper to check expression truth value from simulator info.
        """

        return expr.is_true(self.sim_info)

    def get_true_predicates(self) -> List[Predicate]:
        """
        Get all the predicates that are true in the current simulator state.
        """

        all_entities = self.all_entities.values()
        true_preds: List[Predicate] = []
        for pred in self.predicates.values():
            for entity_input in itertools.combinations(
                all_entities, pred.n_args
            ):
                if not pred.are_args_compatible(entity_input):
                    continue

                use_pred = pred.clone()
                use_pred.set_param_values(entity_input)

                if use_pred.is_true(self.sim_info):
                    true_preds.append(use_pred)
        return true_preds

    def get_possible_predicates(self) -> List[Predicate]:
        """
        Get all predicates that COULD be true. This is independent of the
        simulator state and is the set of compatible predicate and entity
        arguments.
        """

        all_entities = self.all_entities.values()
        poss_preds: List[Predicate] = []
        for pred in self.predicates.values():
            for entity_input in itertools.combinations(
                all_entities, pred.n_args
            ):
                if not pred.are_args_compatible(entity_input):
                    continue

                use_pred = pred.clone()
                use_pred.set_param_values(entity_input)
                if use_pred.are_types_compatible(self.expr_types):
                    poss_preds.append(use_pred)
        return poss_preds

    def get_possible_actions(
        self,
        filter_entities: Optional[List[PddlEntity]] = None,
        allowed_action_names: Optional[List[str]] = None,
        restricted_action_names: Optional[List[str]] = None,
        true_preds: Optional[List[Predicate]] = None,
    ) -> List[PddlAction]:
        """
        Get all actions that can be applied.
        :param filter_entities: ONLY actions with entities that contain all
            entities in `filter_entities` are allowed.
        :param allowed_action_names: ONLY action names allowed.
        :param restricted_action_names: Action names NOT allowed.
        """
        if filter_entities is None:
            filter_entities = []
        if restricted_action_names is None:
            restricted_action_names = []

        all_entities = list(self.all_entities.values())
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

                for entity_input_perm in itertools.permutations(entity_input):
                    entity_inputs = cast(List[PddlEntity], entity_input_perm)
                    if not action.are_args_compatible(entity_inputs):
                        continue
                    new_action = action.clone()
                    new_action.set_param_values(entity_inputs)
                    if (
                        true_preds is not None
                        and not new_action.is_precond_satisfied_from_predicates(
                            true_preds
                        )
                    ):
                        continue
                    matching_actions.append(new_action)
        return matching_actions

    @property
    def all_entities(self) -> Dict[str, PddlEntity]:
        return self._constants


class PddlProblem(PddlDomain):
    stage_goals: Dict[str, LogicalExpr]
    init: List[Predicate]
    goal: LogicalExpr

    def __init__(
        self,
        domain_file_path: str,
        problem_file_path: str,
        cur_task_config: Optional["DictConfig"] = None,
    ):
        super().__init__(domain_file_path, cur_task_config)
        with open(get_full_habitat_config_path(problem_file_path), "r") as f:
            problem_def = yaml.safe_load(f)
        self._objects = {
            o["name"]: PddlEntity(o["name"], self.expr_types[o["expr_type"]])
            for o in problem_def["objects"]
        }

        self.init = [
            self.parse_predicate(p, self._objects)
            for p in problem_def.get("init", [])
        ]
        try:
            self.goal = self._parse_only_logical_expr(
                problem_def["goal"], self.all_entities
            )
            self.goal = self.expand_quantifiers(self.goal)
        except Exception as e:
            raise ValueError(
                f"Could not parse goal cond {problem_def['goal']}"
            ) from e
        self.stage_goals = {}
        for stage_name, cond in problem_def["stage_goals"].items():
            expr = self._parse_only_logical_expr(cond, self.all_entities)
            self.stage_goals[stage_name] = self.expand_quantifiers(expr)

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

                try:
                    action.set_param_values(arg_values)
                except Exception as e:
                    raise ValueError(
                        f"Problem setting action {action} with {arg_values} in solution list"
                    ) from e

                self._solution.append(action)

        for action in self.actions.values():
            action.set_precond(self.expand_quantifiers(action.precond))

    @property
    def solution(self):
        """
        Sequence of actions to solve the task specified in the problem file.
        """

        if self._solution is None:
            raise ValueError("Solution is not supported by this PDDL")
        return self._solution

    @property
    def all_entities(self) -> Dict[str, PddlEntity]:
        return {**self._objects, **self._constants}

    def get_entity(self, k: str) -> PddlEntity:
        return self.all_entities[k]

    def get_ordered_entities_list(self) -> List[PddlEntity]:
        return sorted(
            self.all_entities.values(),
            key=lambda x: x.name,
        )

    def get_ordered_actions(self) -> List[PddlAction]:
        return sorted(
            self.actions.values(),
            key=lambda x: x.name,
        )

    def expand_quantifiers(self, expr: LogicalExpr) -> LogicalExpr:
        """
        Expand out a logical expression that could involve a quantifier into
        only logical expressions that don't involve any quantifier.
        """

        expr.sub_exprs = [
            self.expand_quantifiers(subexpr)
            if isinstance(subexpr, LogicalExpr)
            else subexpr
            for subexpr in expr.sub_exprs
        ]

        if expr.quantifier == LogicalQuantifierType.FORALL:
            combine_type = LogicalExprType.AND
        elif expr.quantifier == LogicalQuantifierType.EXISTS:
            combine_type = LogicalExprType.OR
        elif expr.quantifier is None:
            return expr
        else:
            raise ValueError(f"Unrecongized {expr.quantifier}")

        all_matching_entities = []
        for expand_entity in expr.inputs:
            all_matching_entities.append(
                [
                    e
                    for e in self.all_entities.values()
                    if e.expr_type.is_subtype_of(expand_entity.expr_type)
                ]
            )

        expanded_exprs: List[Union[LogicalExpr, Predicate]] = []
        for poss_input in itertools.product(*all_matching_entities):
            assert len(poss_input) == len(expr.inputs)
            sub_dict = {
                expand_entity: sub_entity
                for expand_entity, sub_entity in zip(expr.inputs, poss_input)
            }

            expanded_exprs.append(expr.clone().sub_in(sub_dict))

        inputs: List[PddlEntity] = []
        return LogicalExpr(combine_type, expanded_exprs, inputs, None)
