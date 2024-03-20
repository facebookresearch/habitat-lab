#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import itertools
import os.path as osp
import time
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import yaml  # type: ignore[import]

from habitat.config.default import get_full_habitat_config_path
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr,
    LogicalExprType,
    LogicalQuantifierType,
)
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    PddlEntity,
    PddlSimInfo,
    SimulatorObjectType,
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
        read_config: bool = True,
    ):
        """
        :param domain_file_path: Either an absolute path or a path relative to
            `habitat/task/rearrange/multi_task/domain_configs/`.
        :param cur_task_config: The task config (`habitat.task`). Needed if the
            PDDL system will set the simulator state.
        """
        self._sim_info: Optional[PddlSimInfo] = None
        self._config = cur_task_config
        self._orig_actions: Dict[str, PddlAction] = {}

        if not osp.isabs(domain_file_path):
            parent_dir = osp.dirname(__file__)
            domain_file_path = osp.join(
                parent_dir, "domain_configs", domain_file_path
            )

        if "." not in domain_file_path.split("/")[-1]:
            domain_file_path += ".yaml"

        with open(get_full_habitat_config_path(domain_file_path), "r") as f:
            domain_def = yaml.safe_load(f)

        self._added_entities: Dict[str, PddlEntity] = {}
        self._added_expr_types: Dict[str, ExprType] = {}

        self._parse_expr_types(domain_def)
        self._parse_constants(domain_def)
        self._parse_predicates(domain_def)
        self._parse_actions(domain_def)

    @property
    def actions(self) -> Dict[str, PddlAction]:
        return self._actions

    def set_actions(self, actions: Dict[str, PddlAction]) -> None:
        self._orig_actions = actions
        self._actions = dict(actions)

    def _parse_actions(self, domain_def) -> None:
        """
        Fetches the PDDL actions into `self.actions`
        """

        for action_d in domain_def["actions"]:
            parameters = [
                PddlEntity(p["name"], self.expr_types[p["expr_type"]])
                for p in action_d["parameters"]
            ]
            name_to_param = {p.name: p for p in parameters}

            pre_cond = self.parse_only_logical_expr(
                action_d["precondition"], name_to_param
            )

            # Include the precondition quantifier inputs.
            postcond_entities = {
                **{x.name: x for x in pre_cond.inputs},
                **name_to_param,
            }
            post_cond = [
                self.parse_predicate(p, postcond_entities)
                for p in action_d["postcondition"]
            ]

            action = PddlAction(
                action_d["name"], parameters, pre_cond, post_cond
            )
            self._orig_actions[action.name] = action
        self._actions = dict(self._orig_actions)

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

            if "set_state_fn" not in pred_d:
                set_state_fn = None
            else:
                set_state_fn = _parse_callable(pred_d["set_state_fn"])

            if "is_valid_fn" not in pred_d:
                is_valid_fn = None
            else:
                is_valid_fn = _parse_callable(pred_d["is_valid_fn"])

            pred = Predicate(
                pred_d["name"],
                is_valid_fn,
                set_state_fn,
                arg_entities,
            )
            self.predicates[pred.name] = pred

    def _parse_constants(self, domain_def) -> None:
        """
        Fetches the constants into `self._constants`.
        """

        self._constants: Dict[str, PddlEntity] = {}
        if domain_def["constants"] is None:
            return
        for c in domain_def["constants"]:
            self._constants[c["name"]] = PddlEntity(
                c["name"],
                self.expr_types[c["expr_type"]],
            )

    def register_type(self, expr_type: ExprType):
        """
        Add a type to `self.expr_types`. Clears every episode
        """
        self._added_expr_types[expr_type.name] = expr_type

    def register_episode_entity(self, pddl_entity: PddlEntity) -> None:
        """
        Add an entity to appear in `self.all_entities`. Clears every episode.
        Note that `pddl_entity.name` should be unique. Otherwise, it will
        overide the existing object with that name.
        """
        self._added_entities[pddl_entity.name] = pddl_entity

    def _parse_expr_types(self, domain_def):
        """
        Fetches the types from the domain into `self._expr_types`.
        """

        # Always add the default `expr_types` from the simulator.
        base_entity = ExprType(SimulatorObjectType.BASE_ENTITY.value, None)
        self._expr_types: Dict[str, ExprType] = {
            SimulatorObjectType.BASE_ENTITY.value: base_entity
        }
        self._expr_types.update(
            {
                obj_type.value: ExprType(obj_type.value, base_entity)
                for obj_type in SimulatorObjectType
                if obj_type.value != SimulatorObjectType.BASE_ENTITY.value
            }
        )

        for parent_type, sub_types in domain_def["types"].items():
            if parent_type not in self._expr_types:
                self._expr_types[parent_type] = ExprType(
                    parent_type, base_entity
                )
            for sub_type in sub_types:
                if sub_type in self._expr_types:
                    self._expr_types[sub_type].parent = self._expr_types[
                        parent_type
                    ]
                else:
                    self._expr_types[sub_type] = ExprType(
                        sub_type, self._expr_types[parent_type]
                    )

    @property
    def expr_types(self) -> Dict[str, ExprType]:
        """
        Mapping from the name of the type to the ExprType definition.
        """
        return {**self._expr_types, **self._added_expr_types}

    def parse_predicate(
        self,
        pred_str: str,
        existing_entities: Optional[Dict[str, PddlEntity]] = None,
    ) -> Predicate:
        """
        Instantiates a predicate from call in string such as "in(X,Y)".
        :param pred_str: The string to parse such as "in(X,Y)".
        :param existing_entities: The valid entities for arguments in the
            predicate. If not specified, uses all defined entities.
        """
        if existing_entities is None:
            existing_entities = {}

        func_name, func_args = parse_func(pred_str)
        pred = self.predicates[func_name].clone()
        arg_values = []
        for func_arg in func_args:
            if func_arg in self.all_entities:
                v = self.all_entities[func_arg]
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

    def parse_only_logical_expr(
        self, load_d: Dict[str, Any], existing_entities: Dict[str, PddlEntity]
    ) -> LogicalExpr:
        """
        Parse a dict config into a `LogicalExpr`. Will only populate the
        `LogicalExpr` with the entities from `existing_entities`.
        """

        ret = self._parse_expr(load_d, existing_entities)
        if not isinstance(ret, LogicalExpr):
            raise ValueError(f"Expected logical expr, got {ret}")
        return ret

    def _parse_expr(
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
            self._parse_expr(
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
        env: RearrangeTask,
    ) -> None:
        """
        Attach the domain to the simulator. This does not bind any entity
        values, but creates `self._sim_info` which is needed to check simulator
        backed values (like truth values of predicates).
        """

        self._added_entities = {}
        self._added_expr_types = {}

        id_to_name = {}
        for k, i in sim.handle_to_object_id.items():
            id_to_name[i] = k

        self._sim_info = PddlSimInfo(
            sim=sim,
            env=env,
            expr_types=self.expr_types,
            obj_ids=sim.handle_to_object_id,
            target_ids={
                f"TARGET_{id_to_name[idx]}": idx
                for idx in sim.get_targets()[0]
            },
            art_handles={k.handle: i for i, k in enumerate(sim.art_objs)},
            marker_handles=sim.get_all_markers(),
            robot_ids={
                f"robot_{agent_id}": agent_id
                for agent_id in range(sim.num_articulated_agents)
            },
            all_entities=self.all_entities,
            predicates=self.predicates,
            receptacles=sim.receptacles,
        )
        # Ensure that all objects are accounted for.
        for entity in self.all_entities.values():
            self._sim_info.search_for_entity(entity)

    def bind_actions(self) -> None:
        """
        Expand all quantifiers in the actions. This should be done per instance
        bind in case the typing changes.
        """
        for k, ac in self._orig_actions.items():
            precond_quant = ac.precond.quantifier
            new_preconds, assigns = self.expand_quantifiers(ac.precond.clone())

            new_ac = ac.set_precond(new_preconds)
            if precond_quant == LogicalQuantifierType.EXISTS:
                # So the action post conditions can use the entities which
                # satisfy the pre-conditions.
                new_ac.set_post_cond_search(assigns)

            self._actions[k] = new_ac

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
            for entity_input in itertools.permutations(
                all_entities, pred.n_args
            ):
                if not pred.are_args_compatible(list(entity_input)):
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
        arguments. The same ordering of predicates is returned every time.
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
                poss_preds.append(use_pred)
        return sorted(poss_preds, key=lambda pred: pred.compact_str)

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

    def get_ordered_actions(self) -> List[PddlAction]:
        """
        Gets an ordered list of all possible PDDL actions in the environment
        based on the entities in the environment. Note that this is different
        from the agent actions. These are the PDDL actions as defined in the
        domain file.
        """
        return sorted(
            self.actions.values(),
            key=lambda x: x.name,
        )

    def get_entity(self, k: str) -> PddlEntity:
        """
        Gets an entity from the `all_entities` dictionary by key name.
        """

        return self.all_entities[k]

    def find_entities(self, entity_type: ExprType) -> Iterable[PddlEntity]:
        """
        Returns all the entities that match the condition.
        """
        for entity in self.all_entities.values():
            if entity.expr_type.is_subtype_of(entity_type):
                yield entity

    def get_ordered_entities_list(self) -> List[PddlEntity]:
        """
        Gets all entities sorted alphabetically by name.
        """

        return sorted(
            self.all_entities.values(),
            key=lambda x: x.name,
        )

    @property
    def all_entities(self) -> Dict[str, PddlEntity]:
        return {**self._constants, **self._added_entities}

    def expand_quantifiers(
        self, expr: LogicalExpr
    ) -> Tuple[LogicalExpr, List[Dict[PddlEntity, PddlEntity]]]:
        """
        Expand out a logical expression that could involve a quantifier into
        only logical expressions that don't involve any quantifier. Doesn't
        require the simulation to be grounded and expands using the current
        defined types.

        :returns: The expanded expression and the list of substitutions in the
            case of an EXISTS quantifier.
        """

        expr.sub_exprs = [
            (
                self.expand_quantifiers(subexpr)[0]
                if isinstance(subexpr, LogicalExpr)
                else subexpr
            )
            for subexpr in expr.sub_exprs
        ]

        if expr.quantifier == LogicalQuantifierType.FORALL:
            combine_type = LogicalExprType.AND
        elif expr.quantifier == LogicalQuantifierType.EXISTS:
            combine_type = LogicalExprType.OR
        elif expr.quantifier is None:
            return expr, []
        else:
            raise ValueError(f"Unrecognized {expr.quantifier}")

        t_start = time.time()
        assigns: List[List[PddlEntity]] = [[]]
        for expand_entity in expr.inputs:
            entity_assigns = []
            for e in self.all_entities.values():
                if not e.expr_type.is_subtype_of(expand_entity.expr_type):
                    continue
                for cur_assign in assigns:
                    if e in cur_assign:
                        continue
                    entity_assigns.append([*cur_assign, e])
            assigns = entity_assigns
        if self._sim_info is not None:
            self.sim_info.sim.add_perf_timing("assigns_search", t_start)

        t_start = time.time()
        assigns = [dict(zip(expr.inputs, assign)) for assign in assigns]
        expanded_exprs = []
        for assign in assigns:
            expanded_exprs.append(expr.sub_in_clone(assign))
        if self._sim_info is not None:
            self.sim_info.sim.add_perf_timing("expand_exprs_set", t_start)

        inputs: List[PddlEntity] = []
        return (
            LogicalExpr(combine_type, expanded_exprs, inputs, None),
            assigns,
        )


class PddlProblem(PddlDomain):
    stage_goals: Dict[str, LogicalExpr]
    init: List[Predicate]
    goal: LogicalExpr

    def __init__(
        self,
        domain_file_path: str,
        problem_file_path: str,
        cur_task_config: Optional["DictConfig"] = None,
        read_config: bool = True,
    ):
        self._objects = {}

        super().__init__(domain_file_path, cur_task_config, read_config)
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
            self.goal = self.parse_only_logical_expr(
                problem_def["goal"], self.all_entities
            )
            self.goal, _ = self.expand_quantifiers(self.goal)
        except Exception as e:
            raise ValueError(
                f"Could not parse goal cond {problem_def['goal']}"
            ) from e
        self.stage_goals = {}
        for stage_name, cond in problem_def["stage_goals"].items():
            expr = self.parse_only_logical_expr(cond, self.all_entities)
            self.stage_goals[stage_name], _ = self.expand_quantifiers(expr)

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
        self.bind_actions()

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
        return {**self._objects, **super().all_entities}


def _parse_callable(callable_d):
    full_fn_name = callable_d.pop("_target_")
    module_name, _, function_name = full_fn_name.rpartition(".")
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)

    return partial(fn, **callable_d)
