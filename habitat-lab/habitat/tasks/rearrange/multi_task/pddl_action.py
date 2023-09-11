#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    PddlSimInfo,
    do_entity_lists_match,
    ensure_entity_lists_match,
)


class PddlAction:
    def __init__(
        self,
        name: str,
        parameters: List[PddlEntity],
        pre_cond: LogicalExpr,
        post_cond: List[Predicate],
        post_cond_search: Optional[List[Dict[PddlEntity, PddlEntity]]] = None,
    ):
        """
        Models the PDDL acton entity.

        :param parameters: The parameters to the PDDL action in the domain file.
        :param pre_cond: The pre condition of the PDDL action.
        :param post_cond: The post conditions of the PDDL action.
        :param post_cond_search: Mapping expanded quantifier inputs from the
            pre-condition to ungrounded entities in the post-condition. One
            mapping per quantifier expansion.
        """
        if not isinstance(pre_cond, LogicalExpr):
            raise ValueError(f"Incorrect type {pre_cond}")

        self._name = name
        self._params = parameters
        self.name_to_param = {p.name: p for p in self._params}
        self._param_values: Optional[List[PddlEntity]] = None
        self._pre_cond = pre_cond
        self._post_cond = post_cond
        self._post_cond_search = post_cond_search

    @property
    def post_cond(self) -> List[Predicate]:
        return self._post_cond

    def set_post_cond_search(
        self, post_cond_search: List[Dict[PddlEntity, PddlEntity]]
    ) -> None:
        self._post_cond_search = post_cond_search

    def apply_if_true(self, sim_info: PddlSimInfo) -> bool:
        """
        Apply the action post-condition to the simulator if the action
        pre-condition is true. This will also dynamically select the right
        entities for the post-condition based on the pre-condition quantifiers.
        """
        is_sat = self._pre_cond.is_true(sim_info)
        if not is_sat:
            return False
        self.apply(sim_info)

        return True

    def get_arg_value(self, param_name: str) -> Optional[PddlEntity]:
        """
        Get the assigned value of a parameter with name `param_name`. Returns
        `None` if the parameter is not yet assigned.
        """

        for param, param_value in zip(self._params, self._param_values):
            if param.name == param_name:
                return param_value
        return None

    def __repr__(self):
        return (
            f"<Action {self._name} ({self._params})->({self._param_values})>"
        )

    @property
    def compact_str(self) -> str:
        """
        Display string of the action.
        """
        params = ",".join([x.name for x in self._param_values])
        return f"{self._name}({params})"

    def is_precond_satisfied_from_predicates(
        self, predicates: List[Predicate]
    ) -> bool:
        """
        Checks if the preconditions of the action are satisfied from the input
        predicates ALONE.
        :param predicates: The set of predicates currently true in the
            environment.
        """

        return self._pre_cond.is_true_from_predicates(predicates)

    def set_precond(self, new_precond) -> "PddlAction":
        """
        Sets the preconditions for the action.
        """
        return PddlAction(
            self._name,
            self._params,
            new_precond,
            self._post_cond,
            self._post_cond_search,
        )

    @property
    def precond(self):
        return self._pre_cond

    @property
    def name(self):
        return self._name

    @property
    def n_args(self):
        return len(self._params)

    def are_args_compatible(self, arg_values: List[PddlEntity]) -> bool:
        return do_entity_lists_match(self._params, arg_values)

    def set_param_values(self, param_values: List[PddlEntity]) -> None:
        """
        Bind the parameters to PDDL entities. An exception is thrown if the arguments don't match (like different number of arguments or wrong type).
        """

        param_values = list(param_values)
        if self._param_values is not None:
            raise ValueError(
                f"Trying to set arg values with {param_values} when current args are set to {self._param_values}"
            )
        ensure_entity_lists_match(self._params, param_values)
        self._param_values = param_values

        sub_dict = dict(zip(self._params, self._param_values))

        # Substitute into the post and pre conditions
        self._param_values = [sub_dict.get(p, p) for p in self._param_values]
        self._post_cond = [p.sub_in(sub_dict) for p in self._post_cond]
        self._pre_cond = self._pre_cond.sub_in(sub_dict)

    def clone(self) -> "PddlAction":
        """
        Clones the action potentially with a new name.
        """
        return PddlAction(
            self._name,
            self._params,
            self._pre_cond.clone(),
            [p.clone() for p in self._post_cond],
            self._post_cond_search,
        )

    def apply(self, sim_info: PddlSimInfo) -> None:
        post_conds = self._post_cond
        if self._post_cond_search is not None:
            found_assign = None
            assert len(self._pre_cond.prev_truth_vals) == len(
                self._post_cond_search
            )
            for sat, assign in zip(
                self._pre_cond.prev_truth_vals, self._post_cond_search
            ):
                if sat is not None and sat:
                    found_assign = assign
                    break
            assert found_assign is not None
            # Clone and sub in so we don't overwrite the original predicates
            post_conds = [p.clone().sub_in(found_assign) for p in post_conds]

        for p in post_conds:
            p.set_state(sim_info)

    @property
    def params(self) -> List[PddlEntity]:
        return self._params

    @property
    def param_values(self) -> Optional[List[PddlEntity]]:
        if self._param_values is None:
            raise ValueError(
                "Accessing action param values before they are set."
            )
        if len(self._param_values) != len(self._params):
            raise ValueError()
        return self._param_values
