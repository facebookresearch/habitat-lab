# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Optional, Union

from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    PddlSimInfo,
)


class LogicalExprType(Enum):
    AND = "and"
    NAND = "nand"
    OR = "or"
    NOR = "nor"


class LogicalQuantifierType(Enum):
    FORALL = "forall"
    EXISTS = "exists"


class LogicalExpr:
    """
    Refers to combinations of PDDL expressions or subexpressions.
    """

    def __init__(
        self,
        expr_type: LogicalExprType,
        sub_exprs: List[Union["LogicalExpr", Predicate]],
        inputs: List[PddlEntity],
        quantifier: Optional[LogicalQuantifierType],
    ):
        self._expr_type = expr_type
        self._sub_exprs = sub_exprs
        self._inputs = inputs
        self._quantifier = quantifier
        self._truth_vals: List[Optional[bool]] = []

    @property
    def prev_truth_vals(self) -> List[Optional[bool]]:
        """
        Sub-expression truth values for the last `self.is_true` computation. A
        value of None is if the truth value was not computed (due to early
        break).
        """
        return self._truth_vals

    @property
    def expr_type(self):
        return self._expr_type

    @property
    def inputs(self):
        return self._inputs

    @property
    def sub_exprs(self):
        return self._sub_exprs

    @sub_exprs.setter
    def sub_exprs(self, value):
        self._sub_exprs = value

    @property
    def quantifier(self):
        return self._quantifier

    def is_true_from_predicates(self, preds: List[Predicate]) -> bool:
        def check_statement(p):
            if isinstance(p, LogicalExpr):
                return p.is_true_from_predicates(preds)
            else:
                return p in preds

        return self._is_true(check_statement)

    def is_true(self, sim_info: PddlSimInfo) -> bool:
        return self._is_true(lambda p: p.is_true(sim_info))

    def _is_true(self, is_true_fn) -> bool:
        self._truth_vals = [None] * len(self._sub_exprs)

        if (
            self._expr_type == LogicalExprType.AND
            or self._expr_type == LogicalExprType.NAND
        ):
            result = True
            for i, sub_expr in enumerate(self._sub_exprs):
                truth_val = is_true_fn(sub_expr)
                if not isinstance(truth_val, bool):
                    raise ValueError(
                        f"Predicate returned non truth value: {sub_expr=}, {truth_val=}"
                    )
                self._truth_vals[i] = truth_val
                result = result and truth_val
                if not result:
                    break
        elif (
            self._expr_type == LogicalExprType.OR
            or self._expr_type == LogicalExprType.NOR
        ):
            result = False
            for i, sub_expr in enumerate(self._sub_exprs):
                truth_val = is_true_fn(sub_expr)
                assert isinstance(truth_val, bool)
                self._truth_vals[i] = truth_val
                result = result or truth_val
                if result:
                    break
        else:
            raise ValueError(
                f"Got unexpected expr_type: {self._expr_type} of type {type(self._expr_type)}"
            )

        if (
            self._expr_type == LogicalExprType.NAND
            or self._expr_type == LogicalExprType.NOR
        ):
            # Invert the entire result for NAND and NOR expressions.
            result = not result
        return result

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "LogicalExpr":
        self._sub_exprs = [e.sub_in(sub_dict) for e in self._sub_exprs]
        return self

    def sub_in_clone(self, sub_dict: Dict[PddlEntity, PddlEntity]):
        return LogicalExpr(
            self._expr_type,
            [e.sub_in_clone(sub_dict) for e in self._sub_exprs],
            self._inputs,
            self._quantifier,
        )

    def __repr__(self):
        return f"({self._expr_type}: {self._sub_exprs}"

    @property
    def compact_str(self):
        sub_s = ",".join((s.compact_str for s in self._sub_exprs))
        return f"{self._expr_type.value}({sub_s})"

    def clone(self) -> "LogicalExpr":
        return LogicalExpr(
            self._expr_type,
            [p.clone() for p in self._sub_exprs],
            self._inputs,
            self._quantifier,
        )
