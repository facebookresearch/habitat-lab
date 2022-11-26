# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import reduce
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
        if (
            self._expr_type == LogicalExprType.AND
            or self._expr_type == LogicalExprType.NAND
        ):
            reduce_op = lambda x, y: x and y
            init_value = True
        elif (
            self._expr_type == LogicalExprType.OR
            or self._expr_type == LogicalExprType.NOR
        ):
            reduce_op = lambda x, y: x or y
            init_value = False
        else:
            raise ValueError()

        ret = reduce(
            reduce_op,
            (is_true_fn(sub_expr) for sub_expr in self._sub_exprs),
            init_value,
        )
        if (
            self._expr_type == LogicalExprType.NAND
            or self._expr_type == LogicalExprType.NOR
        ):
            ret = not ret
        return ret

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "LogicalExpr":
        self._sub_exprs = [e.sub_in(sub_dict) for e in self._sub_exprs]
        return self

    def __repr__(self):
        return f"({self._expr_type}: {self._sub_exprs}"

    def clone(self) -> "LogicalExpr":
        return LogicalExpr(
            self._expr_type,
            [p.clone() for p in self._sub_exprs],
            self._inputs,
            self._quantifier,
        )
