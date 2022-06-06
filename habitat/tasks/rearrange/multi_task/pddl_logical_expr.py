from enum import Enum
from functools import reduce
from typing import Dict, List, Union

from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity,
    PddlSimInfo,
)


class LogicalExprType(Enum):
    AND = "and"
    OR = "or"
    FORALL = "forall"


class LogicalExpr:
    def __init__(
        self,
        expr_type: LogicalExprType,
        sub_exprs: List[Union["LogicalExpr", Predicate]],
        inputs: List[PddlEntity],
    ):
        if expr_type == LogicalExprType.FORALL and len(sub_exprs) != 1:
            raise ValueError()

        self._expr_type = expr_type
        self._sub_exprs = sub_exprs
        self._inputs = inputs

    def is_true(self, sim_info: PddlSimInfo) -> bool:
        # if self._expr_type == LogicalExprType.FORALL:
        #     return self._sub_exprs
        if self._expr_type == LogicalExprType.AND:
            reduce_op = lambda x, y: x and y
            init_value = True
        elif self._expr_type == LogicalExprType.OR:
            reduce_op = lambda x, y: x or y
            init_value = False
        else:
            raise ValueError()

        return reduce(
            reduce_op,
            (sub_expr.is_true(sim_info) for sub_expr in self._sub_exprs),
            init_value,
        )

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "LogicalExpr":
        self._sub_exprs = [e.sub_in(sub_dict) for e in self._sub_exprs]
        return self

    def __repr__(self):
        return f"({self._expr_type}: {self._sub_exprs}"

    def clone(self) -> "LogicalExpr":
        return LogicalExpr(
            self._expr_type, [p.clone() for p in self._sub_exprs], self._inputs
        )
