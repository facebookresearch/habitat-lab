from enum import Enum
from functools import reduce
from typing import List, Union

from habitat.tasks.rearrange.multi_task.predicate import Predicate
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
        inputs: List[PddlEntity],
        sub_exprs: List[Union["LogicalExpr", Predicate]],
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
        elif self._expr_type == LogicalExprType.OR:
            reduce_op = lambda x, y: x or y
        else:
            raise ValueError()

        return reduce(
            reduce_op, (sub_expr.is_true() for sub_expr in self._sub_exprs)
        )
