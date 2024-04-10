# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional

from habitat.tasks.rearrange.multi_task.pddl_sim_state import PddlSimState
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType,
    PddlEntity,
    PddlSimInfo,
    do_entity_lists_match,
    ensure_entity_lists_match,
)


class Predicate:
    _arg_values: List[PddlEntity]

    def __init__(
        self,
        name: str,
        pddl_sim_state: Optional[PddlSimState],
        args: List[PddlEntity],
    ):
        """
        :param name: Predicate identifier. Does not need to be unique because
            predicates have the same name but different arguments.
        :param pddl_sim_state: Optionally specifies conditions that must be
            true in the simulator for the predicate to be true. If None is
            specified, no simulator state will force the Predicate to be true.
        """

        self._name = name
        self._pddl_sim_state = pddl_sim_state
        self._args = args
        self._arg_values = None

    def are_args_compatible(self, arg_values: List[PddlEntity]):
        """
        Checks if the list of argument values matches the types and counts of
        the argument list for this predicate.
        """

        return do_entity_lists_match(self._args, arg_values)

    def are_types_compatible(self, expr_types: Dict[str, ExprType]) -> bool:
        """
        Returns if the argument types match the underlying simulator state.
        """
        if self._pddl_sim_state is None:
            return True

        return self._pddl_sim_state.is_compatible(expr_types)

    def set_param_values(self, arg_values: List[PddlEntity]) -> None:
        arg_values = list(arg_values)
        if self._arg_values is not None:
            raise ValueError(
                f"Trying to set arg values with {arg_values} when current args are set to {self._arg_values}"
            )
        ensure_entity_lists_match(self._args, arg_values)
        self._arg_values = arg_values
        self._pddl_sim_state.sub_in(dict(zip(self._args, self._arg_values)))

    @property
    def n_args(self):
        return len(self._args)

    @property
    def name(self):
        return self._name

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]) -> "Predicate":
        self._arg_values = [
            sub_dict.get(entity, entity) for entity in self._arg_values
        ]
        ensure_entity_lists_match(self._args, self._arg_values)
        self._pddl_sim_state.sub_in(sub_dict)
        return self

    def sub_in_clone(self, sub_dict: Dict[PddlEntity, PddlEntity]):
        p = Predicate(
            self._name,
            self._pddl_sim_state.sub_in_clone(sub_dict),
            self._args,
        )
        if self._arg_values is not None:
            p.set_param_values(
                [sub_dict.get(entity, entity) for entity in self._arg_values]
            )
        return p

    def is_true(self, sim_info: PddlSimInfo) -> bool:
        """
        Returns if the predicate is satisfied in the current simulator state.
        Potentially returns the cached truth value of the predicate depending
        on `sim_info`.
        """
        self_repr = repr(self)
        if (
            sim_info.pred_truth_cache is not None
            and self_repr in sim_info.pred_truth_cache
        ):
            # Return the cached value.
            return sim_info.pred_truth_cache[self_repr]

        # Recompute and potentially cache the result.
        result = self._pddl_sim_state.is_true(sim_info)
        if sim_info.pred_truth_cache is not None:
            sim_info.pred_truth_cache[self_repr] = result
        return result

    def set_state(self, sim_info: PddlSimInfo) -> None:
        """
        Sets the simulator state to satisfy the predicate.
        """
        return self._pddl_sim_state.set_state(sim_info)

    def clone(self):
        p = Predicate(self._name, self._pddl_sim_state.clone(), self._args)
        if self._arg_values is not None:
            p.set_param_values(self._arg_values)
        return p

    def __str__(self):
        return f"<Predicate: {self._name} [{self._args}] [{self._arg_values}]>"

    def __repr__(self):
        return str(self)

    @property
    def compact_str(self):
        args = ",".join((x.name for x in self._arg_values))
        return f"{self._name}({args})"

    def __eq__(self, other_pred):
        return (
            self._name == other_pred._name
            and self._args == other_pred._args
            and self._arg_values == other_pred._arg_values
        )
