# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict, List, Optional

from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
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
        is_valid_fn: Optional[Callable],
        set_state_fn: Optional[Callable],
        args: List[PddlEntity],
    ):
        """
        :param name: Predicate identifier. Does not need to be unique because
            predicates have the same name but different arguments.
        :param is_valid_fn: Function that returns if the predicate is true in
            the current state. This function must return a bool and
            take as input the predicate parameters specified by `args`. If
            None, then this always returns True.
        :param set_state_fn: Function that sets the state to satisfy the
            predicate. This function must return nothing and take as input the
            values set in the predicate parameters specified by `args`. If
            None, then no simulator state is set.
        :param args: The names of the arguments to the predicate. Note that
            these are only placeholders. Actual entities are substituted in later
            via `self.set_param_values`.
        """

        self._name = name
        self._args = args
        self._arg_values = None
        self._is_valid_fn = is_valid_fn
        self._set_state_fn = set_state_fn

    def are_args_compatible(self, arg_values: List[PddlEntity]):
        """
        Checks if the list of argument values matches the types and counts of
        the argument list for this predicate.
        """

        return do_entity_lists_match(self._args, arg_values)

    def set_param_values(self, arg_values: List[PddlEntity]) -> None:
        arg_values = list(arg_values)
        if self._arg_values is not None:
            raise ValueError(
                f"Trying to set arg values with {arg_values} when current args are set to {self._arg_values}"
            )
        ensure_entity_lists_match(self._args, arg_values)
        self._arg_values = arg_values

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
        return self

    def sub_in_clone(self, sub_dict: Dict[PddlEntity, PddlEntity]):
        p = Predicate(
            self._name,
            self._is_valid_fn,
            self._set_state_fn,
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
        if self._is_valid_fn is None:
            result = True
        else:
            result = self._is_valid_fn(
                sim_info=sim_info, **self._create_kwargs()
            )
        if sim_info.pred_truth_cache is not None:
            sim_info.pred_truth_cache[self_repr] = result
        return result

    def set_state(self, sim_info: PddlSimInfo) -> None:
        """
        Sets the simulator state to satisfy the predicate.
        """
        if self._set_state_fn is not None:
            self._set_state_fn(sim_info=sim_info, **self._create_kwargs())

    def _create_kwargs(self):
        return {
            arg.name: val for arg, val in zip(self._args, self._arg_values)
        }

    def clone(self):
        p = Predicate(
            self._name, self._is_valid_fn, self._set_state_fn, self._args
        )
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
