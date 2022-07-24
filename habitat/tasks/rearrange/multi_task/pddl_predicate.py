from typing import Dict, List, Optional

from habitat.tasks.rearrange.multi_task.pddl_set_state import PddlSetState
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
        set_state: Optional[PddlSetState],
        args: List[PddlEntity],
    ):
        self._name = name
        self._set_state = set_state
        self._args = args
        self._arg_values = None

    def are_args_compatible(self, arg_values: List[PddlEntity]):
        return do_entity_lists_match(self._args, arg_values)

    def is_sim_compatible(self, expr_types):
        return self._set_state.is_compatible(expr_types)

    def set_param_values(self, arg_values: List[PddlEntity]) -> None:
        arg_values = list(arg_values)
        if self._arg_values is not None:
            raise ValueError(
                f"Trying to set arg values with {arg_values} when current args are set to {self._arg_values}"
            )
        ensure_entity_lists_match(self._args, arg_values)
        self._arg_values = arg_values
        self._set_state.sub_in(
            {k: v for k, v in zip(self._args, self._arg_values)}
        )

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
        self._set_state.sub_in(sub_dict)
        return self

    def is_true(self, sim_info: PddlSimInfo) -> bool:
        return self._set_state.is_true(sim_info)

    def set_state(self, sim_info: PddlSimInfo) -> None:
        return self._set_state.set_state(sim_info)

    def clone(self):
        p = Predicate(self._name, self._set_state.clone(), self._args)
        if self._arg_values is not None:
            p.set_param_values(self._arg_values)
        return p

    def __str__(self):
        return f"<Predicate: {self._name} [{self._args}] [{self._arg_values}]>"

    def __repr__(self):
        return str(self)

    @property
    def compact_str(self):
        args = ",".join([str(x) for x in self._arg_values])
        return f"{self._name}({args})"

    def __eq__(self, other_pred):
        return (
            self._name == other_pred._name
            and self._args == other_pred._args
            and self._arg_values == other_pred._arg_values
        )
