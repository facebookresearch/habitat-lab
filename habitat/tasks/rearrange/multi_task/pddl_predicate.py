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

    def set_param_values(self, arg_values: List[PddlEntity]):
        ensure_entity_lists_match(self._args, arg_values)
        self._arg_values = arg_values

    @property
    def n_args(self):
        return len(self._args)

    @property
    def name(self):
        return self._name

    def sub_in(self, sub_dict: Dict[PddlEntity, PddlEntity]):
        self._arg_values = [
            sub_dict.get(entity, entity) for entity in self._arg_values
        ]
        ensure_entity_lists_match(self._args, self._arg_values)
        return self

    def is_true(self, sim_info: PddlSimInfo) -> bool:
        return self._set_state.is_true(sim_info)

    def set_state(self, sim_info: PddlSimInfo) -> None:
        return self._set_state.set_state(sim_info)

    def clone(self):
        return Predicate(self._name, self.set_state.clone(), self.args)

    def __str__(self):
        return f"<Predicate: {self._name} [{self._args}] [{self._arg_values}]>"

    def __repr__(self):
        return str(self)
