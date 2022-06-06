#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from habitat.core.dataset import Episode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask


def parse_func(x: str) -> Tuple[str, List[str]]:
    """
    Parses out the components of a function string.
    :returns: First element is the name of the function, second argument are the function arguments.
    """
    try:
        name = x.split("(")[0]
        args = x.split("(")[1].split(")")[0]
        args = args.split(",")
        args = [x.strip() for x in args]
    except IndexError as e:
        raise ValueError(f"Cannot parse '{x}'") from e

    if len(args) == 1 and args[0] == "":
        args = []

    return name, args


class ExprType:
    def __init__(self, name: str, parent: "ExprType"):
        self.name = name
        self.parent = parent

    def is_subtype_of(self, other_type: "ExprType") -> bool:
        # Check if this or any of the parents match
        all_types = [self.name]
        parent = self.parent
        while parent is not None:
            all_types.append(parent.name)
            parent = parent.parent

        return other_type.name in all_types

    def __repr__(self):
        return f"T:{self.name}"


@dataclass(frozen=True)
class PddlEntity:
    name: str
    expr_type: ExprType

    def __repr__(self):
        return f"{self.name}-{self.expr_type}"


def do_entity_lists_match(
    to_set: List[PddlEntity], set_value: List[PddlEntity]
) -> bool:
    if len(to_set) != len(set_value):
        return False
    if to_set is not None:
        return False
    # Check types are compatible
    return all(
        set_arg.expr_type.is_subtype_of(arg.expr_type)
        for arg, set_arg in zip(to_set, set_value)
    )


def ensure_entity_lists_match(
    to_set: List[PddlEntity], set_value: List[PddlEntity]
) -> None:
    """ """
    if len(to_set) != len(set_value):
        raise ValueError(
            f"Set arg values are unequal size {to_set} vs {set_value}"
        )
    # Check types are compatible
    for arg, set_arg in zip(to_set, set_value):
        if not set_arg.expr_type.is_subtype_of(arg.expr_type):
            # breakpoint()
            # set_arg.expr_type.is_subtype_of(arg.expr_type)
            raise ValueError(
                f"Arg type is incompatible \n{to_set}\n vs \n{set_value}"
            )


# Hardcoded pddl types needed for setting simulator states.
ROBOT_TYPE = "robot_type"
STATIC_OBJ_TYPE = "static_obj_type"
ART_OBJ_TYPE = "art_obj_type"
OBJ_TYPE = "obj_type"
CAB_TYPE = "cab_type"
FRIDGE_TYPE = "fridge_type"
GOAL_TYPE = "goal_type"
RIGID_OBJ_TYPE = "rigid_obj_type"


@dataclass
class PddlSimInfo:
    obj_ids: Dict[str, int]
    target_ids: Dict[str, int]
    art_handles: Dict[str, int]
    marker_handles: List[str]
    robot_ids: Dict[str, int]

    sim: RearrangeSim
    dataset: RearrangeDatasetV0
    env: RearrangeTask
    episode: Episode
    obj_thresh: float
    art_thresh: float
    expr_types: Dict[str, ExprType]

    def check_type_matches(self, entity: PddlEntity, match_name: str) -> bool:
        return entity.expr_type.is_subtype_of(self.expr_types[match_name])

    def search_for_entity_any(self, entity: PddlEntity):
        ename = entity.name
        if self.check_type_matches(entity, ROBOT_TYPE):
            return self.robot_ids[ename]
        elif self.check_type_matches(entity, ART_OBJ_TYPE):
            return self.marker_handles[ename]
        elif self.check_type_matches(entity, GOAL_TYPE):
            return self.target_ids[ename]
        elif self.check_type_matches(entity, RIGID_OBJ_TYPE):
            return self.obj_ids[ename]
        else:
            raise ValueError()

    def search_for_entity(
        self, entity: PddlEntity, expected_type: str
    ) -> Union[int, str, MarkerInfo]:
        if not self.check_type_matches(entity, expected_type):
            raise ValueError(
                f"Type mismatch {entity} but expected {expected_type}"
            )

        ename = entity.name

        if expected_type == ROBOT_TYPE:
            return self.robot_ids[ename]
        elif expected_type == ART_OBJ_TYPE:
            return self.marker_handles[ename]
        elif expected_type == GOAL_TYPE:
            return self.target_ids[ename]
        elif expected_type == RIGID_OBJ_TYPE:
            return self.obj_ids[ename]
        else:
            raise ValueError()
