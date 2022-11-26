#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np

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
        args_list = args.split(",")
        args_list = [x.strip() for x in args_list]
    except IndexError as e:
        raise ValueError(f"Cannot parse '{x}'") from e

    if len(args_list) == 1 and args_list[0] == "":
        args_list = []

    return name, args_list


class ExprType:
    def __init__(self, name: str, parent: "ExprType"):
        self.name = name
        self.parent = parent

    def is_subtype_of(self, other_type: "ExprType") -> bool:
        """
        If true, then `self` is compatible with `other_type` but `other_type`
        is NOT necessarily compatible with `self`. In other words, `other_type`
        is higher on the hierarchy of sub-types than `self`.
        """
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

    def __eq__(self, other):
        if not isinstance(other, PddlEntity):
            return False
        return (self.name == other.name) and (
            self.expr_type.name == other.expr_type.name
        )


def do_entity_lists_match(
    to_set: List[PddlEntity], set_value: List[PddlEntity]
) -> bool:
    """
    Returns if the two predicate lists match in count and argument types.
    """

    if len(to_set) != len(set_value):
        return False
    # Check types are compatible
    return all(
        set_arg.expr_type.is_subtype_of(arg.expr_type)
        for arg, set_arg in zip(to_set, set_value)
    )


def ensure_entity_lists_match(
    to_set: List[PddlEntity], set_value: List[PddlEntity]
) -> None:
    """
    Checks if the two predicate lists match in count and argument types. If
    they don't match, an exception is thrown.
    """

    if len(to_set) != len(set_value):
        raise ValueError(
            f"Set arg values are unequal size {to_set} vs {set_value}"
        )
    # Check types are compatible
    for arg, set_arg in zip(to_set, set_value):
        if not set_arg.expr_type.is_subtype_of(arg.expr_type):
            raise ValueError(
                f"Arg type is incompatible \n{to_set}\n vs \n{set_value}"
            )


# Hardcoded pddl types needed for setting simulator states.
robot_type = "robot_type"
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
    marker_handles: Dict[str, MarkerInfo]
    robot_ids: Dict[str, int]

    sim: RearrangeSim
    dataset: RearrangeDatasetV0
    env: RearrangeTask
    episode: Episode
    obj_thresh: float
    art_thresh: float
    robot_at_thresh: float
    expr_types: Dict[str, ExprType]
    predicates: Dict[str, Any]
    all_entities: Dict[str, Any]

    def get_predicate(self, pred_name: str):
        return self.predicates[pred_name]

    def check_type_matches(self, entity: PddlEntity, match_name: str) -> bool:
        return entity.expr_type.is_subtype_of(self.expr_types[match_name])

    def get_entity_pos(self, entity: PddlEntity) -> np.ndarray:
        ename = entity.name
        if self.check_type_matches(entity, robot_type):
            robot_id = self.robot_ids[ename]
            return self.sim.get_robot_data(robot_id).robot.base_pos
        elif self.check_type_matches(entity, ART_OBJ_TYPE):
            marker_info = self.marker_handles[ename]
            return marker_info.get_current_position()
        elif self.check_type_matches(entity, GOAL_TYPE):
            idx = self.target_ids[ename]
            targ_idxs, pos_targs = self.sim.get_targets()
            rel_idx = targ_idxs.tolist().index(idx)
            return pos_targs[rel_idx]
        elif self.check_type_matches(entity, RIGID_OBJ_TYPE):
            rom = self.sim.get_rigid_object_manager()
            idx = self.obj_ids[ename]
            abs_obj_id = self.sim.scene_obj_ids[idx]
            cur_pos = rom.get_object_by_id(
                abs_obj_id
            ).transformation.translation
            return cur_pos
        else:
            raise ValueError()

    def search_for_entity_any(self, entity: PddlEntity):
        ename = entity.name
        if self.check_type_matches(entity, robot_type):
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

        if expected_type == robot_type:
            return self.robot_ids[ename]
        elif expected_type == ART_OBJ_TYPE:
            return self.marker_handles[ename]
        elif expected_type == GOAL_TYPE:
            return self.target_ids[ename]
        elif expected_type == RIGID_OBJ_TYPE:
            return self.obj_ids[ename]
        else:
            raise ValueError()
