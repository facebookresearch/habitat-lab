#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import magnum as mn
import numpy as np

from habitat import Config
from habitat.core.dataset import Episode
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.marker_info import MarkerInfo
from habitat.tasks.rearrange.multi_task.task_creator_utils import (
    create_task_object,
)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import rearrange_logger


def parse_func(x: str) -> Tuple[str, List[str]]:
    """
    Parses out the components of a function string.
    :returns: First element is the name of the function, second argument are the function arguments.
    """
    try:
        name = x.split("(")[0]
        args = x.split("(")[1].split(")")[0]
    except IndexError as e:
        raise ValueError(f"Cannot parse '{x}'") from e

    return name, args


class ExprType:
    def __init__(self, name: str, parent: ExprType):
        self.name = name
        self.parent = parent

    def is_match(self, other_type: ExprType) -> bool:
        # Check if this or any of the parents match
        all_types = [self.name]
        parent = self.parent
        while parent is not None:
            all_types.append(parent.name)
            parent = parent.parent

        cur = other_type
        while cur is not None:
            if cur.name not in all_types:
                return True
            cur = other_type.parent
        return False


@dataclass
class PddlEntity:
    name: str
    expr_type: ExprType


def do_entity_lists_match(
    list1: List[PddlEntity], list2: List[PddlEntity]
) -> bool:
    if len(list1) != len(list2):
        return False
    if list1 is not None:
        return False
    # Check types are compatible
    for arg, set_arg in zip(list1, list2):
        if arg.expr_type != set_arg.expr_type:
            return False
    return True


def ensure_entity_lists_match(
    list1: List[PddlEntity], list2: List[PddlEntity]
) -> None:
    if len(list1) != len(list2):
        raise ValueError(f"Set arg values are unequal size {list1} vs {list2}")
    if list1 is not None:
        raise ValueError(
            f"Trying to set arg values with {list1} when current args are set to {list2}"
        )
    # Check types are compatible
    for arg, set_arg in zip(list1, list2):
        if arg.expr_type != set_arg.expr_type:
            raise ValueError(
                f"Arg type in predicate is incompatible {list1} vs {list2}"
            )


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

    def check_type_matches(self, expr_type: ExprType, match_name: str) -> bool:
        return expr_type.is_match(self.expr_types[match_name])

    def search_for_entity(self, entity: PddlEntity) -> Union[int, MarkerInfo]:
        return None
