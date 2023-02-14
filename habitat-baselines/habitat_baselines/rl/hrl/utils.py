# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

from habitat.core.spaces import ActionSpace
from habitat_baselines.utils.common import get_num_actions


def find_action_range(
    action_space: ActionSpace, search_key: str
) -> Tuple[int, int]:
    """
    Returns the start and end indices of an action key in the action tensor. If
    the key is not found, a Value error will be thrown.
    """

    start_idx = 0
    found = False
    end_idx = get_num_actions(action_space[search_key])
    for k in action_space:
        if k == search_key:
            found = True
            break
        start_idx += get_num_actions(action_space[k])
    if not found:
        raise ValueError(f"Could not find stop action in {action_space}")
    return start_idx, end_idx


def find_action_range_pddl(pddl_space, action_name: str) -> Tuple[int, int]:
    index = 0
    for action in pddl_space:
        if action.name == action_name:
            return index, index + action.n_args
        else:
            index += action.n_args
