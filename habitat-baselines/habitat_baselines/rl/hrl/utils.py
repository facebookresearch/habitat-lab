# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

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


class skill_io_manager:
    def __init__(self):
        self._prev_action = None
        self._hidden_state = None

    @property
    def prev_action(self):
        return self._prev_action

    @prev_action.setter
    def prev_action(self, value):
        self._prev_action = value

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value):
        self._hidden_state = value

    def init_hidden_state(self, obs, hs, nr):
        for k in obs:
            bs = obs[k].shape[0]
            break
        self._hidden_state = torch.zeros((bs, nr, hs)).to(obs[k].device)

    def init_prev_action(self, prev_actions, na):
        bs = prev_actions.shape[0]
        self._prev_action = torch.zeros((bs, na)).to(prev_actions.device)
