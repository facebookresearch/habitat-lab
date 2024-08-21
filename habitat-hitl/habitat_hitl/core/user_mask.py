#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import IntFlag
from math import log2
from typing import Any, Final, Generator, List


class Mask(IntFlag):
    """
    Represents a mask.
    This is an IntFlag object - bitwise operators can be used.
    - Example:\n
    .. code-block:: python
    user_1_and_3 = Mask.from_index(1) | Mask.from_index(3)
    """

    NONE: Final[int] = 0
    ALL: Final[int] = ~0
    MAX_VALUE: Final[int] = 32

    @staticmethod
    def from_index(index: int) -> Mask:
        """Create a Mask from an index."""
        return Mask(1 << index)

    @staticmethod
    def from_indices(indices: List[int]) -> Mask:
        """Create a Mask from a list of indices."""
        mask = Mask(0)
        for index in indices:
            mask |= Mask.from_index(index)
        return mask

    @staticmethod
    def all_except_index(index: int) -> Mask:
        """Create a Mask for all indices except one."""
        return Mask(~Mask.from_index(index))

    @staticmethod
    def all_except_indices(indices: List[int]) -> Mask:
        """Create a Mask for all indices except a list of indices."""
        return Mask(~Mask.from_indices(indices))


class Users:
    """
    Represents a set of users with a max_user_count upper bound.
    By default, all users are "inactive". Add new users using the 'add_user' method.
    """

    _max_user_count: int
    _active_user_mask: Mask

    def __init__(
        self, max_user_count: int, activate_users: bool = False
    ) -> None:
        assert max_user_count >= 0
        assert max_user_count <= Mask.MAX_VALUE
        self._max_user_count = max_user_count

        if activate_users:
            user_mask = 0
            for _ in range(max_user_count):
                user_mask = (user_mask << 1) + 1
            self._active_user_mask = Mask(user_mask)
        else:
            self._active_user_mask = Mask.NONE

    def indices(self, user_mask: Mask) -> Generator[int, Any, None]:
        """
        Generator that allows for iterating upon the specified Mask.
        - Example:\n
        .. code-block:: python
        for user_index in users.indices(Mask.all_except_indices([0,2])):
            ...
        """
        bitset = user_mask & self._active_user_mask
        while bitset != 0:
            user_bit = bitset & -bitset
            user_index = log2(user_bit)
            yield int(user_index)
            bitset ^= user_bit

    def activate_user(self, user_index: int) -> None:
        """Activate an user."""
        if user_index < self._max_user_count:
            self._active_user_mask |= Mask.from_index(user_index)

    def deactivate_user(self, user_index: int) -> None:
        """Deactivate an user."""
        self._active_user_mask &= ~Mask.from_index(user_index)

    def to_index_list(self, user_mask: Mask) -> List[int]:
        """Returns a list of user indices from the specified Mask."""
        output: List[int] = []
        for user_index in self.indices(user_mask):
            output.append(user_index)
        return output

    @property
    def max_user_count(self) -> int:
        """Returns the size of the user set."""
        return self._max_user_count

    @property
    def active_user_count(self) -> int:
        """Returns the number of active users."""
        return len(list(self.indices(Mask.ALL)))
