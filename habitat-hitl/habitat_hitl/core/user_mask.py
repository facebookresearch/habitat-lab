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
    """

    _max_user_mask: Mask
    _max_user_count: int

    def __init__(self, max_user_count: int) -> None:
        assert max_user_count > 0
        assert max_user_count <= Mask.MAX_VALUE
        self._max_user_count = max_user_count

        max_user_mask = 0
        for _ in range(max_user_count):
            max_user_mask = (max_user_mask << 1) + 1
        self._max_user_mask = Mask(max_user_mask)

    def indices(self, user_mask: Mask) -> Generator[int, Any, None]:
        """
        Generator that allows for iterating upon the specified Mask.
        - Example:\n
        .. code-block:: python
        for user_index in users.indices(Mask.all_except_indices([0,2])):
            ...
        """
        bitset = user_mask & self._max_user_mask
        while bitset != 0:
            user_bit = bitset & -bitset
            user_index = log2(user_bit)
            yield int(user_index)
            bitset ^= user_bit

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


if __name__ == "__main__":
    # TODO: Move this to a unit test when testing is added habitat-hitl.
    four_users = Users(4)
    assert four_users.max_user_count == 4
    assert len(four_users.to_index_list(Mask.ALL)) == 4
    assert len(four_users.to_index_list(Mask.NONE)) == 0
    user_indices = four_users.to_index_list(
        Mask.from_index(1) | Mask.from_index(2) | Mask.from_index(11)
    )
    assert 1 in user_indices
    assert 2 in user_indices
    assert 11 not in user_indices
    user_indices = four_users.to_index_list(Mask.all_except_index(1))
    assert 0 in user_indices
    assert 1 not in user_indices
    assert 2 in user_indices
    assert 3 in user_indices
    assert 4 not in user_indices

    six_users = Users(6)
    assert six_users.max_user_count == 6
    assert len(six_users.to_index_list(Mask.ALL)) == 6
    assert len(six_users.to_index_list(Mask.NONE)) == 0
    user_indices = six_users.to_index_list(Mask.all_except_indices([0, 2]))
    assert 0 not in user_indices
    assert 1 in user_indices
    assert 2 not in user_indices
    assert 3 in user_indices
    assert 4 in user_indices
    assert 5 in user_indices
    assert 6 not in user_indices

    two_users = Users(2)
    assert two_users.max_user_count == 2
    assert len(two_users.to_index_list(Mask.ALL)) == 2
    assert len(two_users.to_index_list(Mask.NONE)) == 0
    user_indices = two_users.to_index_list(Mask.from_indices([1, 2]))
    assert 0 not in user_indices
    assert 1 in user_indices
    assert 2 not in user_indices

    max_users = Users(32)
    assert max_users.max_user_count == 32
    assert len(max_users.to_index_list(Mask.ALL)) == 32
    assert len(max_users.to_index_list(Mask.NONE)) == 0
    assert (
        len(max_users.to_index_list(Mask.all_except_indices([17, 22]))) == 30
    )
    assert len(max_users.to_index_list(Mask.from_indices([3, 15]))) == 2
