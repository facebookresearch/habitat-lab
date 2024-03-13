#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from enum import IntFlag
from math import log2
from typing import Any, Callable, Generator, List

class UserMask(IntFlag):
    """
    Represents a set of users.
    This is an IntFlag object - bitwise operators can be used.
    E.g.: user_1_and_3 = UserMask.from_index(1) | UserMask.from_index(3)
    """
    SERVER_ONLY = 0
    BROADCAST   = ~0

    @staticmethod
    def from_index(index: int) -> UserMask:
        """Create a UserMask from an index."""
        return UserMask(1 << index)
    
    @staticmethod
    def from_indices(indices: List[int]) -> UserMask:
        """Create a UserMask from a list of indices."""
        mask = UserMask(0)
        for index in indices:
            mask |= UserMask.from_index(index)
        return mask

    @staticmethod
    def all_except_index(index: int) -> UserMask:
        """Create a UserMask for all users except one index."""
        return ~(UserMask.from_index(index))
    
    @staticmethod
    def all_except_indices(indices: List[int]) -> UserMask:
        """Create a UserMask for all users except a list of indices."""
        return ~(UserMask.from_indices(indices))

class UserMaskIterator():
    """
    Allows for iterating upon UserMasks.
    """
    max_user_mask: UserMask
    max_user_count: int

    def __init__(self, max_user_count: int) -> None:
        assert(max_user_count >= 0)
        assert(max_user_count <= 32)
        self.max_user_count = max_user_count
        self.max_user_mask = 0
        for _ in range(max_user_count):
            self.max_user_mask = (self.max_user_mask << 1) + 1

    def foreach_user(self, user_mask: UserMask, function: Callable[[int], None]) -> None:
        """
        Execute a function for each user in a specified UserMask.
        """
        for user_index in self.user_indices(user_mask):
            function(user_index)

    def user_indices(self, user_mask: UserMask) -> Generator[float, Any, None]:
        """
        Generator that allows to iterate on UserMask.
        E.g.: for user_index in user_mask_iterator.user_indices(mask)
        """
        bitset = user_mask & self.max_user_mask
        while bitset != 0:
            user_bit = bitset & -bitset
            user_index = log2(user_bit)
            yield int(user_index)
            bitset ^= user_bit
