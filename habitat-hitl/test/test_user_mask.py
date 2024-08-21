#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat_hitl.core.user_mask import Mask, Users


def test_hitl_user_mask_0_user():
    zero_users = Users(0)
    zero_users.activate_user(1)
    assert zero_users.max_user_count == 0
    assert zero_users.active_user_count == 0
    assert len(zero_users.to_index_list(Mask.ALL)) == 0
    assert len(zero_users.to_index_list(Mask.NONE)) == 0
    user_indices = zero_users.to_index_list(
        Mask.from_index(0) | Mask.from_index(1)
    )
    assert 0 not in user_indices
    assert 1 not in user_indices
    user_indices = zero_users.to_index_list(Mask.all_except_index(0))
    assert 0 not in user_indices
    zero_users.deactivate_user(1)
    assert zero_users.active_user_count == 0


def test_hitl_user_mask_4_users():
    four_users = Users(4)
    assert four_users.max_user_count == 4
    assert four_users.active_user_count == 0
    for user_index in range(4):
        four_users.activate_user(user_index)
        assert four_users.active_user_count == user_index + 1
    assert four_users.active_user_count == 4
    four_users.activate_user(5)
    assert four_users.active_user_count == 4
    four_users.deactivate_user(5)
    assert four_users.active_user_count == 4
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
    for user_index in range(4):
        four_users.deactivate_user(user_index)
        assert (
            four_users.active_user_count
            == four_users.max_user_count - user_index - 1
        )
    assert four_users.active_user_count == 0


def test_hitl_user_mask_32_users():
    max_users = Users(32)
    assert max_users.max_user_count == 32
    assert max_users.active_user_count == 0
    for user_index in range(32):
        max_users.activate_user(user_index)
        assert max_users.active_user_count == user_index + 1
    assert len(max_users.to_index_list(Mask.ALL)) == 32
    assert len(max_users.to_index_list(Mask.NONE)) == 0
    assert (
        len(max_users.to_index_list(Mask.all_except_indices([17, 22]))) == 30
    )
    assert len(max_users.to_index_list(Mask.from_indices([3, 15]))) == 2
    for user_index in range(32):
        max_users.deactivate_user(user_index)
        assert (
            max_users.active_user_count
            == max_users.max_user_count - user_index - 1
        )
    assert max_users.active_user_count == 0


def test_hitl_user_mask_activate_users():
    four_users = Users(4, activate_users=True)
    assert four_users.max_user_count == 4
    assert four_users.active_user_count == 4
    assert len(four_users.to_index_list(Mask.ALL)) == 4
    assert len(four_users.to_index_list(Mask.NONE)) == 0
    four_users.deactivate_user(3)
    assert four_users.max_user_count == 4
    assert four_users.active_user_count == 3
    assert len(four_users.to_index_list(Mask.ALL)) == 3
    assert len(four_users.to_index_list(Mask.NONE)) == 0
