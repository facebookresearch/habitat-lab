#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat_hitl.core.user_mask import Mask, Users


def test_hitl_user_mask():
    # Test without any user.
    zero_users = Users(0)
    assert zero_users.max_user_count == 0
    assert len(zero_users.to_index_list(Mask.ALL)) == 0
    assert len(zero_users.to_index_list(Mask.NONE)) == 0
    user_indices = zero_users.to_index_list(
        Mask.from_index(0) | Mask.from_index(1)
    )
    assert 0 not in user_indices
    assert 1 not in user_indices
    user_indices = zero_users.to_index_list(Mask.all_except_index(0))
    assert 0 not in user_indices

    # Test without 4 users.
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

    # Test without 6 users.
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

    # Test without 2 users.
    two_users = Users(2)
    assert two_users.max_user_count == 2
    assert len(two_users.to_index_list(Mask.ALL)) == 2
    assert len(two_users.to_index_list(Mask.NONE)) == 0
    user_indices = two_users.to_index_list(Mask.from_indices([1, 2]))
    assert 0 not in user_indices
    assert 1 in user_indices
    assert 2 not in user_indices

    # Test without max users (32).
    max_users = Users(32)
    assert max_users.max_user_count == 32
    assert len(max_users.to_index_list(Mask.ALL)) == 32
    assert len(max_users.to_index_list(Mask.NONE)) == 0
    assert (
        len(max_users.to_index_list(Mask.all_except_indices([17, 22]))) == 30
    )
    assert len(max_users.to_index_list(Mask.from_indices([3, 15]))) == 2
