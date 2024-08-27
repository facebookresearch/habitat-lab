#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

from examples.hitl.rearrange_v2.s3_upload import (
    generate_unique_session_id,
    make_s3_filename,
    validate_experiment_name,
)
from examples.hitl.rearrange_v2.util import timestamp
from habitat_hitl.core.types import ConnectionRecord


def test_generate_unique_session_id():
    episode_ids: List[int] = []
    connection_records: Dict[int, ConnectionRecord] = {}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_no-user_{timestamp()}"
    episode_ids = [2]
    connection_records: Dict[int, ConnectionRecord] = {}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"2_no-user_{timestamp()}"
    episode_ids = [2, 3, 4, 5]
    connection_records: Dict[int, ConnectionRecord] = {}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"2-3-4-5_no-user_{timestamp()}"
    episode_ids: List[int] = []
    connection_records = {0: {"user_id": "test"}}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_test_{timestamp()}"
    episode_ids: List[int] = []
    connection_records = {2: {"user_id": "test"}}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_test_{timestamp()}"
    episode_ids: List[int] = []
    connection_records = {
        0: {"user_id": "a"},
        1: {"user_id": "b"},
        2: {"user_id": "c"},
        3: {"user_id": "d"},
    }
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_a-b-c-d_{timestamp()}"
    episode_ids: List[int] = []
    connection_records = {
        0: {"uid": "test"},
        1: {"uid": "test"},
    }
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_invalid-user-invalid-user_{timestamp()}"


def test_make_s3_filename():
    s3_filename = make_s3_filename("id", "te-st.txt")
    assert s3_filename == "id_te-st.txt"
    s3_filename = make_s3_filename("id", "te???st.txt")
    assert s3_filename == "id_te!!!st.txt"
    s3_filename = make_s3_filename("", "")
    assert s3_filename == "_"
    s3_filename = make_s3_filename("ab", "cd\nef\0gh\3.txt")
    assert s3_filename == "ab_cd!ef!gh!.txt"

    long_name = "0" * 500
    long_name += ".txt"
    s3_filename = make_s3_filename("ab", long_name)
    assert len(s3_filename) == 128
    assert s3_filename[-4:] == ".txt"


def test_validate_experiment_name():
    assert validate_experiment_name(None) == False
    assert validate_experiment_name("test") == True
    assert validate_experiment_name("test_test-test.123") == True
    assert validate_experiment_name("test?") == False
    assert (
        validate_experiment_name(
            "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
        )
        == True
    )
    assert (
        validate_experiment_name(
            "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
        )
        == False
    )
