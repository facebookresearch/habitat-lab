#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Dict, List, Optional

from habitat_hitl.core.types import ConnectionRecord
from util import timestamp

try:
    import boto3

    boto3_imported = True
except ImportError:
    print("Unable to import 'boto3'.")
    boto3_imported = False

if boto3_imported:

    def upload_file_to_s3(local_file: str, file_name: str, s3_folder: str):
        try:
            # Check if local file exists
            if not os.path.isfile(local_file):
                raise ValueError(f"Local file {local_file} does not exist")

            s3_client = boto3.client("s3")
            if not s3_folder.endswith("/"):
                s3_folder += "/"

            s3_path = os.path.join(s3_folder, file_name)
            s3_path = s3_path.replace(os.path.sep, "/")

            if "S3_BUCKET" in os.environ:
                bucket_name = os.environ["S3_BUCKET"]
                print(f"Uploading {local_file} to {bucket_name}/{s3_path}")
                s3_client.upload_file(local_file, bucket_name, s3_path)
            else:
                print(
                    "'S3_BUCKET' environment variable is not set. Cannot upload."
                )
        except Exception as e:
            print(e)

else:

    def upload_file_to_s3(local_file: str, file_name: str, s3_folder: str):
        print("Unable to upload data to S3 because 'boto3' is not imported.")


def generate_unique_session_id(
        episode_ids: List[str],
        connection_records: Dict[int, ConnectionRecord]
    ) -> str:
    """Generate a unique session name."""
    # Generate episodes string
    episodes_str = "no-episode"
    if len(episode_ids) == 1:
        episodes_str = episode_ids[0]
    elif len(episode_ids) > 1:
        episodes_str = f"{episode_ids[0]}-{episode_ids[-1]}"

    # Generate users string
    users_str = ""
    if len(connection_records) == 0:
        users_str = "no-user"
    for _, connection_record in connection_records.items():
        if users_str != "":
            users_str += "-"
        if "user_id" in connection_record:
            users_str += str(connection_record["user_id"])
        else:
            users_str += "invalid-user"

    return f"{episodes_str}_{users_str}_{timestamp()}"

def make_s3_filename(session_id: str, orig_file_name: str) -> str:
    """Transformation of a path for S3. Removes invalid characters and checks length."""
    filename = f"{session_id}_{orig_file_name}"

    # Limit the filename size. Use last characters to preserve extension.
    if len(filename) > 128:
        filename = filename[-128:]

    # Replace unauthorized characters by '!'
    s3_filename = ""
    authorized_chars = ["_", "-", "."]
    for c in filename:
        if c.isalnum() or c in authorized_chars:
            s3_filename += c
        else:
            s3_filename += "!"

    return s3_filename

if __name__ == "__main__":
    # TODO: Temp test.
    episode_ids = []
    connection_records = {}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_no-user_{timestamp()}"
    episode_ids = [2]
    connection_records = {}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"2_no-user_{timestamp()}"
    episode_ids = [2, 3, 4, 5]
    connection_records = {}
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"2-5_no-user_{timestamp()}"
    episode_ids = []
    connection_records = { 0: {"user_id": "test"} }
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_test_{timestamp()}"
    episode_ids = []
    connection_records = { 2: {"user_id": "test"} }
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_test_{timestamp()}"
    episode_ids = []
    connection_records = {
        0: {"user_id": "a"},
        1: {"user_id": "b"},
        2: {"user_id": "c"},
        3: {"user_id": "d"},
    }
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_a-b-c-d_{timestamp()}"
    episode_ids = []
    connection_records = {
        0: {"uid": "test"},
        1: {"uid": "test"},
    }
    session_id = generate_unique_session_id(episode_ids, connection_records)
    assert session_id == f"no-episode_invalid-user-invalid-user_{timestamp()}"


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

