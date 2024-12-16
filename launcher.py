#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import struct
import subprocess
from typing import Any, Optional, Union
import boto3
import tarfile
from pathlib import Path
import psutil
from tqdm import tqdm
from boto3.s3.transfer import S3Transfer, TransferConfig



def get_uncompressed_file_size(file_name: str):
    with open(file_name, "rb") as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]

def extract(file_name: str, dest_dir: str):
    file_size = get_uncompressed_file_size(file_name)
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Extracting")
    def track_progress(members):
        for member in members:
            yield member
            progress_bar.update(member.size)

    with tarfile.open(file_name, "r") as tarball:
        tarball.extractall(path=dest_dir, members=track_progress(tarball))

    os.remove(file_name)

def download_fast(bucket_name: str, s3_key: str, file_name: str):
    # Check if local file exists
    if os.path.exists(file_name):
        raise ValueError(f"Local file {file_name} exists.")
    
    s3_client = boto3.client("s3")
    cpu_count = psutil.cpu_count()
    transfer_config = TransferConfig(max_concurrency=cpu_count)

    response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
    file_size: int = response["ContentLength"]
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading")
    def progress_callback(bytes_transferred):
        progress_bar.update(bytes_transferred)

    transfer = S3Transfer(client=s3_client, config=transfer_config)
    transfer.download_file(bucket_name, s3_key, file_name, callback=progress_callback)
    progress_bar.close()

def get_bucket_names_with_tag(tag_key: str, tag_value: str) -> list[str]:
    s3 = boto3.client("s3")
    response = s3.list_buckets()
    buckets = response["Buckets"]
    matching_bucket_names: list[str] = []
    
    for bucket in buckets:
        bucket_name = bucket["Name"]
        try:
            tagging = s3.get_bucket_tagging(Bucket=bucket_name)
            tags = tagging["TagSet"]
            for tag in tags:
                if tag["Key"] == tag_key and tag["Value"] == tag_value:
                    matching_bucket_names.append(bucket_name)
                    break
        except s3.exceptions.ClientError as e:
            # If the bucket has no tags, a ClientError will be raised
            if e.response["Error"]["Code"] == "NoSuchTagSet":
                continue
            else:
                raise e
    
    return matching_bucket_names

def find_server_data_bucket_name() -> str:
    matching_bucket_names = get_bucket_names_with_tag(tag_key="Name", tag_value="server-data")
    assert len(matching_bucket_names) > 0, "This cloud does not have a 'server-data' bucket."
    if len(matching_bucket_names) > 1:
        print("Warning: This cloud has multiple 'server-data' buckets.")
    return matching_bucket_names[0]

def is_directory_empty(directory_path: Union[str, Path]):
    return not os.listdir(directory_path)

def main(data_archive_name: Optional[str], launch_command: str):
    if data_archive_name is not None:
        data_folder = Path("data")
        archive_file = Path.joinpath(data_folder, "data.tar.gz")
        s3_data_folder_directory = "hitl_data"
        s3_key = os.path.join(s3_data_folder_directory, data_archive_name)

        os.makedirs(data_folder, exist_ok=True)

        if is_directory_empty(data_folder):
            if not archive_file.exists():
                bucket_name = find_server_data_bucket_name()
                download_fast(bucket_name, s3_key, archive_file)
        
        if archive_file.exists():
            extract(archive_file, data_folder)

        # Sanity check
        data_folder = Path("data")
        assert data_folder.exists()
        assert data_folder.is_dir()
        assert not is_directory_empty(data_folder)
        assert not archive_file.exists()

    # Run launch command
    print(f"Launching command: `{launch_command}`")
    process = subprocess.Popen(launch_command, shell=True)
    process.wait()
    print(f"Launch command terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-archive-name",
        type=str,
        required=False,
        help="[REQUIES boto3] The data folder archive name to download and extract as './data'. E.g. experiment_001.tar.gz. If omitted, no data folder will be downloaded.",
    )
    parser.add_argument(
        "--launch-command",
        type=str,
        required=True,
        help="The launch command.",
    )
    args = parser.parse_args()
    main(data_archive_name=args.data_archive_name, launch_command=args.launch_command)