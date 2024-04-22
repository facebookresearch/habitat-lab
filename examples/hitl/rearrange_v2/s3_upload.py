#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os


try:
    import boto3
    boto3_imported = True
except ImportError:
    print("Unable to import 'boto3'.")
    boto3_imported = False

if boto3_imported:
    def upload_file_to_s3(local_file: str, file_name:str , s3_folder:str):
        try:
            import boto3
            # Check if local file exists
            if not os.path.isfile(local_file):
                raise ValueError(f"Local file {local_file} does not exist")

            s3_client = boto3.client('s3')
            if not s3_folder.endswith('/'):
                s3_folder += '/'

            s3_path = os.path.join(s3_folder, file_name)
            s3_path = s3_path.replace(os.path.sep, '/')

            if "S3_BUCKET" in os.environ:
                bucket_name = os.environ["S3_BUCKET"]
                print(f"Uploading {local_file} to {bucket_name}/{s3_path}")
                s3_client.upload_file(local_file, bucket_name, s3_path)
            else:
                print("'S3_BUCKET' environment variable is not set. Cannot upload.")
        except Exception as e:
            print(e)
else:
    def upload_file_to_s3(local_file: str, file_name:str , s3_folder:str):
        print("Unable to upload data to S3 because 'boto3' is not imported.")
