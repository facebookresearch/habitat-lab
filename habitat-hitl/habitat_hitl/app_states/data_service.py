#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import boto3


class DataService:
    def __init__(
        self,
        *,
        hitl_config,
    ):
        self._hitl_config = hitl_config

        self.upload_to_s3 = self._hitl_config.data_collection.upload_to_s3
        self.bucket_name = self._hitl_config.data_collection.s3_bucket
        self.client = self.get_s3_client()

    def get_s3_client(self):
        s3_client = boto3.client("s3")
        return s3_client

    def upload(self, path, object_name):
        if not self.upload_to_s3:
            return None
        response = self.client.upload_file(path, self.bucket_name, object_name)
        return response
