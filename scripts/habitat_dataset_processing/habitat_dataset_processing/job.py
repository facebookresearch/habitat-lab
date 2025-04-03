#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from habitat_dataset_processing.configs import Operation
from habitat_dataset_processing.util import resolve_relative_path


class Job:
    def __init__(
        self,
        asset_path: str,
        dest_path: str,
        operation: Operation,
        simplify: bool,
    ):
        """
        Defines a single asset processing job (input -> output).
        """
        self.source_path = resolve_relative_path(asset_path)
        self.dest_path = resolve_relative_path(dest_path)
        self.operation = operation
        self.simplify = simplify
