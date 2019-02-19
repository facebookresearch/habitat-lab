#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.datasets.registration import (
    dataset_registry,
    register_dataset,
    make_dataset,
)

register_dataset(
    id_dataset="MP3DEQA-v1",
    entry_point="habitat.datasets.eqa.mp3d_eqa_dataset:Matterport3dDatasetV1",
)

register_dataset(
    id_dataset="PointNav-v1",
    entry_point="habitat.datasets.pointnav.pointnav_dataset:PointNavDatasetV1",
)

__all__ = ["dataset_registry", "register_dataset", "make_dataset"]
