#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import registry
from habitat.core.dataset import Dataset


def _try_register_pointnavdatasetv1():
    try:
        from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

        has_pointnav = True
    except ImportError as e:
        has_pointnav = False
        pointnav_import_error = e

    if has_pointnav:
        from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
    else:
        @registry.register_dataset(name="MP3DEQA-v1")
        class PointnavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pointnav_import_error
