#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_mp3d_eqa_dataset():
    try:
        from habitat.datasets.eqa.mp3d_eqa_dataset import (  # noqa: F401 isort:skip
            Matterport3dDatasetV1,
        )
    except ImportError as e:
        mp3deqa_import_error = e

        @registry.register_dataset(name="MP3DEQA-v1")
        class Matterport3dDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise mp3deqa_import_error
