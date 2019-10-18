#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_mp3d_eqa_dataset():
    try:
        from habitat.datasets.eqa.mp3d_eqa_dataset import Matterport3dDatasetV1

        has_mp3deqa = True
    except ImportError as e:
        has_mp3deqa = False
        mp3deqa_import_error = e

    if has_mp3deqa:
        from habitat.datasets.eqa.mp3d_eqa_dataset import Matterport3dDatasetV1
    else:

        @registry.register_dataset(name="MP3DEQA-v1")
        class Matterport3dDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise mp3deqa_import_error
