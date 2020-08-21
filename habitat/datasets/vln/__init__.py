#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_r2r_vln_dataset():
    try:
        from habitat.datasets.vln.r2r_vln_dataset import (  # noqa: F401 isort:skip
            VLNDatasetV1,
        )
    except ImportError as e:
        r2r_vln_import_error = e

        @registry.register_dataset(name="R2RVLN-v1")
        class R2RDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise r2r_vln_import_error
