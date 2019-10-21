#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_vln_task():
    try:
        from habitat.tasks.vln.vln import VLNTask

        has_vlntask = True
    except ImportError as e:
        has_vlntask = False
        vlntask_import_error = e

    if has_vlntask:
        from habitat.tasks.vln.vln import VLNTask
    else:

        @registry.register_task(name="VLN-v0")
        class VLNTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise vlntask_import_error
