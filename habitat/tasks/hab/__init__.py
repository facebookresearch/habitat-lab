#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_rearrange_task():
    try:
    ort habitat.tasks.hab.envs.rearrang_env  # noqa: F401
        import habitat.tasks.hab.envs.rearrang_pick_env
        import habitat.tasks.hab.rearrange_sesors
    #import habitat.tasks.hab.envs.hab_simulator
    except ImportError as e:
        print(e)
        rearrangetask_import_error = e
    #
    #     @registry.register_task(name="Rearrange-v0")
    #     class RearrangeTaskImportError(EmbodiedTask):
    #         def __init__(self, *args, **kwargs):
    #             raise rearrangetask_import_error
