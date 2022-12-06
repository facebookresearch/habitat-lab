# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from omegaconf import OmegaConf
from omegaconf.base import Node

if TYPE_CHECKING:
    from omegaconf import Container


# TODO : Delete this when migration to hydra is complete
@contextmanager
def read_write(config: "Container") -> Generator[Node, None, None]:
    prev_state_readonly = config._get_node_flag("readonly")
    prev_state_struct = config._get_node_flag("struct")
    try:
        OmegaConf.set_struct(config, False)
        OmegaConf.set_readonly(config, False)
        yield config
    finally:
        OmegaConf.set_readonly(config, prev_state_readonly)
        OmegaConf.set_struct(config, prev_state_struct)
