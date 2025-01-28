# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Any, Generator

from omegaconf import Container, OmegaConf
from omegaconf.base import Node


@contextmanager
def read_write(config: Any) -> Generator[Node, None, None]:
    r"""
    Temporarily authorizes the modification of a OmegaConf configuration
    within a context. Use the 'with' statement to enter the context.

    :param config: The configuration object that should get writing access
    """
    assert isinstance(config, Container)
    prev_state_readonly = config._get_node_flag("readonly")
    prev_state_struct = config._get_node_flag("struct")
    try:
        OmegaConf.set_struct(config, False)
        OmegaConf.set_readonly(config, False)
        yield config
    finally:
        OmegaConf.set_readonly(config, prev_state_readonly)
        OmegaConf.set_struct(config, prev_state_struct)
