from contextlib import contextmanager
from typing import Generator, Union

from omegaconf import OmegaConf
from omegaconf.base import Node

from habitat.config.default import Config


# TODO : Delete this when migration to hydra is complete
@contextmanager
def read_write(config: Union[Node, Config]) -> Generator[Node, None, None]:
    if isinstance(config, Node):
        prev_state_readonly = config._get_node_flag("readonly")
        prev_state_struct = config._get_node_flag("struct")
        try:
            OmegaConf.set_struct(config, False)
            OmegaConf.set_readonly(config, False)
            yield config
        finally:
            OmegaConf.set_readonly(config, prev_state_readonly)
            OmegaConf.set_struct(config, prev_state_struct)
    else:
        try:
            config.defrost()
            yield config
        finally:
            config.freeze()
