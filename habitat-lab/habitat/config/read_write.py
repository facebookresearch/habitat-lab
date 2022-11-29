from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Union

from omegaconf import OmegaConf
from omegaconf.base import Node

if TYPE_CHECKING:
    from omegaconf import DictConfig


# TODO : Delete this when migration to hydra is complete
@contextmanager
def read_write(
    config: Union[Node, "DictConfig"]
) -> Generator[Node, None, None]:
    prev_state_readonly = config._get_node_flag("readonly")
    prev_state_struct = config._get_node_flag("struct")
    try:
        OmegaConf.set_struct(config, False)
        OmegaConf.set_readonly(config, False)
        yield config
    finally:
        OmegaConf.set_readonly(config, prev_state_readonly)
        OmegaConf.set_struct(config, prev_state_struct)
