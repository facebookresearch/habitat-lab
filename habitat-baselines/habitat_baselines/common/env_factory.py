from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from habitat import VectorEnv
from habitat_baselines.common.tensor_dict import DictTree, TensorDict

if TYPE_CHECKING:
    from omegaconf import DictConfig


class VectorEnvFactory(ABC):
    @abstractmethod
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        is_eval: bool = False,
    ) -> VectorEnv:
        pass
