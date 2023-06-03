from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from habitat import VectorEnv

if TYPE_CHECKING:
    from omegaconf import DictConfig


class VectorEnvFactory(ABC):
    @abstractmethod
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        pass
