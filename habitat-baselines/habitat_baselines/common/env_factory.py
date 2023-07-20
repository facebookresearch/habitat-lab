from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from habitat import VectorEnv

if TYPE_CHECKING:
    from omegaconf import DictConfig


class VectorEnvFactory(ABC):
    """
    Interface responsible for constructing vectorized environments used in training.
    """

    @abstractmethod
    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        """
        Setup a vectorized environment.

        :param config: configs that contain num_environments as well as information
        :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
        :param enforce_scenes_greater_eq_environments: Make sure that there are more (or equal)
        :param enforce_scenes_greater_eq_environments: Make sure that there are more (or equal)
            scenes than environments. This is needed for correct evaluation.
        :param is_first_rank: If these environments are being constructed on the rank0 GPU.

        :return: VectorEnv object created according to specification.
        """
