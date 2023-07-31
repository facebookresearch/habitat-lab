from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

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
        make_env_fn: Callable = None,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        """
        Setup a vectorized environment.

        :param config: configs that contain num_environments as well as information
        :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
        :param enforce_scenes_greater_eq_environments: Make sure that there are more (or equal)
        :param enforce_scenes_greater_eq_environments: Make sure that there are more (or equal)
            scenes than environments. This is needed for correct evaluation.
        :param make_env_fn: function which creates a single environment. An
            environment can be of type :ref:`env.Env`,  :ref:`env.RLEnv`.
        :param is_first_rank: If these environments are being constructed on the rank0 GPU.

        :return: VectorEnv object created according to specification.
        """
