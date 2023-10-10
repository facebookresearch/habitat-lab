from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ppo.updater import Updater

if TYPE_CHECKING:
    from omegaconf import DictConfig


class AgentAccessMgr(ABC):
    """
    Consists of:
    - Policy: How actions are selected from observations.
    - Data Storage: How data collected from the environment is stored.
    - Updater: How the Policy is updated.
    """

    @abstractmethod
    def __init__(
        self,
        config: "DictConfig",
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done_fn: Callable[[], float],
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
    ):
        pass

    @property
    @abstractmethod
    def nbuffers(self) -> int:
        """
        Number of storage buffers.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def masks_shape(self) -> Tuple:
        """
        Shape of the masks tensor.
        """
        raise NotImplementedError()

    @abstractmethod
    def post_init(self, create_rollouts_fn: Optional[Callable] = None) -> None:
        """
        Called after the constructor. Sets up the rollout storage.

        :param create_rollouts_fn: Override behavior for creating the
            rollout storage. Default behavior for this and the call signature is
            `default_create_rollouts`.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def rollouts(self) -> Storage:
        """
        Gets the current rollout storage.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def actor_critic(self) -> Policy:
        """
        Gets the current policy
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def updater(self) -> Updater:
        """
        Gets the current policy updater.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_resume_state(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_save_state(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load_ckpt_state_dict(self, ckpt: Dict) -> None:
        """
        Loads a state dict for evaluation. The difference from
        `load_state_dict` is that this will not load the policy state if the
        policy does not request it.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_state_dict(self, state: Dict) -> None:
        raise NotImplementedError()

    @abstractmethod
    def after_update(self) -> None:
        """
        Must be called by the trainer after the updater has called `update` and
        the rollout `after_update` is called.
        """

    @abstractmethod
    def pre_rollout(self) -> None:
        """
        Called before a rollout is collected.
        """
        raise NotImplementedError()
