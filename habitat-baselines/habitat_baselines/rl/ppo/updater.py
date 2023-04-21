import abc
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ppo.policy import NetPolicy


class Updater(abc.ABC):
    """
    Base Updater behavior.
    """

    @abc.abstractmethod
    def update(self, rollouts: Storage) -> Dict[str, float]:
        """
        Perform an update from data in the storage objet.
        """

    @classmethod
    @abc.abstractmethod
    def from_config(
        cls, actor_critic: NetPolicy, config: "DictConfig"
    ) -> "Updater":
        """
        Instantiate the Updater object from the actor_critic to update and the config.
        """
