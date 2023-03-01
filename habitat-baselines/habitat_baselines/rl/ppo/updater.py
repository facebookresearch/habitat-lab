import abc
from typing import Dict

from habitat_baselines.common.storage import Storage


class Updater(abc.ABC):
    """
    Base Updater behavior.
    """

    @abc.abstractmethod
    def update(self, rollouts: Storage) -> Dict[str, float]:
        """
        Perform an update from data in the storage objet.
        """
