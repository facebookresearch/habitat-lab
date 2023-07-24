import abc
from typing import Any, Dict

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

    @property
    def lr_scheduler(self):
        return None

    def after_update(self) -> None:
        """
        Called after the policy update.
        """

    def get_resume_state(self) -> Dict[str, Any]:
        """
        Gets the optimizer resume state.
        """

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Loads an optimizer state.
        """
