import abc

import torch


class Storage(abc.ABC):
    """
    Storage interface.
    """

    @abc.abstractmethod
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def to(self, device) -> None:
        pass

    @abc.abstractmethod
    def insert_first_observations(self, batch) -> None:
        pass

    @abc.abstractmethod
    def advance_rollout(self, buffer_index: int = 0) -> None:
        pass

    @abc.abstractmethod
    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, tau: float
    ) -> None:
        pass

    @abc.abstractmethod
    def after_update(self) -> None:
        pass

    def get_last_step(self):
        pass

    def get_current_step(self, env_slice, buffer_index):
        pass
