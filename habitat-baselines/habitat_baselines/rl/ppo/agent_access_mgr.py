from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gym.spaces as spaces

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

    @abstractmethod
    def init_distributed(self, find_unused_params: bool = True) -> None:
        """
        Setup any components for distributed training.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_hidden_state(self, rnn_hxs, prev_actions, action_data):
        """
        Update the hidden state of the policies in the population. Writes to the
        data in place.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def nbuffers(self) -> int:
        """
        Number of storage buffers.
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
    def policy_action_space(self) -> spaces.Space:
        """
        The action space the policy acts in. This can be different from the
        environment action space for hierarchical policies.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def policy_action_space_shape_lens(self) -> List[int]:
        """
        A list with the dimensionality of action space of each of the agents.
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

    @property
    @abstractmethod
    def hidden_state_shape(self) -> Tuple[int]:
        """
        The shape of the tensor to track the hidden state, such as the RNN hidden state.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def hidden_state_shape_lens(self) -> List[int]:
        """
        A list with the dimensionality of the hidden state of each of the agents.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def masks_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def after_update(self) -> None:
        """
        Must be called by the trainer after the updater has called `update` and
        the rollout `after_update` is called.
        """
        raise NotImplementedError()

    @abstractmethod
    def pre_rollout(self) -> None:
        """
        Called before a rollout is collected.
        """
        raise NotImplementedError()
