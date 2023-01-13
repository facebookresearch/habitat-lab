from typing import TYPE_CHECKING, Optional, Tuple

from torch.optim.lr_scheduler import LambdaLR

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import Policy

if TYPE_CHECKING:
    from omegaconf import DictConfig


class AgentData:
    """
    Stores all the data specific to 1 policy.
    """

    def __init__(
        self,
        actor_critic: Policy,
        updater: Optional[PPO],
        is_static_encoder: bool,
        ppo_cfg: "DictConfig",
    ):
        """
        :param updater: If None, then the agent is not updated and is treated
            as fixed. This is needed when the agent is acting with a fixed partner.
        """

        self._rollouts: Optional[RolloutStorage] = None
        self._actor_critic = actor_critic
        self._updater = updater
        self._is_static_encoder = is_static_encoder
        self._lr_scheduler: Optional[LambdaLR] = None
        self.discrete_actions = False
        self.action_shape: Optional[Tuple[int]] = None
        self._ppo_cfg = ppo_cfg

    @property
    def encoder(self):
        return self.actor_critic.net.visual_encoder

    @property
    def rollouts(self) -> RolloutStorage:
        if self._rollouts is not None:
            raise ValueError("Rollout storage is not set.")
        return self._rollouts

    @property
    def actor_critic(self) -> Policy:
        return self._actor_critic

    @property
    def lr_scheduler(self) -> LambdaLR:
        return self._lr_scheduler

    @property
    def should_update(self) -> bool:
        return self._updater is not None

    @property
    def is_static_encoder(self) -> bool:
        return self._is_static_encoder

    @property
    def updater(self) -> Optional[PPO]:
        """
        The updater to update this policy. If None, then the policy should not
        be updated.
        """

        return self._updater

    def post_step(self):
        if self.should_update and self._ppo_cfg.use_linear_lr_decay:
            self._lr_scheduler.step()  # type: ignore

    def set_post_init_data(
        self,
        rollouts: RolloutStorage,
        discrete_actions: bool,
        action_shape: Tuple[int],
        lr_scheduler: Optional[LambdaLR],
    ) -> None:
        """
        Rollout creation happens at a later stage, which is why it's set later.
        """

        self._rollouts = rollouts
        self.discrete_actions = discrete_actions
        self.action_shape = action_shape
        self._lr_scheduler = lr_scheduler
