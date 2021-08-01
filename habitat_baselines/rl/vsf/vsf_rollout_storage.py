from multiprocessing.context import BaseContext
from typing import Optional, Tuple

import torch

from habitat_baselines.common.rollout_storage import RolloutStorage


class VSFRolloutStorage(RolloutStorage):
    def __init__(
        self,
        mp_ctx: BaseContext,
        n_policy_workers: int,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        discrete_actions: bool = True,
    ):
        super().__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            action_shape,
            False,
            discrete_actions,
        )

        self.current_steps = torch.zeros((num_envs,), dtype=torch.int64)
        self.n_actors_done = torch.zeros((1,), dtype=torch.int64)
        self.lock = mp_ctx.Lock()
        self.storage_free = mp_ctx.Event()
        self.storage_free.is_set()
        self.all_done = mp_ctx.Barrier(n_policy_workers)

    def share_memory_(self):
        self.buffers.map(lambda t: t.share_memory_())

        self.current_steps.share_memory_()
        self.n_actors_done.share_memory_()

    def to(self, device):
        super().to(device)
        self.current_steps = self.current_steps.to(device=device)

    def after_update(self):
        with self.lock:
            self.buffers[0:1] = self.buffers[self.current_steps - 1]

            self.current_steps.fill_(0)
            self.n_actors_done.fill_(0)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        self.storage_free.set()

    @torch.no_grad()
    def compute_returns(self, use_gae, gamma, tau, use_v_trace, actor_critic):
        with self.lock:
            self.current_rollout_step_idxs[0] = self.numsteps
            if not use_gae:
                tau = 1.0

            rewards = self.buffers["rewards"].to(
                device="cpu", non_blocking=True
            )
            returns = self.buffers["returns"].to(
                device="cpu", non_blocking=True
            )
            masks = self.buffers["masks"].to(device="cpu", non_blocking=True)
            if use_v_trace:
                batch = self.buffers.map(lambda t: t.view(-1, *t.size()[2:]))
                (
                    values,
                    action_log_probs,
                    _,
                    _,
                ) = actor_critic.evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                )

                ratios = torch.exp(
                    action_log_probs - batch["action_log_probs"]
                ).view_as(rewards)
                ratios.clamp_(max=1.0)
                ratios = ratios.to(device="cpu", non_blocking=True)
                values = values.view_as(rewards).to(
                    device="cpu", non_blocking=True
                )

            else:
                values = self.buffers["value_preds"].to(
                    device="cpu", non_blocking=True
                )
                ratios = torch.ones_like(values)

            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    rewards[step]
                    + gamma * values[step + 1] * masks[step + 1]
                    - values[step]
                )
                gae = ratios[step] * (
                    delta + gamma * tau * gae * masks[step + 1]
                )
                returns[step] = gae + values[step]

            self.buffers["returns"].copy_(returns)
