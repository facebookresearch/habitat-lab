import math
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

        self.next_hidden_states = self.buffers["recurrent_hidden_states"][
            0
        ].clone()
        self.next_prev_actions = self.buffers["prev_actions"][0].clone()

        self.current_steps = torch.zeros((num_envs,), dtype=torch.int64)
        self.n_actors_done = torch.zeros((1,), dtype=torch.int64)
        self.rollout_done = torch.zeros((1,), dtype=torch.bool)

        self.lock = mp_ctx.Lock()
        self.storage_free = mp_ctx.Event()
        self.storage_free.is_set()
        self.storage_free.set()

        self.all_done = mp_ctx.Barrier(n_policy_workers)

    def share_memory_(self):
        self.buffers.map_in_place(lambda t: t.share_memory_())

        self.current_steps.share_memory_()
        self.n_actors_done.share_memory_()
        self.rollout_done.share_memory_()
        self.next_hidden_states.share_memory_()
        self.next_prev_actions.share_memory_()

    def to(self, device):
        super().to(device)
        self.current_steps = self.current_steps.to(device=device)
        self.next_hidden_states = self.next_hidden_states.to(device=device)
        self.next_prev_actions = self.next_prev_actions.to(device=device)

    def after_update(self):
        with self.lock:
            # We do a full roll here, that way [1:numsteps+1] becomes
            # all full steps of experience!
            self.buffers.map(lambda t: t.copy_(torch.roll(t, 1, 0)))

            not_finished = self.current_steps <= self.numsteps
            finished = torch.logical_not(not_finished)
            self.current_steps.fill_(0)
            # If a rollout wasn't fully finished when we added,
            # then there is a step in flight, so we are actually
            # on step 1 for the next rollout, not step 0
            self.current_steps[not_finished] += 1

            # For rollouts that finished, we need to "replay" the last step, so we also
            # need to reset the next hidden state as currently it would be appropriate
            # for t=1, not t=0
            self.next_hidden_states[finished] = self.buffers[
                "recurrent_hidden_states"
            ][0, finished]
            self.next_prev_actions[finished] = self.buffers["prev_actions"][
                0, finished
            ]

            self.buffers["rewards"][
                self.current_steps, range(self._num_envs)
            ] = 0.0

            self.n_actors_done.fill_(0)
            self.rollout_done.fill_(False)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        self.storage_free.set()

    @torch.no_grad()
    def compute_returns(
        self, use_gae, gamma, tau, num_mini_batch, use_v_trace, actor_critic
    ):
        with self.lock:
            self.current_rollout_step_idxs[0] = self.numsteps
            if not use_gae:
                tau = 1.0

            has_stale = False
            if torch.any(self.current_steps <= self.numsteps):
                has_stale = True
                num_stale_needed = (self.numsteps + 1) - self.current_steps
                max_stale = num_stale_needed.max().item()
                indexer = torch.arange(
                    0, self.numsteps + 1, device=self.device
                ).view(-1, 1) - num_stale_needed.view(1, -1)

                is_stale = indexer < 0
                indexer[is_stale] += self.numsteps + 1
                indexer *= self._num_envs
                indexer += torch.arange(
                    0, self._num_envs, device=self.device
                ).view(1, -1)
                indexer = indexer.long().view(-1)
                if (
                    indexer.max().item()
                    >= (self.numsteps + 1) * self._num_envs
                ):
                    print(self.current_steps)

                    raise RuntimeError("Uh oh")

                self.buffers.map(
                    lambda t: t.copy_(
                        t.view(-1, *t.size()[2:])
                        .index_select(0, indexer)
                        .view_as(t)
                    )
                )

            rewards = self.buffers["rewards"].to(
                device="cpu", non_blocking=True
            )
            returns = self.buffers["returns"].to(
                device="cpu", non_blocking=True
            )
            masks = self.buffers["masks"].to(device="cpu", non_blocking=True)
            if use_v_trace and has_stale:

                values, ratios = [], []
                values = self.buffers["value_preds"].clone()
                ratios = torch.ones_like(values)

                n_slices = num_mini_batch
                slice_size = int(math.floor(self._num_envs / n_slices))
                for start_idx in range(0, self._num_envs, slice_size):
                    idx = (
                        slice(0, max_stale),
                        slice(start_idx, start_idx + slice_size),
                    )
                    batch = self.buffers[idx].map(lambda t: t.flatten(0, 1))

                    v, a, _, _ = actor_critic.evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                    )

                    values[idx].copy_(v.view(max_stale, -1, 1))
                    ratios[idx].copy_(
                        torch.exp(a - batch["action_log_probs"]).view(
                            max_stale, -1, 1
                        )
                    )

                ratios.clamp_(max=1.0)
                ratios = ratios.to(device="cpu", non_blocking=True)
                values = values.to(device="cpu", non_blocking=True)

            else:
                values = self.buffers["value_preds"].to(
                    device="cpu", non_blocking=True
                )
                ratios = torch.ones_like(values)

            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device)

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
