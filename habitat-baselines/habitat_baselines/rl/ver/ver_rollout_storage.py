#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.models.rnn_state_encoder import (
    _np_invert_permutation,
    build_pack_info_from_episode_ids,
    build_rnn_build_seq_info,
)


def _np_unique_not_sorted(arr: np.ndarray) -> np.ndarray:
    sorted_unq_arr, indexes = np.unique(arr, return_index=True)
    # Indexes contains the index of the first instance of each unique
    # value in the array. So we can use as the sort-key to re-order
    # the sorted unique array such that each value is in the order
    # it appeared in the array.
    return sorted_unq_arr[np.argsort(indexes)]


def compute_movements_for_aliased_swaps(
    dst_locations: np.ndarray, src_locations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes the movements needed as a result of performing the all the swaps from
    source to dest.

    This is needed because just swapping t[dst_locations] with t[src_locations] will not be
    correct in the case of aliasing. This returns a tuple of (dst, src) such that the desired
    swaps can be performed as t[dst] = t[src]
    """

    assert len(dst_locations) == len(src_locations)

    # We can think of swapping as follows:
    # 1. We take all values from source and put them in their dst
    # location.
    # 2. Now the array has N - k duplicated values and N - k deleted values,
    # where N = len(dst_locations) and k = len(intersection(dst_locations, src_locations))
    # 3. The N - k delete values were at indices setdiff(dst_locations, src_locations)
    # and the duplicated values are at setdiff(src_locations, dst_locations). So we
    # put the deleted values into the locations of the duplicated values
    # The _np_unique_not_sorted calls bellow take
    # cat(a, b) and make it cat(a, setdiff(b, a))
    swap_dst = _np_unique_not_sorted(
        np.concatenate((dst_locations, src_locations), axis=0)
    )
    swap_src = _np_unique_not_sorted(
        np.concatenate((src_locations, dst_locations), axis=0)
    )

    assert len(swap_src) == len(swap_dst)
    return swap_dst, swap_src


def partition_n_into_p(n: int, p: int) -> List[int]:
    r"""Creates a partitioning of n elements into p bins."""
    return [n // p + (1 if i < (n % p) else 0) for i in range(p)]


def generate_ver_mini_batches(
    num_mini_batch: int,
    sequence_lengths: np.ndarray,
    num_seqs_at_step: np.ndarray,
    select_inds: np.ndarray,
    last_sequence_in_batch_mask: np.ndarray,
    episode_ids: np.ndarray,
) -> Iterator[np.ndarray]:
    r"""Generate mini-batches for VER.

    This works by taking all the sequences of experience, putting them in a random order,
    and then slicing their steps into :ref:`num_mini_batch` batches.
    """

    sequence_lengths = sequence_lengths.copy()
    # We don't want to include the last step from the last sequence from each environment
    # as this is the step we collected for bootstrapping the return.
    sequence_lengths[last_sequence_in_batch_mask] -= 1

    offset_to_step = (
        np.cumsum(num_seqs_at_step, dtype=np.int64) - num_seqs_at_step
    )
    seq_ordering = np.random.permutation(len(sequence_lengths))

    all_seq_steps_l = [
        select_inds[i + offset_to_step[0 : sequence_lengths[i]]]
        for i in range(len(sequence_lengths))
    ]

    for s_steps in all_seq_steps_l:
        if len(s_steps) == 0:
            continue

        assert len(np.unique(episode_ids[s_steps])) == 1, episode_ids[s_steps]

    all_seq_steps = np.concatenate(
        [all_seq_steps_l[seq] for seq in seq_ordering]
    )

    mb_sizes = np.array(
        partition_n_into_p(int(np.sum(sequence_lengths)), num_mini_batch),
        dtype=np.int64,
    )
    mb_starts = np.cumsum(mb_sizes, dtype=np.int64) - mb_sizes

    # Yield the mini-batches in random order since where each mb is cut
    # isn't random otherwise
    for mb_idx in np.random.permutation(num_mini_batch):
        yield all_seq_steps[
            mb_starts[mb_idx] : mb_starts[mb_idx] + mb_sizes[mb_idx]
        ]


class VERRolloutStorage(RolloutStorage):
    r"""Rollout storage for VER."""
    ptr: np.ndarray
    prev_inds: np.ndarray
    num_steps_collected: np.ndarray
    rollout_done: np.ndarray
    cpu_current_policy_version: np.ndarray
    actor_steps_collected: np.ndarray
    current_steps: np.ndarray
    will_replay_step: np.ndarray
    _first_rollout: np.ndarray

    next_hidden_states: torch.Tensor
    next_prev_actions: torch.Tensor
    current_policy_version: torch.Tensor

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        actor_critic,
        variable_experience: bool,
        is_double_buffered: bool = False,
    ):
        super().__init__(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            actor_critic,
            is_double_buffered,
        )
        self.use_is_coeffs = variable_experience

        if self.use_is_coeffs:
            self.buffers["is_coeffs"] = torch.ones_like(
                self.buffers["returns"]
            )

        for k in (
            "policy_version",
            "environment_ids",
            "episode_ids",
            "step_ids",
        ):
            self.buffers[k] = torch.zeros_like(
                self.buffers["returns"], dtype=torch.int64
            )

        self.buffers["is_stale"] = torch.ones_like(
            self.buffers["returns"], dtype=torch.bool
        )

        self.variable_experience = variable_experience
        self.buffer_size = (self.num_steps + 1) * self._num_envs

        # The aux buffers are auxiliary things that are accessed
        # via the annotations on this class. This is done to ensure
        # that all of these things end up in shared memory and that
        # we keep the torch tensor that backs the shared memory around.
        # We use numpy arrays to access CPU-side data as that is faster,
        # but use pytorch to manage the shared memory.
        self._aux_buffers = TensorDict()
        self.aux_buffers_on_device = set()

        assert isinstance(
            self.buffers["recurrent_hidden_states"], torch.Tensor
        )
        self._aux_buffers["next_hidden_states"] = self.buffers[
            "recurrent_hidden_states"
        ][0].clone()
        assert isinstance(self.buffers["prev_actions"], torch.Tensor)
        self._aux_buffers["next_prev_actions"] = self.buffers["prev_actions"][
            0
        ].clone()
        self._aux_buffers["current_policy_version"] = torch.ones(
            (1, 1), dtype=torch.int64
        )

        self.aux_buffers_on_device.update(set(self._aux_buffers.keys()))

        self._aux_buffers["cpu_current_policy_version"] = torch.ones(
            (1, 1), dtype=torch.int64
        )

        self._aux_buffers["num_steps_collected"] = torch.zeros(
            (1,), dtype=torch.int64
        )
        self._aux_buffers["rollout_done"] = torch.zeros((1,), dtype=torch.bool)
        self._aux_buffers["current_steps"] = torch.zeros(
            (num_envs,), dtype=torch.int64
        )
        self._aux_buffers["actor_steps_collected"] = torch.zeros(
            (num_envs,), dtype=torch.int64
        )

        self._aux_buffers["ptr"] = torch.zeros((1,), dtype=torch.int64)
        self._aux_buffers["prev_inds"] = torch.full(
            (num_envs,), -1, dtype=torch.int64
        )
        self._aux_buffers["_first_rollout"] = torch.full(
            (1,), True, dtype=torch.bool
        )
        self._aux_buffers["will_replay_step"] = torch.zeros(
            (num_envs,), dtype=torch.bool
        )

        if self.variable_experience:
            # In VER mode, there isn't a clean assignment from
            # (env_id, current_step) to place in the rollouts storage
            # anymore. Instead we treat this as just a linear buffer
            # and write into that.
            self.buffers.map_in_place(lambda t: t.flatten(0, 1))

        self._set_aux_buffers()

    @property
    def num_steps_to_collect(self) -> int:
        if self._first_rollout:
            return self.buffer_size
        else:
            return self._num_envs * self.num_steps

    def _set_aux_buffers(self):
        for k, v in self.__annotations__.items():
            if k not in self._aux_buffers:
                if v in (torch.Tensor, np.ndarray):
                    raise RuntimeError(f"Annotation {k} not in aux buffers")
                else:
                    continue

            if k in self.aux_buffers_on_device:  # noqa: SIM401
                buf = self._aux_buffers[k]
            else:
                buf = self._aux_buffers[k].numpy()

            assert isinstance(
                buf, v
            ), f"Expected aux buffer of type {v} but got {type(buf)}"
            setattr(self, k, buf)

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__().copy()

        # We have to remove the aux_buffers
        # that we assigned to the class via annotations
        # otherwise they will get copied by pickling
        for k in self.__annotations__.keys():
            if k in self._aux_buffers:
                del state[k]

        return state

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)

        # On unpickle, we reset the aux buffers.
        self._set_aux_buffers()

    def copy(self, other: "VERRolloutStorage"):
        self.buffers[:] = other.buffers
        self._aux_buffers[:] = other._aux_buffers

    def share_memory_(self):
        self.buffers.map_in_place(lambda t: t.share_memory_())

        self._aux_buffers.map_in_place(lambda t: t.share_memory_())

        self._set_aux_buffers()

    def to(self, device):
        super().to(device)

        for k, t in self._aux_buffers.items():
            if k in self.aux_buffers_on_device:
                assert isinstance(t, torch.Tensor)
                self._aux_buffers[k] = t.to(device=device)

        self._set_aux_buffers()

    def after_update(self):
        self.current_steps[:] = 1
        self.current_steps[self.will_replay_step] -= 1
        assert isinstance(self.buffers["is_stale"], torch.Tensor)
        self.buffers["is_stale"].fill_(True)

        if not self.variable_experience:
            assert np.all(self.will_replay_step)
            self.next_hidden_states[:] = self.buffers[
                "recurrent_hidden_states"
            ][-1]
            self.next_prev_actions[:] = self.buffers["prev_actions"][-1]
        else:
            # With ver, we will have some actions
            # in flight as we can't reset the simulator. In that case we need
            # to save the prev rollout step data for that action (as the
            # reward for the action in flight is for the prev rollout
            # step). We put those into [0:num_with_action_in_flight]
            # because that range won't be overwritten.
            has_action_in_flight = np.logical_not(self.will_replay_step)
            num_with_action_in_flight = np.count_nonzero(has_action_in_flight)

            prev_inds_for_swap: np.ndarray = np.concatenate(
                (
                    self.prev_inds[has_action_in_flight],
                    # For previous steps without an action in flight,
                    # we will "replay" that step of experience
                    # so we can use the new policy for inference.
                    # We need to make sure they get overwritten in the next
                    # rollout, so we place them at the
                    # start of where we will write during the next rollout.
                    self.prev_inds[np.logical_not(has_action_in_flight)],
                )
            )
            dst_locations, src_locations = map(
                lambda n: torch.from_numpy(n).to(device=self.device),
                compute_movements_for_aliased_swaps(
                    np.arange(len(prev_inds_for_swap)), prev_inds_for_swap
                ),
            )
            self.buffers[dst_locations] = self.buffers[src_locations]

            self.prev_inds[:] = -1
            self.prev_inds[has_action_in_flight] = np.arange(
                num_with_action_in_flight,
                dtype=self.prev_inds.dtype,
            )
            self.will_replay_step[:] = False
            self.ptr[:] = num_with_action_in_flight

            # For the remaining steps, we order them such that the oldest experience
            # get's overwritten first.
            assert isinstance(self.buffers["policy_version"], torch.Tensor)
            version_diff = (
                self.current_policy_version.view(-1)
                - self.buffers["policy_version"].view(-1)[self._num_envs :]
            )
            # Add index to make the sort stable
            _, version_ordering = torch.sort(
                version_diff * version_diff.numel()
                + torch.arange(
                    version_diff.numel() - 1,
                    -1,
                    step=-1,
                    dtype=version_diff.dtype,
                    device=version_diff.device,
                ),
                descending=True,
            )

            self.buffers.apply(
                lambda t: t[self._num_envs :].copy_(
                    t[self._num_envs :].index_select(0, version_ordering)
                )
            )

        self.num_steps_collected[:] = 0
        self.rollout_done[:] = False
        self._first_rollout[:] = False

    def increment_policy_version(self):
        self.current_policy_version += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.cpu_current_policy_version += 1

    def after_rollout(self):
        assert isinstance(self.buffers["policy_version"], torch.Tensor)
        assert isinstance(self.buffers["is_stale"], torch.Tensor)
        self.buffers["is_stale"][:] = (
            self.buffers["policy_version"] < self.current_policy_version
        )

        self.current_rollout_step_idxs[0] = self.num_steps + 1

        if self.use_is_coeffs:
            # To correct for the biased sampling in ver, we use importance
            # sampling weighting. To do this we must count the number of
            # steps of experience we got from each environment
            assert isinstance(self.buffers["environment_ids"], torch.Tensor)
            environment_ids = self.buffers["environment_ids"].view(-1)
            unique_envs = torch.unique(environment_ids, sorted=False)
            samples_per_env = (
                (environment_ids.view(1, -1) == unique_envs.view(-1, 1))
                .float()
                .sum(-1)
            )
            is_per_env = torch.empty(
                (self._num_envs,), dtype=torch.float32, device=self.device
            )
            # Use a scatter so that is_per_env.size() == num_envs
            # and so that we don't have to have unique_envs be sorted
            is_per_env.scatter_(
                0,
                unique_envs.view(-1),
                # With uniform sampling, we'd get (numsteps + 1)
                # per actor
                (self.num_steps + 1) / samples_per_env,
            )
            assert isinstance(self.buffers["is_coeffs"], torch.Tensor)
            self.buffers["is_coeffs"].copy_(
                is_per_env[environment_ids].view(-1, 1)
            )

    def compute_returns(
        self,
        use_gae,
        gamma,
        tau,
    ):
        if not use_gae:
            tau = 1.0

        assert isinstance(self.buffers["masks"], torch.Tensor)
        not_masks = torch.logical_not(self.buffers["masks"]).to(
            device="cpu", non_blocking=True
        )
        assert isinstance(self.buffers["episode_ids"], torch.Tensor)
        episode_ids_cpu_t = self.buffers["episode_ids"].to(
            device="cpu", non_blocking=True
        )
        assert isinstance(self.buffers["environment_ids"], torch.Tensor)
        environment_ids_cpu_t = self.buffers["environment_ids"].to(
            device="cpu", non_blocking=True
        )
        assert isinstance(self.buffers["step_ids"], torch.Tensor)
        step_ids_cpu_t = self.buffers["step_ids"].to(
            device="cpu", non_blocking=True
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.dones_cpu = not_masks.view(-1, self._num_envs).numpy()
        self.episode_ids_cpu = episode_ids_cpu_t.view(-1).numpy()
        self.environment_ids_cpu = environment_ids_cpu_t.view(-1).numpy()
        self.step_ids_cpu = step_ids_cpu_t.view(-1).numpy()

        assert isinstance(self.buffers["rewards"], torch.Tensor)
        rewards_t = self.buffers["rewards"].to(device="cpu", non_blocking=True)
        assert isinstance(self.buffers["returns"], torch.Tensor)
        returns_t = self.buffers["returns"].to(device="cpu", non_blocking=True)
        assert isinstance(self.buffers["is_stale"], torch.Tensor)
        is_not_stale_t = torch.logical_not(self.buffers["is_stale"]).to(
            device="cpu", non_blocking=True
        )
        assert isinstance(self.buffers["value_preds"], torch.Tensor)
        values_t = self.buffers["value_preds"].to(
            device="cpu", non_blocking=True
        )

        rnn_build_seq_info = build_pack_info_from_episode_ids(
            self.episode_ids_cpu,
            self.environment_ids_cpu,
            self.step_ids_cpu,
        )

        (
            self.select_inds,
            self.num_seqs_at_step,
            self.sequence_lengths,
            self.sequence_starts,
            self.last_sequence_in_batch_mask,
        ) = (
            rnn_build_seq_info["select_inds"],
            rnn_build_seq_info["num_seqs_at_step"],
            rnn_build_seq_info["sequence_lengths"],
            rnn_build_seq_info["sequence_starts"],
            rnn_build_seq_info["last_sequence_in_batch_mask"],
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        rewards, values, is_not_stale = map(
            lambda t: t.view(-1, 1).numpy()[self.select_inds],
            (rewards_t, values_t, is_not_stale_t),
        )
        returns = returns_t.view(-1, 1).numpy()
        returns[:] = returns[self.select_inds]

        gae = np.zeros((self.num_seqs_at_step[0], 1))
        last_values = gae.copy()
        ptr = returns.size
        for len_minus_1, n_seqs in reversed(
            list(enumerate(self.num_seqs_at_step))
        ):
            curr_slice = slice(ptr - n_seqs, ptr)
            q_est = rewards[curr_slice] + gamma * last_values[:n_seqs]
            delta = q_est - values[curr_slice]
            gae[:n_seqs] = delta + (tau * gamma) * gae[:n_seqs]

            is_last_step = self.sequence_lengths == (len_minus_1 + 1)
            is_last_step_for_env = (
                is_last_step & self.last_sequence_in_batch_mask
            )

            # For the last step from each worker, we do an extra loop
            # to fill last_values with the bootstrap.  So we re-zero
            # the GAE value here
            gae[is_last_step_for_env] = 0.0

            # If the step isn't stale or we don't have a return
            # calculate, use the newly calculated return value,
            # otherwise keep the current one
            use_new_value = is_not_stale[curr_slice] | np.logical_not(
                np.isfinite(returns[curr_slice])
            )
            returns[curr_slice][use_new_value] = (
                gae[:n_seqs] + values[curr_slice]
            )[use_new_value]

            # We also mark these with a nan
            returns[curr_slice][is_last_step_for_env[:n_seqs]] = float("nan")

            last_values[:n_seqs] = values[curr_slice]

            ptr -= n_seqs

        assert ptr == 0

        returns[:] = returns[_np_invert_permutation(self.select_inds)]

        if not self.variable_experience:
            assert torch.all(torch.isfinite(returns_t[:-1])), dict(
                returns=returns_t.squeeze(),
                dones=self.dones_cpu,
                episode_ids=self.episode_ids_cpu.reshape(-1, self._num_envs),
                environment_ids=self.environment_ids_cpu.reshape(
                    -1, self._num_envs
                ),
                step_ids=self.step_ids_cpu.reshape(-1, self._num_envs),
                is_not_stale=is_not_stale[
                    _np_invert_permutation(self.select_inds)
                ].reshape(-1, self._num_envs),
            )
        else:
            assert torch.isfinite(returns_t).long().sum() == (
                self.num_steps * self._num_envs
            ), returns_t.squeeze().numpy()[self.select_inds]

        self.buffers["returns"].copy_(returns_t, non_blocking=True)
        self.current_rollout_step_idxs[0] = self.num_steps

    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
    ) -> Iterator[DictTree]:
        if not self.variable_experience:
            yield from super().data_generator(advantages, num_mini_batch)
        else:
            for mb_inds in generate_ver_mini_batches(
                num_mini_batch,
                self.sequence_lengths,
                self.num_seqs_at_step,
                self.select_inds,
                self.last_sequence_in_batch_mask,
                self.episode_ids_cpu,
            ):
                mb_inds_cpu = torch.from_numpy(mb_inds)
                mb_inds = mb_inds_cpu.to(device=self.device)

                if not self.variable_experience:
                    batch = self.buffers.map(lambda t: t.flatten(0, 1))[
                        mb_inds
                    ]
                    if advantages is not None:
                        batch["advantages"] = advantages.flatten(0, 1)[mb_inds]
                else:
                    batch = self.buffers[mb_inds]
                    if advantages is not None:
                        batch["advantages"] = advantages[mb_inds]

                batch["rnn_build_seq_info"] = build_rnn_build_seq_info(
                    device=self.device,
                    build_fn_result=build_pack_info_from_episode_ids(
                        self.episode_ids_cpu[mb_inds_cpu],
                        self.environment_ids_cpu[mb_inds_cpu],
                        self.step_ids_cpu[mb_inds_cpu],
                    ),
                )

                rnn_build_seq_info = batch["rnn_build_seq_info"]
                assert isinstance(
                    batch["recurrent_hidden_states"], torch.Tensor
                )
                batch["recurrent_hidden_states"] = batch[
                    "recurrent_hidden_states"
                ].index_select(
                    0,
                    rnn_build_seq_info["first_step_for_env"],
                )

                yield batch.to_tree()
