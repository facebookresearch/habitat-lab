from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import gym.spaces as spaces
import numpy as np
import torch

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.multi_agent.utils import (
    add_agent_names,
    add_agent_prefix,
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.rl.ppo.updater import Updater


@dataclass
class MultiAgentPolicyActionData(PolicyActionData):
    """
    Information returned from the `Policy.act` method representing the
    information from multiple agent's action. This class is needed to store
    actions of multiple agents together
    :property length_actions: List containing, for every agent, the size of
        their high-level action space.
    :property length_take_actions: List containing, for every agent, the size
        of their low-level action space.
    :property length_rnn_hidden_states: List containing for every agent the
        dimensionality of the rnn hidden state.
    :property num_agents: The number of agents represented in this
        `PolicyActionData`.
    """

    rnn_hidden_states: torch.Tensor
    actions: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    action_log_probs: Optional[torch.Tensor] = None
    take_actions: Optional[torch.Tensor] = None
    policy_info: Optional[List[Dict[str, Any]]] = None
    should_inserts: Optional[np.ndarray] = None

    # Indices
    length_rnn_hidden_states: Optional[torch.Tensor] = None
    length_actions: Optional[torch.Tensor] = None
    length_take_actions: Optional[torch.Tensor] = None
    num_agents: Optional[int] = 1

    def _unpack(self, tensor_to_unpack, unpack_lengths=None):
        """
        Splits the tensor tensor_to_unpack in the last dimension in the last dimension
        according to unpack lengths, so that the ith tensor will have unpack_lengths[i]
        in the last dimension. If unpack_lenghts is None, splits tensor_to_unpack evenly
        according to self.num_agents.
        :property tensor_to_unpack: The tensor we want to split into different chunks
        :unpack_lengths: List of integers indicating the sizes to unpack, or None if we want to unpack evenly
        """

        if unpack_lengths is None:
            unpack_lengths = [
                int(tensor_to_unpack.shape[-1] / self.num_agents)
            ] * self.num_agents

        return torch.split(tensor_to_unpack, unpack_lengths, dim=-1)

    def unpack(self):
        """
        Returns attributes of the policy unpacked per agent
        """
        return {
            "next_recurrent_hidden_states": self._unpack(
                self.rnn_hidden_states, self.length_rnn_hidden_states
            ),
            "actions": self._unpack(self.actions, self.length_actions),
            "value_preds": self._unpack(self.values),
            "action_log_probs": self._unpack(self.action_log_probs),
            "take_actions": self._unpack(
                self.take_actions, self.length_take_actions
            ),
            # This is numpy array and must be split differently.
            "should_inserts": np.split(
                self.should_inserts, self.num_agents, axis=-1
            ),
        }


def _merge_list_dict(inputs: List[List[Dict]]) -> List[Dict]:
    ret: List[Dict] = []
    for agent_i, ac in enumerate(inputs):
        if ac is None:
            continue
        for env_i, env_d in enumerate(ac):
            if len(ret) <= env_i:
                ret.append(
                    {add_agent_prefix(k, agent_i): v for k, v in env_d.items()}
                )
            else:
                for k, v in env_d.items():
                    ret[env_i][add_agent_prefix(k, agent_i)] = v
    return ret


class MultiPolicy(Policy):
    """
    Wraps a set of policies. Splits inputs and concatenates outputs.
    """

    def __init__(self, update_obs_with_agent_prefix_fn):
        self._active_policies = []
        if update_obs_with_agent_prefix_fn is None:
            update_obs_with_agent_prefix_fn = update_dict_with_agent_prefix
        self._update_obs_with_agent_prefix_fn = update_obs_with_agent_prefix_fn

    def set_active(self, active_policies):
        self._active_policies = active_policies

    def on_envs_pause(self, envs_to_pause):
        for policy in self._active_policies:
            policy.on_envs_pause(envs_to_pause)

    @property
    def hidden_state_shape_lens(self):
        """
        Get the length of the hidden states of all the policies
        """
        hidden_indices = [
            policy.hidden_state_shape[-1] for policy in self._active_policies
        ]
        return hidden_indices

    @property
    def hidden_state_shape(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        hidden_shapes = np.stack(
            [policy.hidden_state_shape for policy in self._active_policies]
        )
        # We do max because some policies may be non-neural
        # And will have a hidden state of [0, hidden_dim]
        max_hidden_shape = hidden_shapes.max(0)
        # The hidden states will be concatenated over the last dimension.
        return [*max_hidden_shape[:-1], np.sum(hidden_shapes[:, -1])]

    def update_hidden_state(self, rnn_hxs, prev_actions, action_data):
        # TODO: will not work with hidden states with different number of layers
        n_agents = len(self._active_policies)
        hxs_dim = rnn_hxs.shape[-1] // n_agents
        ac_dim = prev_actions.shape[-1] // n_agents
        # Not very efficient, but update each policies's hidden state individually.
        for env_i, should_insert in enumerate(action_data.should_inserts):
            for policy_i, agent_should_insert in enumerate(should_insert):
                if not agent_should_insert.item():
                    continue
                rnn_sel = slice(policy_i * hxs_dim, (policy_i + 1) * hxs_dim)
                rnn_hxs[env_i, :, rnn_sel] = action_data.rnn_hidden_states[
                    env_i, :, rnn_sel
                ]

                ac_sel = slice(policy_i * ac_dim, (policy_i + 1) * ac_dim)
                prev_actions[env_i, ac_sel].copy_(
                    action_data.actions[env_i, ac_sel]
                )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ):
        n_agents = len(self._active_policies)
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = rnn_hidden_states.split(
            split_index_dict["index_len_recurrent_hidden_states"], -1
        )
        agent_prev_actions = prev_actions.split(
            split_index_dict["index_len_prev_actions"], -1
        )
        agent_masks = masks.split([1, 1], -1)
        agent_actions = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = self._update_obs_with_agent_prefix_fn(
                observations, agent_i
            )
            agent_actions.append(
                policy.act(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                    deterministic,
                )
            )
        policy_info = _merge_list_dict(
            [ac.policy_info for ac in agent_actions]
        )
        batch_size = masks.shape[0]
        device = masks.device

        action_dims = split_index_dict["index_len_prev_actions"]

        # We need to split the `take_actions` if they are being assigned from
        # `actions`. This will be the case if `take_actions` hasn't been
        # assigned, like in a monolithic policy where there is no policy
        # hierarchicy.
        if any(ac.take_actions is None for ac in agent_actions):
            length_take_actions = action_dims
        else:
            length_take_actions = None

        def _maybe_cat(get_dat, feature_dims, dtype):
            all_dat = [get_dat(ac) for ac in agent_actions]
            # Replace any None with dummy data.
            all_dat = [
                torch.zeros(
                    (batch_size, feature_dims[ind]), device=device, dtype=dtype
                )
                if dat is None
                else dat
                for ind, dat in enumerate(all_dat)
            ]
            return torch.cat(all_dat, -1)

        rnn_hidden_lengths = [
            ac.rnn_hidden_states.shape[-1] for ac in agent_actions
        ]
        return MultiAgentPolicyActionData(
            rnn_hidden_states=torch.cat(
                [ac.rnn_hidden_states for ac in agent_actions], -1
            ),
            actions=_maybe_cat(
                lambda ac: ac.actions, action_dims, prev_actions.dtype
            ),
            values=_maybe_cat(
                lambda ac: ac.values, [1] * len(agent_actions), torch.float32
            ),
            action_log_probs=_maybe_cat(
                lambda ac: ac.action_log_probs,
                [1] * len(agent_actions),
                torch.float32,
            ),
            take_actions=torch.cat(
                [
                    ac.take_actions
                    if ac.take_actions is not None
                    else ac.actions
                    for ac in agent_actions
                ],
                -1,
            ),
            policy_info=policy_info,
            should_inserts=np.concatenate(
                [
                    ac.should_inserts
                    if ac.should_inserts is not None
                    else np.ones(
                        (batch_size, 1), dtype=bool
                    )  # None for monolithic policy, the buffer should be updated
                    for ac in agent_actions
                ],
                -1,
            ),
            length_rnn_hidden_states=rnn_hidden_lengths,
            length_actions=action_dims,
            length_take_actions=length_take_actions,
            num_agents=n_agents,
        )

    def _build_index_split(self, rnn_hidden_states, prev_actions, kwargs):
        """
        Return a dictionary with rnn_hidden_states lengths and action lengths that
        will be used to split these tensors into different agents. If the lengths
        are already in kwargs, we return them as is, if not, we assume agents
        have the same action/hidden dimension, so the tensors will be split equally.
        Therefore, the lists become [dimension_tensor // num_agents] * num_agents
        """
        n_agents = len(self._active_policies)
        index_names = [
            "index_len_recurrent_hidden_states",
            "index_len_prev_actions",
        ]
        split_index_dict = {}
        for name_index in index_names:
            if name_index not in kwargs:
                if name_index == "index_len_recurrent_hidden_states":
                    all_dim = rnn_hidden_states.shape[-1]
                else:
                    all_dim = prev_actions.shape[-1]
                split_indices = int(all_dim / n_agents)
                split_indices = [split_indices] * n_agents
            else:
                split_indices = kwargs[name_index]
            split_index_dict[name_index] = split_indices
        return split_index_dict

    def get_value(
        self, observations, rnn_hidden_states, prev_actions, masks, **kwargs
    ):
        split_index_dict = self._build_index_split(
            rnn_hidden_states, prev_actions, kwargs
        )
        agent_rnn_hidden_states = torch.split(
            rnn_hidden_states,
            split_index_dict["index_len_recurrent_hidden_states"],
            dim=-1,
        )
        agent_prev_actions = torch.split(
            prev_actions, split_index_dict["index_len_prev_actions"], dim=-1
        )
        agent_masks = torch.split(masks, [1, 1], dim=-1)
        all_value = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = self._update_obs_with_agent_prefix_fn(
                observations, agent_i
            )
            all_value.append(
                policy.get_value(
                    agent_obs,
                    agent_rnn_hidden_states[agent_i],
                    agent_prev_actions[agent_i],
                    agent_masks[agent_i],
                )
            )
        return torch.stack(all_value, -1)

    def get_extra(
        self, action_data: PolicyActionData, infos, dones
    ) -> List[Dict[str, float]]:
        all_extra = []
        for policy in self._active_policies:
            all_extra.append(policy.get_extra(action_data, infos, dones))
        # The action_data is shared across all policies, so no need to reutrn multiple times
        inputs = all_extra[0]
        ret: List[Dict] = []
        for env_d in inputs:
            ret.append(env_d)

        return ret

    @property
    def policy_action_space(self):
        # TODO: Hack for discrete HL action spaces.
        all_discrete = np.all(
            [
                isinstance(policy.policy_action_space, spaces.MultiDiscrete)
                for policy in self._active_policies
            ]
        )
        if all_discrete:
            return spaces.MultiDiscrete(
                tuple(
                    [
                        policy.policy_action_space.n
                        for policy in self._active_policies
                    ]
                )
            )
        else:
            return spaces.Dict(
                {
                    policy_i: policy.policy_action_space
                    for policy_i, policy in enumerate(self._active_policies)
                }
            )

    @property
    def policy_action_space_shape_lens(self):
        lens = []
        for policy in self._active_policies:
            if isinstance(policy.policy_action_space, spaces.Discrete):
                lens.append(1)
            elif isinstance(policy.policy_action_space, spaces.Box):
                lens.append(policy.policy_action_space.shape[0])
            else:
                raise ValueError(
                    f"Action distribution {policy.policy_action_space}"
                    "not supported."
                )
        return lens

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn)


class MultiStorage(Storage):
    def __init__(self, update_obs_with_agent_prefix_fn, **kwargs):
        self._active_storages = []
        self._agent_type_ids = []
        if update_obs_with_agent_prefix_fn is None:
            update_obs_with_agent_prefix_fn = update_dict_with_agent_prefix
        self._update_obs_with_agent_prefix_fn = update_obs_with_agent_prefix_fn

    def set_active(self, active_storages, agent_type_ids):
        self._agent_type_ids = agent_type_ids
        self._active_storages = active_storages

    def insert(
        self,
        next_observations=None,
        rewards=None,
        buffer_index=0,
        next_masks=None,
        **kwargs,
    ):
        n_agents = len(self._active_storages)

        def _maybe_chunk(tensor):
            if tensor is None:
                return None
            elif isinstance(tensor, np.ndarray):
                return np.split(tensor, n_agents, axis=-1)
            else:
                return tensor.chunk(n_agents, -1)

        # Assumed that all other arguments are tensors that need to be chunked
        # per-agent.
        if "action_data" not in kwargs:
            insert_d = {k: _maybe_chunk(v) for k, v in kwargs.items()}
        else:
            insert_d = {
                k: list(v)
                for k, v in kwargs["action_data"].unpack().items()
                if k in kwargs
            }
            args1 = sorted(list(insert_d.keys()) + ["action_data"])
            args2 = sorted(kwargs.keys())
            assert (
                args1 == args2
            ), "You are trying to insert more values than those defined in the PolicyActionData"

        for agent_i, storage in enumerate(self._active_storages):
            # TODO: this only works if we assume that the policy will always be recurrent
            if (
                "next_recurrent_hidden_states" in insert_d
                and insert_d["next_recurrent_hidden_states"][agent_i].numel()
                == 0
            ):
                insert_d["next_recurrent_hidden_states"][agent_i] = None

            if next_observations is not None:
                agent_next_observations = (
                    self._update_obs_with_agent_prefix_fn(
                        next_observations, agent_i
                    )
                )
            else:
                agent_next_observations = None
            storage.insert(
                next_observations=agent_next_observations,
                rewards=rewards,
                buffer_index=buffer_index,
                next_masks=next_masks,
                **{
                    k: v[agent_i] if v is not None else v
                    for k, v in insert_d.items()
                },
            )

    def to(self, device):
        # The active storages already need to be on the correct device.
        pass

    def insert_first_observations(self, batch):
        for agent_i, storage in enumerate(self._active_storages):
            agent_idx = self._agent_type_ids[agent_i]
            obs_dict = self._update_obs_with_agent_prefix_fn(batch, agent_idx)
            storage.insert_first_observations(obs_dict)

    def advance_rollout(self, buffer_index=0):
        for storage in self._active_storages:
            storage.advance_rollout(buffer_index)

    def compute_returns(self, next_value, use_gae, gamma, tau):
        for storage in self._active_storages:
            storage.compute_returns(next_value, use_gae, gamma, tau)

    def after_update(self):
        for storage in self._active_storages:
            storage.after_update()

    def _merge_step_outputs(self, get_step):
        obs: Dict[str, torch.Tensor] = {}
        agent_step_data: Dict[str, Any] = defaultdict(list)
        for agent_i, storage in enumerate(self._active_storages):
            agent_step = get_step(storage)
            add_agent_names(agent_step["observations"], obs, agent_i)
            for k, v in agent_step.items():
                if k == "observations":
                    continue
                agent_step_data[k].append(v)
        obs = TensorDict(obs)
        new_agent_step_data = {}

        # Concatenate the fields in agent_step_data in the last dimension.
        # Since we want to be able to split this tensor later, we store the original lenghts
        # of the tensor, stored as index_len_{tensor_name}
        for k in agent_step_data:
            new_name = "index_len_" + k
            lengths_data = [
                t.shape[-1] if t.numel() > 0 else 0 for t in agent_step_data[k]
            ]
            as_data_greater = [
                as_data
                for as_data in agent_step_data[k]
                if as_data.numel() > 0
            ]
            new_agent_step_data[k] = torch.cat(as_data_greater, dim=-1)
            new_agent_step_data[new_name] = lengths_data

        agent_step_data = dict(new_agent_step_data)
        agent_step_data["observations"] = obs

        return TensorDict(agent_step_data)

    def get_current_step(self, env_slice, buffer_index):
        return self._merge_step_outputs(
            lambda storage: storage.get_current_step(env_slice, buffer_index)
        )

    def get_last_step(self):
        return self._merge_step_outputs(
            lambda storage: storage.get_last_step()
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        update_obs_with_agent_prefix_fn: Optional[Callable] = None,
        **kwargs,
    ):
        return cls(update_obs_with_agent_prefix_fn, **kwargs)


class MultiUpdater(Updater):
    def __init__(self):
        self._active_updaters = []

    def set_active(self, active_updaters):
        self._active_updaters = active_updaters

    def update(self, rollouts):
        assert isinstance(rollouts, MultiStorage)

        losses: Dict[str, float] = {}
        for agent_i, (rollout, updater) in enumerate(
            zip(rollouts._active_storages, self._active_updaters)
        ):
            if len(list(updater.parameters())) > 0:
                agent_losses = updater.update(rollout)
                add_agent_names(agent_losses, losses, agent_i)
        return losses

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiUpdater()
