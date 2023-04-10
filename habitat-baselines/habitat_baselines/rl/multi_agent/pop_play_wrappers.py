from collections import defaultdict
from typing import Any, Dict, List

import torch

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.multi_agent.utils import (
    add_agent_names,
    add_agent_prefix,
    update_dict_with_agent_prefix,
)
from habitat_baselines.rl.ppo.policy import (
    MultiAgentPolicyActionData,
    Policy,
    PolicyActionData,
)
from habitat_baselines.rl.ppo.updater import Updater


class MultiPolicy(Policy):
    """
    Wraps a set of policies. Splits inputs and concatenates outputs.
    """

    def __init__(self):
        self._active_policies = []

    def set_active(self, active_policies):
        self._active_policies = active_policies

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

        agent_rnn_hidden_states = rnn_hidden_states.split(
            split_index_dict["index_len_recurrent_hidden_states"], -1
        )
        agent_prev_actions = prev_actions.split(
            split_index_dict["index_len_prev_actions"], -1
        )
        agent_masks = masks.split([1, 1], -1)
        agent_actions = []
        for agent_i, policy in enumerate(self._active_policies):
            agent_obs = update_dict_with_agent_prefix(observations, agent_i)
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

        # Action dim is split evenly between the agents.
        action_dims = split_index_dict["index_len_prev_actions"]

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
            should_inserts=torch.cat(
                [
                    ac.should_inserts
                    if ac.should_inserts is not None
                    else torch.ones((batch_size, 1), dtype=torch.bool)
                    for ac in agent_actions
                ],
                -1,
            ),
            length_rnn_hidden_states=rnn_hidden_lengths,
            length_actions=action_dims,
            num_agents=n_agents,
        )

    def get_value(
        self, observations, rnn_hidden_states, prev_actions, masks, **kwargs
    ):
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
            agent_obs = update_dict_with_agent_prefix(observations, agent_i)
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
        return _merge_list_dict(all_extra)

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiPolicy()


class MultiStorage(Storage):
    def __init__(self):
        self._active_storages = []
        self._agent_type_ids = []

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
            else:
                return tensor.chunk(n_agents, -1)

        # Assumed that all other arguments are tensors that need to be chunked
        # per-agent.
        if "action_data" not in kwargs:
            insert_d = {k: _maybe_chunk(v) for k, v in kwargs.items()}
        else:
            insert_d = {
                k: v
                for k, v in kwargs["action_data"].unpack().items()
                if k in kwargs
            }
            args1 = sorted(list(insert_d.keys()) + ["action_data"])
            args2 = sorted(kwargs.keys())
            assert args1 == args2

        for agent_i, storage in enumerate(self._active_storages):
            agent_type_idx = self._agent_type_ids[agent_i]
            if next_observations is not None:
                agent_next_observations = update_dict_with_agent_prefix(
                    next_observations, agent_type_idx
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
            obs_dict = update_dict_with_agent_prefix(batch, agent_idx)
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
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiStorage()


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
            if len(list(updater.parameters())):
                agent_losses = updater.update(rollout)
                add_agent_names(losses, agent_losses, agent_i)
        return losses

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiUpdater()


def _merge_list_dict(inputs: List[List[Dict]]) -> List[Dict]:
    ret: List[Dict] = []
    for agent_i, ac in enumerate(inputs):
        if ac is None:
            ac = [{}]
        for env_i, env_d in enumerate(ac):
            if len(ret) <= env_i:
                ret.append(
                    {add_agent_prefix(k, agent_i): v for k, v in env_d.items()}
                )
            else:
                for k, v in env_d.items():
                    ret[env_i][add_agent_prefix(k, agent_i)] = v
    return ret
