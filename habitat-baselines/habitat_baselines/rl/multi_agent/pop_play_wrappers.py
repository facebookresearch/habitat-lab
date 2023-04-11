from collections import defaultdict
from typing import Dict, List

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
    ):
        n_agents = len(self._active_policies)

        agent_rnn_hidden_states = rnn_hidden_states.chunk(n_agents, -1)
        agent_prev_actions = prev_actions.chunk(n_agents, -1)
        agent_masks = masks.chunk(n_agents, -1)

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
        action_dim = prev_actions.shape[-1] // n_agents

        def _maybe_cat(get_dat, feature_dim, dtype):
            all_dat = [get_dat(ac) for ac in agent_actions]
            # Replace any None with dummy data.
            all_dat = [
                torch.zeros(
                    (batch_size, feature_dim), device=device, dtype=dtype
                )
                if dat is None
                else dat
                for dat in all_dat
            ]
            return torch.cat(all_dat, -1)

        return PolicyActionData(
            rnn_hidden_states=torch.cat(
                [ac.rnn_hidden_states for ac in agent_actions], -1
            ),
            actions=_maybe_cat(
                lambda ac: ac.actions, action_dim, prev_actions.dtype
            ),
            values=_maybe_cat(lambda ac: ac.values, 1, torch.float32),
            action_log_probs=_maybe_cat(
                lambda ac: ac.action_log_probs, 1, torch.float32
            ),
            take_actions=torch.cat(
                [ac.take_actions for ac in agent_actions], -1
            ),
            policy_info=policy_info,
            should_inserts=torch.cat(
                [ac.should_inserts for ac in agent_actions], -1
            ),
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        n_agents = len(self._active_policies)
        agent_rnn_hidden_states = rnn_hidden_states.chunk(n_agents, -1)
        agent_prev_actions = prev_actions.chunk(n_agents, -1)
        agent_masks = masks.chunk(n_agents, -1)
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
        # The action_data is shared across all policies, so no need ot reutrn multiple times
        inputs = all_extra[0]
        ret: List[Dict] = []
        for env_d in inputs:
            ret.append(env_d)

        return ret

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
        insert_d = {k: _maybe_chunk(v) for k, v in kwargs.items()}
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
        agent_step_data = defaultdict(list)
        for agent_i, storage in enumerate(self._active_storages):
            agent_step = get_step(storage)
            add_agent_names(agent_step["observations"], obs, agent_i)
            for k, v in agent_step.items():
                if k == "observations":
                    continue
                agent_step_data[k].append(v)
        obs = TensorDict(obs)
        for k in agent_step_data:
            agent_step_data[k] = torch.cat(agent_step_data[k], dim=-1)
        agent_step_data = dict(agent_step_data)
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
            agent_losses = updater.update(rollout)
            add_agent_names(losses, agent_losses, agent_i)
        return losses

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiUpdater()


def _merge_list_dict(inputs: List[List[Dict]]) -> List[Dict]:
    ret: List[Dict] = []
    for agent_i, ac in enumerate(inputs):
        for env_i, env_d in enumerate(ac):
            if len(ret) <= env_i:
                ret.append(
                    {add_agent_prefix(k, agent_i): v for k, v in env_d.items()}
                )
            else:
                for k, v in env_d.items():
                    ret[env_i][add_agent_prefix(k, agent_i)] = v
    return ret
