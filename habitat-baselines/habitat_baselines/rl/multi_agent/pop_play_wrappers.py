from collections import defaultdict
from typing import Dict, List

import torch

from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.rl.ppo.updater import Updater


class MultiPolicy(Policy):
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
        breakpoint()
        print("here")

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    def get_extra(self, action_data: PolicyActionData, infos, dones):
        pass

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiPolicy()


class MultiStorage(Storage):
    def __init__(self):
        self._active_storages = []

    def set_active(self, active_storages):
        self._active_storages = active_storages

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index=0,
        **kwargs,
    ):
        pass

    def to(self, device):
        # The active storages already need to be on the correct device.
        pass

    def insert_first(self, batch):
        breakpoint()
        print("done")

    def advance_rollout(self, buffer_index=0):
        pass

    def compute_returns(self, next_value, use_gae, gamma, tau):
        pass

    def after_update(self):
        pass

    def get_current_step(self, env_slice, buffer_index):
        obs = {}
        agent_step_data = defaultdict(list)
        for agent_i, storage in enumerate(self._active_storages):
            agent_step = storage.get_current_step(env_slice, buffer_index)
            for k, v in agent_step["observations"]:
                obs[f"agent_{agent_i}_{k}"] = v
            for k, v in agent_step.items():
                if k == "observations":
                    continue
                agent_step_data[k].append(v)
        obs = TensorDict(obs)
        for k in agent_step_data:
            agent_step_data[k] = torch.cat(agent_step_data[k], dim=1)
        agent_step_data = dict(agent_step_data)
        agent_step_data["observations"] = obs
        return TensorDict(obs)

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiStorage()


class MultiUpdater(Updater):
    def __init__(self):
        self._active_updaters = []

    def set_active(self, active_updaters):
        self._active_updaters = active_updaters

    def update(self, rollouts):
        pass

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return MultiUpdater()
