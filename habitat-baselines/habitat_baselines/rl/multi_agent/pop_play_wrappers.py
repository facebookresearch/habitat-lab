from typing import Dict, List

from habitat_baselines.common.storage import Storage
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
        pass

    def insert_first(self, batch):
        pass

    def advance_rollout(self, buffer_index=0):
        pass

    def compute_returns(self, next_value, use_gae, gamma, tau):
        pass

    def after_update(self):
        pass

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
