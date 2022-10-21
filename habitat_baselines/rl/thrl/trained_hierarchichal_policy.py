from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy
from habitat_baselines.rl.ppo.policy import Policy
import gym.spaces as spaces
from habitat.core.spaces import ActionSpace
import copy
from gym import spaces
import numpy as np
import torch
from torch import nn as nn

@baseline_registry.register_policy
class TrainedHierarchicalPolicy(nn.Module, Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space,
        original_action_space: ActionSpace,
        num_envs: int,
    ):
        super().__init__()
        self.hp = HierarchicalPolicy(
            config,
            full_config,
            observation_space,
            original_action_space,
            num_envs,
        )

        if not full_config.RL.POLICY.get("order_keys", False):
            fuse_keys = full_config.TASK_CONFIG.GYM.OBS_KEYS
        else:
            fuse_keys = None

        new_observation_space = copy.deepcopy(observation_space)
        #a = action_space.low[:-1]
        #b = action_space.high[:-1]
        #new_action_space = spaces.Box(a, b, dtype=np.float32)
        new_observation_space["proposed_action"] = action_space

        '''
        new_action_space = copy.deepcopy(original_action_space)
        arm_action_space = spaces.Box(
            original_action_space["ARM_ACTION"]["arm_action"].low / 10,
            original_action_space["ARM_ACTION"]["arm_action"].high / 10,
            dtype=np.float32,
        )
        new_action_space["ARM_ACTION"]["arm_action"] = arm_action_space
        base_action_space = spaces.Box(
            original_action_space["BASE_VELOCITY"]["base_vel"].low / 10,
            original_action_space["BASE_VELOCITY"]["base_vel"].high / 10,
            dtype=np.float32,
        )
        new_action_space["BASE_VELOCITY"]["base_vel"] = base_action_space
        '''

        a = action_space.low
        a[-1] = -1
        a[-4] = -1
        b = action_space.high
        b[-1] = 1
        b[-4] = 1
        new_action_space = spaces.Box(a, b, dtype=np.float32)

        self.metacontroler_net = PointNavResNetPolicy.from_config(
            full_config, new_observation_space, new_action_space
        )
        self.net = self.metacontroler_net.net

    def eval(self):
        pass

    @property
    def num_recurrent_layers(self):
        return self.hp._skills[0].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self.metacontroler_net.parameters()

    def to(self, device):
        self.metacontroler_net.to(device)
        self.net.to(device)
        self.hp.to(device)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        rnn_1, rnn_2 = rnn_hidden_states

        with torch.no_grad():
            _, n_actions, _, rnn_hidden_states_1 = self.hp.act(
                observations, rnn_1, prev_actions, masks, deterministic
            )
            n_actions = n_actions.to(observations[list(observations.keys())[0]].device)
            new_observations = copy.deepcopy(observations)
            new_observations["proposed_action"] = n_actions
        
        value, actions, action_log_probs, rnn_hidden_states_2 = self.metacontroler_net.act(
            new_observations, rnn_2, prev_actions, masks, deterministic
        )
        linear_layer = nn.Linear(actions.size()[1]*2, actions.size()[1]).to(actions.device)
        action_response = linear_layer(torch.cat((actions, n_actions), dim=1))
        #actions[:,-1] = 0
        #action_response = new_observations["proposed_action"] + actions/4
        #action_response = n_actions
        #from IPython import embed; embed()

        return (
            value,
            action_response,
            action_log_probs,
            torch.cat(
                [
                    rnn_hidden_states_1.unsqueeze(0),
                    rnn_hidden_states_2.unsqueeze(0),
                ],
                dim=0,
            ),
        )

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        return self.metacontroler_net.get_value(
            observations, rnn_hidden_states, prev_actions, masks
        )

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        return self.metacontroler_net.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        **kwargs,
    ):
        return cls(
            config.RL.POLICY,
            config,
            observation_space,
            action_space,
            orig_action_space,
            config.NUM_ENVIRONMENTS,
        )
