from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Iterable, Union
from torch import Tensor
import abc

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet

from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy, Policy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import (
    CategoricalNet,
    get_num_actions,
)
from habitat_baselines.utils.timing import g_timer


if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_auxiliary_loss
class AuxL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, aux_loss_state_actor, aux_loss_state_critic):
        return self.mse(aux_loss_state_actor, aux_loss_state_critic)

@baseline_registry.register_policy
class PointNavGoalDistancePolicy(nn.Module, Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        hidden_size: int,
        num_rnn_layers: int,
        rnn_type: str,
        noise_coefficient: float = 0.1,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
    ) -> None:
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)
        
        if policy_config is not None:
            assert policy_config.action_distribution_type == "categorical", f"Unsupported action distribution type {policy_config.action_distribution_type}"
        
        self.dim_actions = get_num_actions(action_space)
        self.action_distribution: CategoricalNet = CategoricalNet(
            hidden_size, self.dim_actions)
        
        self._hidden_size = hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._rnn_type = rnn_type
        self._n_input_goal = 1
        self._n_prev_action = hidden_size
        
        self.actor = PointNavGoalDistanceActor(
            observation_space=observation_space,
            hidden_size=hidden_size,
            action_space=action_space,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            noise_coefficient=noise_coefficient,
        )
        
        self.critic = PointNavGoalDistanceCritic(
            observation_space=observation_space,
            hidden_size=hidden_size,
            action_space=action_space,
        )
        
        if aux_loss_config is not None:
            assert "AuxL2Loss" in aux_loss_config, f"Only MSE auxiliary loss is supported, got {aux_loss_config}"
            aux_l2_loss = baseline_registry.get_auxiliary_loss("AuxL2Loss")
            
            self.aux_loss_modules = nn.ModuleDict(
                {
                    "AuxL2Loss": aux_l2_loss(),
                }
            )
        else:
            self.aux_loss_modules = nn.ModuleDict()
        
    
    @property
    def hidden_state_shape(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        return (self.num_recurrent_layers, self.recurrent_hidden_size)

    @property
    def hidden_state_shape_lens(self):
        """
        Stack the hidden states of all the policies in the active population.
        """
        return [self.recurrent_hidden_size]
        
    @property
    def num_recurrent_layers(self) -> int:
        return self._num_rnn_layers

    @property
    def recurrent_hidden_size(self) -> int:
        return self._hidden_size
    
    @property
    def should_load_agent_state(self):
        return False
    
    def forward(self, *x):
        raise NotImplementedError("Forward should be implemented in the child class.")
    
    def _get_policy_components(self) -> List[nn.Module]:
        return [self.actor, self.critic, self.action_distribution]
    
    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        """
        Gets parameters of auxiliary modules, not directly used in the policy,
        but used for auxiliary training objectives. Only necessary if using
        auxiliary losses.
        """
        return {
            name: aux_loss.parameters()
            for name, aux_loss in self.aux_loss_modules.items()
        }
    
    @g_timer.avg_time("net_policy.get_value", level=1)
    def get_value(
        self, observations, rnn_hidden_states, prev_actions, masks
    ) -> torch.Tensor:
        return self.critic(observations, prev_actions, masks)[0]
        

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Only necessary to implement if performing RL training with the policy.

        :returns: Tuple containing
            - Predicted value.
            - Log probabilities of actions.
            - Action distribution entropy.
            - RNN hidden states.
            - Auxiliary module losses.
        """

        features, rnn_hidden_states, aux_loss_state_actor = self.actor(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
        )
        distribution = self.action_distribution(features)
        
        value, aux_loss_state_critic = self.critic(observations, prev_actions, masks)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        
        aux_loss_res = {
            k: dict(loss=v(aux_loss_state_actor, aux_loss_state_critic))
            for k, v in self.aux_loss_modules.items()
        }

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ) -> PolicyActionData:
        features, rnn_hidden_states, _ = self.actor(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value, _ = self.critic(observations, prev_actions, masks)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        return PolicyActionData(
            values=value,
            actions=action,
            action_log_probs=action_log_probs,
            rnn_hidden_states=rnn_hidden_states,
        )
        
    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        policy_config = config.habitat_baselines.rl.policy[agent_name]

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            num_rnn_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            noise_coefficient=policy_config.noise_coefficient,
            policy_config=policy_config,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
        )
        

class PointNavGoalDistanceActor(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        action_space: spaces.Discrete,
        num_rnn_layers: int,
        rnn_type: str,
        noise_coefficient: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.prev_action_embedding = nn.Embedding(
            get_num_actions(action_space) + 1, hidden_size
        )
        self.prev_action_embedding.weight.data.uniform_(-3e-3, 3e-3)
        
        self._hidden_size = hidden_size
        self._num_rnn_layers = num_rnn_layers
        self._rnn_type = rnn_type
        self._n_input_goal = 3
        self._n_prev_action = hidden_size
        self.noise_coefficient = noise_coefficient
        
        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
        
        self.tgt_embedding = nn.Linear(self._n_input_goal, self._hidden_size)
        
        self.state_encoder = build_rnn_state_encoder(
            self._hidden_size + self._n_prev_action,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=self._num_rnn_layers,
        )
        
        self.head = nn.Linear(self._hidden_size, hidden_size)
        self.gelu = nn.GELU()

        self.train()
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        
        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations
        goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]

        if goal_observations.shape[1] == 2:
            goal_observations = torch.stack(
                    [
                        torch.clamp(self._goal_dist_noise(goal_observations[:, 0], coeff=self.noise_coefficient * 10), 0.0, None),
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            goal_observations[:, 1] = torch.clamp(self._goal_dist_noise(goal_observations[:, 1], coeff=self.noise_coefficient), -1, 1)
            goal_observations[:, 2] = torch.clamp(self._goal_dist_noise(goal_observations[:, 2], coeff=self.noise_coefficient), -1, 1)
        else:
            raise ValueError("Only support 2D goal encoding", goal_observations.shape)
        
        tgt_encoding = self.tgt_embedding(goal_observations)
        x.append(tgt_encoding)
        
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # The mask means the previous action will be zero, an extra dummy action
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        x.append(prev_actions)
                
        latent = torch.cat(x, dim=1)
        latent, rnn_hidden_states = self.state_encoder(latent, rnn_hidden_states, masks, rnn_build_seq_info)
        
        out = self.head(self.gelu(latent))
        return out, rnn_hidden_states, latent
    
    def _goal_dist_noise(self, goal_dist):
        return goal_dist + torch.randn_like(goal_dist) * self.noise_coefficient
    
    def _goal_dist_noise(self, goal_dist, coeff):
        return goal_dist + torch.randn_like(goal_dist) * coeff
    

class PointNavGoalDistanceCritic(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        action_space: spaces.Discrete,
    ) -> None:
        super().__init__()
        
        self.prev_action_embedding = nn.Embedding(
            get_num_actions(action_space) + 1, hidden_size
        )
        self.prev_action_embedding.weight.data.uniform_(-3e-3, 3e-3)
        
        self._hidden_size = hidden_size
        self._n_input_goal = 3
        self._n_prev_action = hidden_size  
        
        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
        
        self.tgt_embedding = nn.Linear(self._n_input_goal, self._hidden_size)
        self.feature_mixing = nn.Linear(2 * self._hidden_size, self._hidden_size)
        
        self.fc = nn.Linear(self._hidden_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
        self.gelu = nn.GELU()
        
        self.train()
        
    def forward(self, observations : Dict[str, torch.Tensor], prev_actions, masks):
        x = []
        
        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations
        goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]

        if goal_observations.shape[1] == 2:
            goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
        else:
            raise ValueError("Only support 2D goal encoding:", goal_observations.shape)
        
        tgt_encoding = self.tgt_embedding(goal_observations)
        x.append(tgt_encoding)
        
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # The mask means the previous action will be zero, an extra dummy action
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        
        x.append(prev_actions)
        
        latent = torch.cat(x, dim=1)
        latent = self.gelu(self.feature_mixing(latent))
        
        return self.fc(latent), latent
        