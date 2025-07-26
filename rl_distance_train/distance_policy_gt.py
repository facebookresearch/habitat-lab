import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from habitat_baselines.rl.ppo import Net, Policy, NetPolicy
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from rl_distance_train.models import TemporalDistanceEncoder
from typing import Union, List, Dict, Optional, Tuple

@baseline_registry.register_policy
class GeometricNavPolicy(NetPolicy):
    """
    Policy using a ground truth geometric distance and RNN state encoder.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        use_confidence=False,
        policy_config: "DictConfig" = None,
        **kwargs
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        # build network and pass to PPO Policy
        net = GeometricDistanceNavNet(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            use_confidence=use_confidence)

        super().__init__(net, action_space, policy_config=policy_config)

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
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


        ppo_cfg = config.habitat_baselines.rl.ppo
        ddppo_cfg = config.habitat_baselines.rl.ddppo
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=ppo_cfg.hidden_size,
            num_recurrent_layers=ddppo_cfg.num_recurrent_layers,
            rnn_type=ddppo_cfg.rnn_type,
            use_confidence=getattr(config, "use_confidence", False),
            policy_config=config.habitat_baselines.rl.policy[agent_name],
        )


class GeometricDistanceNavNet(Net):
    """
    Net combining TemporalDistanceEncoder with other sensor embeddings into an RNN.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        num_recurrent_layers,
        rnn_type,
        use_confidence,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self.use_confidence = use_confidence

        # 1) Previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        prev_emb_size = 32

        # 2) Sensor embeddings
        rnn_input_size = prev_emb_size
        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
        if self.use_confidence:
            self.tgt_emb = nn.Linear(2, 32)
        else:
            self.tgt_emb = nn.Linear(1, 32)
        rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[0]
            self.gps_emb = nn.Linear(gps_dim, 32)
            rnn_input_size += 32
        if HeadingSensor.cls_uuid in observation_space.spaces:
            self.heading_emb = nn.Linear(2, 32)
            rnn_input_size += 32
        if ProximitySensor.cls_uuid in observation_space.spaces:
            prox_dim = observation_space.spaces[ProximitySensor.cls_uuid].shape[0]
            self.prox_emb = nn.Linear(prox_dim, 32)
            rnn_input_size += 32
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            self.compass_emb = nn.Linear(2, 32)
            rnn_input_size += 32

        # 3) RNN State Encoder
        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
    
    @property
    def is_blind(self):
        return False

    @property
    def perception_embedding_size(self):
        return self._hidden_size
    
    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    def _format_pose(self, pose_tensor):
        x, y, theta, t = torch.unbind(pose_tensor, dim=1)
        return torch.stack([x, y, torch.cos(theta), torch.sin(theta), torch.exp(-t)], dim=1)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        features = []
        aux_loss_state = {}

        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations
        o = observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
        if self.use_confidence:
            features.append(self.tgt_emb(o[:, :2]))
        else:
            features.append(self.tgt_emb(o[:, :1]))

        # Other sensors
        if EpisodicGPSSensor.cls_uuid in observations:
            features.append(self.gps_emb(observations[EpisodicGPSSensor.cls_uuid]))
        if HeadingSensor.cls_uuid in observations:
            h = observations[HeadingSensor.cls_uuid]
            vec = torch.stack([torch.cos(-h[:, 0]), torch.sin(-h[:, 0])], dim=-1)
            features.append(self.heading_emb(vec))
        if ProximitySensor.cls_uuid in observations:
            features.append(self.prox_emb(observations[ProximitySensor.cls_uuid]))
        if EpisodicCompassSensor.cls_uuid in observations:
            c = observations[EpisodicCompassSensor.cls_uuid]
            vec = torch.stack([torch.cos(c), torch.sin(c)], dim=-1).squeeze(1)
            features.append(self.compass_emb(vec))

        # Prev action
        pa = ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        features.append(self.prev_action_embedding(pa))

        # Concat and RNN
        h_in = torch.cat(features, dim=1)
        out, h_out = self.state_encoder(h_in, rnn_hidden_states, masks, rnn_build_seq_info)
        aux_loss_state["rnn_output"] = h_in

        return out, h_out, aux_loss_state
