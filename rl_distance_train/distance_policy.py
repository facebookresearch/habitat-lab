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
class TemporalDistanceNavPolicy(NetPolicy):
    """
    Policy using a temporal-distance image-goal encoder and RNN state encoder.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        random_crop=False,
        rgb_color_jitter=0.0,
        encoder_base="dist_decoder_conf_100max",
        freeze_encoder=True,
        distance_scale=1.0,
        use_confidence=False,
        pretrained_weights=None,
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
        net = TemporalDistanceNavNet(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            random_crop=random_crop,
            rgb_color_jitter=rgb_color_jitter,
            encoder_base=encoder_base,
            freeze_encoder=freeze_encoder,
            distance_scale=distance_scale,
            use_confidence=use_confidence)

        super().__init__(net, action_space, policy_config=policy_config)

        if pretrained_weights is not None:
            try:
                print(f"Loading pretrained weights from: {pretrained_weights}")
                checkpoint = torch.load(pretrained_weights, map_location="cpu")
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                load_res = self.load_state_dict(checkpoint, strict=False)
                print("== Pretrained weights loaded (non-strict) ==")
                print("Missing keys:", {k for k in load_res.missing_keys if not k.startswith('net.visual_encoder')})
                print("Unexpected keys:", load_res.unexpected_keys)
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")

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
            random_crop=getattr(ppo_cfg, 'random_crop', False),
            rgb_color_jitter=getattr(ppo_cfg, 'rgb_color_jitter', 0.0),
            encoder_base=getattr(ddppo_cfg, 'encoder_backbone', 'dist_decoder_conf_100max'),
            freeze_encoder=getattr(ddppo_cfg, 'freeze_encoder', True),
            distance_scale=getattr(ddppo_cfg, 'distance_scale', 1.0),
            use_confidence=getattr(ddppo_cfg, "use_confidence", False),
            pretrained_weights=getattr(ddppo_cfg, "pretrained_weights", None),
            policy_config=config.habitat_baselines.rl.policy[agent_name],
        )


class TemporalDistanceNavNet(Net):
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
        random_crop,
        rgb_color_jitter,
        encoder_base,
        freeze_encoder,
        distance_scale,
        use_confidence,
    ):
        super().__init__()
        self._hidden_size = hidden_size
        self.use_confidence = use_confidence

        # 1) Previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        prev_emb_size = 32
        rnn_input_size = prev_emb_size

        # 2) Joint image-goal encoder
        self.visual_encoder = TemporalDistanceEncoder(
            encoder_base=encoder_base,
            freeze=freeze_encoder,
            random_crop=random_crop,
            rgb_color_jitter=rgb_color_jitter,
            mode='sparse',
            distance_scale=distance_scale,
        )

        if self.use_confidence:
            self.tgt_emb = nn.Linear(2, 32)
        else:
            self.tgt_emb = nn.Linear(1, 32)
        rnn_input_size += 32

        # 3) Sensor embeddings
        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[0]
            self.gps_emb = nn.Linear(gps_dim, 32)
            rnn_input_size += 32
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            self.compass_emb = nn.Linear(2, 32)
            rnn_input_size += 32

        # 4) RNN State Encoder
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
        # Joint image-goal
        rgb = observations.get('rgb', None)
        goal_img = observations.get(ImageGoalSensor.cls_uuid, None)
        assert rgb is not None and goal_img is not None, \
            "Missing 'rgb' or goal image in observations"
        
        dist = self.visual_encoder(rgb, goal_img)
        if self.use_confidence:
            features.append(self.tgt_emb(dist[:, :2]))
        else:
            features.append(self.tgt_emb(dist[:, :1]))

        # Other sensors
        if EpisodicGPSSensor.cls_uuid in observations:
            features.append(self.gps_emb(observations[EpisodicGPSSensor.cls_uuid]))
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
