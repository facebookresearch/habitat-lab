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
    GeometricOverlapSensor,
    GeometricOverlapSeedSensor,
    PointGoalSensor,
    ProximitySensor,
)
from rl_distance_train.models import DistanceEstimator
from typing import Union, List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from torchvision import transforms


@baseline_registry.register_policy
class GeometricOverlapNavPolicy(NetPolicy):
    """
    Policy using a ground truth geometric distance and RNN state encoder.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        use_vision=False,
        backbone='efficientnet-b0',
        use_pretrained_encoder=False,
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
        net = GeometricOverlapDistanceNavNet(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            use_vision=use_vision,
            backbone=backbone,
            use_pretrained_encoder=use_pretrained_encoder,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            use_confidence=use_confidence)

        super().__init__(net, action_space, policy_config=policy_config)

        # self.load_state_dict(torch.load('/cluster/home/lmilikic/rl_distance_nav/habitat-lab/data/overlap_dist_vision_pre_noise_conf_checkpoints/ckpt.21.pth', weights_only=False)["state_dict"])

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
            use_vision=getattr(ddppo_cfg, "use_vision", False),
            backbone=getattr(ddppo_cfg, "backbone", 'efficientnet-b0'),
            use_pretrained_encoder=getattr(ddppo_cfg, "use_pretrained_encoder", False),
            num_recurrent_layers=ddppo_cfg.num_recurrent_layers,
            rnn_type=ddppo_cfg.rnn_type,
            use_confidence=getattr(ddppo_cfg, "use_confidence", False),
            policy_config=config.habitat_baselines.rl.policy[agent_name],
        )


class GeometricOverlapDistanceNavNet(Net):
    """
    Net combining TemporalDistanceEncoder with other sensor embeddings into an RNN.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        use_vision,
        backbone,
        use_pretrained_encoder,
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
        assert GeometricOverlapSensor.cls_uuid in observation_space.spaces
        self.dist_estimator = DistanceEstimator(in_dim=13, hidden=[256, 128, 64], 
                                                dropout=0.0, n_dist=50, n_conf=20)
        self.dist_estimator.load_state_dict(
            torch.load("/cluster/home/lmilikic/rl_distance_nav/habitat-lab/rl_distance_train/models/distance_estimator/model.pt", weights_only=False)['model'])
        for param in self.dist_estimator.parameters():
            param.requires_grad = False

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

        # 3) Visual Encoder
        self.use_pretrained_encoder = use_pretrained_encoder
        self.use_vision = use_vision
        if self.use_vision:
            assert "rgb" in observation_space.spaces
            assert backbone.startswith('efficientnet')
    
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            if self.use_pretrained_encoder:
                self.visual_encoder = EfficientNet.from_name(backbone, in_channels=3, num_classes=self._hidden_size)
                self.visual_encoder.load_state_dict(
                    torch.load('/cluster/home/lmilikic/rl_distance_nav/habitat-lab/rl_distance_train/models/visual_encoder/model.pth', weights_only=False))

                # Freeze all backbone layers
                for param in self.visual_encoder.parameters():
                    param.requires_grad = False
            else:
                self.visual_encoder = EfficientNet.from_name(backbone, in_channels=3, num_classes=self._hidden_size)

            rnn_input_size += self._hidden_size

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

        if self.use_vision:
            rgb = observations['rgb']
            if rgb.shape[-1] == 3:
                if rgb.ndim == 4:
                    rgb = rgb.permute(0, 3, 1, 2) # → [B, 3, H, W]
                elif rgb.ndim == 3:
                    rgb = rgb.permute(2, 0, 1) # → [3, H, W]

            rgb = rgb.div(255.0)
            rgb = self.transform(rgb)
            rgb_encoded = self.visual_encoder(rgb).contiguous()
            features.append(rgb_encoded)

        assert GeometricOverlapSensor.cls_uuid in observations
        o = observations[GeometricOverlapSensor.cls_uuid]
        if GeometricOverlapSeedSensor.cls_uuid in observations:
            seeds = observations[GeometricOverlapSeedSensor.cls_uuid].int().squeeze().tolist()
            if not isinstance(seeds, list):
                seeds = [seeds]
        else:
            seeds = None

        with torch.no_grad():
            o = self.dist_estimator(o, sample=True, seeds=seeds)

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
        h_in = torch.cat(features, dim=1).contiguous()
        out, h_out = self.state_encoder(h_in, rnn_hidden_states, masks, rnn_build_seq_info)
        aux_loss_state["rnn_output"] = h_in

        return out, h_out, aux_loss_state
