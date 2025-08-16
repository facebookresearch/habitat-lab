import torch
import torch.nn as nn

from torchvision.transforms import ColorJitter, RandomResizedCrop, Compose
from typing import Union, List, Dict
from PIL import Image

from vint_based import load_distance_model


class TemporalDistanceEncoder(nn.Module):
    """
    Wraps a pretrained DistanceConfidenceModel as an RL encoder.

    Args:
        encoder_base:  the model ID to pass to load_distance_model(...)
        freeze:        if True, freezes all parameters in the loaded model
        rgb_color_jitter: float amount for ColorJitter (0 ⇒ no jitter)
        random_crop:   if True, applies a RandomResizedCrop augmentation
        mode:          'dense' => [dist; conf; last_hidden]
                       'sparse'=> [dist; conf]
    """
    def __init__(
        self,
        encoder_base: str = "dist_decoder_conf_100max",
        freeze: bool = True,
        rgb_color_jitter: float = 0.,
        random_crop: bool = False,
        mode: str = "dense",
        distance_scale=1.0,
    ):
        super().__init__()
        assert mode in ("dense", "sparse"), "mode must be 'dense' or 'sparse'"
        self.mode = mode
        self.distance_scale = distance_scale

        # 1) load the pretrained Distance+Confidence model
        dm = load_distance_model(modelid=encoder_base)
        # unwrap DataParallel if present
        self.base: nn.Module = getattr(dm, "module", dm)

        # 2) optionally freeze everything
        self.freeze = freeze
        if freeze:
            for p in self.base.parameters():
                p.requires_grad = False

        # 3) grab the embedding size
        #    DistanceConfidenceModel inherits embed_dim from DistanceModel
        self.embed_dim = self.base.embed_dim

        transform_list = []

        # 4) set image augmentations if applicable
        #    Color jitter on RGB channels
        if rgb_color_jitter > 0:
            transform_list.append(ColorJitter(
                brightness=rgb_color_jitter,
                contrast=rgb_color_jitter,
                saturation=rgb_color_jitter,
                hue=rgb_color_jitter
            ))

        #    Random resized crop (applied before jitter), using the model's expected input size if available
        if random_crop:
            # try to infer the target size from the base model, default to 224 if not present
            target_size = getattr(self.base, "img_size", 224)
            transform_list.append(RandomResizedCrop(size=target_size))

        if transform_list:
            self._common_transform = Compose(transform_list)
        else:
            self._common_transform = None

        # 5) determine our output vector length
        if self.mode == "dense":
            # last_hidden (embed_dim) + dist (1) + conf (1)
            self.output_dim = self.embed_dim + 2
        else:
            # just [dist, conf]
            self.output_dim = 2

    @property
    def is_blind(self) -> bool:
        return False

    def forward(
        self,
        observations: Union[Image.Image, List[Image.Image], torch.Tensor],
        goals:        Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns:
          - if `dense`:   Tensor of shape [B, embed_dim+2]
          - if `sparse`:  Tensor of shape [B, 2]
        """
        # apply random crop to both observations and goals if enabled
        if self._common_transform is not None:
            if torch.is_tensor(observations):
                observations = self._common_transform(observations.to(torch.float32).div(255.0)).prod(255) 
                goals = self._common_transform(goals.to(torch.float32).div(255.0)).prod(255)
            else:
                observations = self._common_transform(observations)
                goals = self._common_transform(goals)
                   
        # Let the base model handle preprocessing (list of PIL, single PIL, or tensor)
        observations = self.base.preprocess(observations)
        goals = self.base.preprocess(goals)

        return_last_hidden_state = self.mode == "dense"
        if self.freeze:
            with torch.no_grad():
                output = self.base(observations, goals, return_last_hidden_state=return_last_hidden_state)
        else:
            output = self.base(observations, goals, return_last_hidden_state=return_last_hidden_state)

        # Base model forward pass
        if return_last_hidden_state:
            dist, conf, last_hidden = output
            dist = dist / self.distance_scale
            # dist,conf: [B], last_hidden: [B, embed_dim]
            dist = dist.unsqueeze(1)  # → [B,1]
            conf = conf.unsqueeze(1)  # → [B,1]
            out = torch.cat([last_hidden, dist, conf], dim=1)
        else:
            dist, conf = output
            dist = dist / self.distance_scale
            out = torch.cat([
                dist.unsqueeze(1),  # [B,1]
                conf.unsqueeze(1)   # [B,1]
            ], dim=1)

        return out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.trunk = nn.Sequential(*layers)

    def forward(self, x):
        return self.trunk(x)

class DistanceEstimator(nn.Module):
    """
    MLP trunk + two classification heads:
      - dist_head -> n_dist_bins
      - conf_head -> n_conf_bins
    """
    def __init__(self, in_dim: int, hidden: List[int], dropout: float, n_dist: int, n_conf: int):
        super().__init__()
        self.backbone = MLPHead(in_dim, hidden, dropout)
        last_dim = hidden[-1] if len(hidden) > 0 else in_dim
        self.dist_head = nn.Linear(last_dim, n_dist)
        self.conf_head = nn.Linear(last_dim, n_conf)

    def forward(self, x, sample=False):
        z = self.backbone(x)
        dist_logits = self.dist_head(z)
        conf_logits = self.conf_head(z)

        if not sample:
            return dist_logits, conf_logits

        probs_dist = torch.softmax(dist_logits, dim=-1)
        probs_conf = torch.softmax(conf_logits, dim=-1)

        dist = self.sample_from_bins(probs_dist)
        conf = self.sample_from_bins(probs_conf)

        out = torch.cat([
            dist.unsqueeze(1),  # [B,1]
            conf.unsqueeze(1)   # [B,1]
        ], dim=1)

        return out

    @staticmethod
    def sample_from_bins(probs: torch.Tensor) -> torch.Tensor:
        """
        probs: torch tensor [B, n_bins], each row sums to 1
        returns: torch tensor [B] with sampled values in [0, 1]
        """
        B, n_bins = probs.shape
        device = probs.device
        dtype = probs.dtype

        # Bin edges in normalized [0, 1]
        bin_edges = torch.linspace(0, 1, n_bins + 1, device=device, dtype=dtype)

        # Sample bin indices from probs
        bin_indices = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)

        # Get bin low/high edges
        lows = bin_edges[bin_indices]
        highs = bin_edges[bin_indices + 1]

        # Gaussian sample inside chosen bin
        lows = bin_edges[bin_indices]
        highs = bin_edges[bin_indices + 1]
        mid = (lows + highs) / 2
        std = (highs - lows) / 6  # ~99.7% of values within [lows, highs]

        return np.random.normal(mid, std)