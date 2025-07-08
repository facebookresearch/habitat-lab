import torch
import torch.nn as nn

from torchvision.transforms import ColorJitter, RandomResizedCrop, Compose
from typing import Union, List
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
    ):
        super().__init__()
        assert mode in ("dense", "sparse"), "mode must be 'dense' or 'sparse'"
        self.mode = mode

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
            # dist,conf: [B], last_hidden: [B, embed_dim]
            dist = dist.unsqueeze(1)  # → [B,1]
            conf = conf.unsqueeze(1)  # → [B,1]
            out = torch.cat([last_hidden, dist, conf], dim=1)
        else:
            dist, conf = output
            out = torch.cat([
                dist.unsqueeze(1),  # [B,1]
                conf.unsqueeze(1)   # [B,1]
            ], dim=1)

        return out
