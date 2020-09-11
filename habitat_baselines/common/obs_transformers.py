#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import copy
import numbers
from typing import Dict, List, Tuple

import attr
import torch
from gym.spaces.dict_space import Dict as SpaceDict
from torch import nn

from habitat.config import Config
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.utils import (
    center_crop,
    get_image_height_width,
    image_resize_shortest_edge,
    overwrite_gym_box_shape,
)


class ObservationTransformer(nn.Module, metaclass=abc.ABCMeta):
    def transform_observation_space(
        self, observation_space: SpaceDict, **kwargs
    ):
        return observation_space

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config, envs):
        pass

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return observations


@baseline_registry.register_obs_transformer()
@attr.s(auto_attribs=True)
class ResizeShortestEdge(ObservationTransformer):
    r"""An nn module the resizes your the shortest edge of the input while maintaining aspect ratio.
    This module assumes that all images in the batch are of the same size.
    Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension
    """
    size: int
    channels_last: bool = False
    trans_keys: Tuple[str] = ("rgb", "depth", "semantic")

    def transform_observation_space(
        self,
        observation_space: SpaceDict,
    ):
        size = self.size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in self.trans_keys:
                    # In the observation space dict, the channels are always last
                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    if size == min(h, w):
                        continue
                    scale = size / min(h, w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    new_size = (new_h, new_w)
                    logger.info(
                        "Resizing observation of %s: from %s to %s"
                        % (key, (h, w), new_size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], new_size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return image_resize_shortest_edge(
            obs, self._size, channels_last=self.channels_last
        )

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self._trans_keys
                }
            )
        return observations

    @classmethod
    def from_config(cls, config, envs):
        return cls(
            size=config.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE
        )


@baseline_registry.register_obs_transformer()
class CenterCropper(ObservationTransformer):
    def __init__(
        self,
        size,
        channels_last: bool = False,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        r"""An nn module that center crops your input.
        Args:
            size: A sequence (h, w) or int of the size you wish to resize/center_crop.
                    If int, assumes square crop
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (h, w)"
        self._size = size
        self.channels_last = channels_last
        self.trans_keys = trans_keys

    def transform_observation_space(
        self,
        observation_space: SpaceDict,
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in self.trans_keys
                    and observation_space.spaces[key].shape[-3:-1] != size
                ):
                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    logger.info(
                        "Center cropping observation size of %s from %s to %s"
                        % (key, (h, w), size)
                    )

                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return center_crop(
            obs,
            self._size,
            channels_last=self.channels_last,
        )

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is None:
            return observations
        return {
            sensor: self._transform_obs(observations[sensor])
            for sensor in self.trans_keys
        }

    @classmethod
    def from_config(cls, config: Config, envs):
        return cls(
            (
                config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER.HEIGHT,
                config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER.WIDTH,
            )
        )


def get_active_obs_transforms(
    config: Config, envs
) -> List[ObservationTransformer]:
    active_obs_transforms = []
    if hasattr(config.RL.POLICY, "OBS_TRANSFORMS"):
        obs_transform_names = config.RL.POLICY.OBS_TRANSFORMS.active
        for obs_transform_name in obs_transform_names:
            obs_trans_cls = baseline_registry.get_obs_transformer(
                obs_transform_name
            )
            obs_transform = obs_trans_cls.from_config(config, envs)
            active_obs_transforms.append(obs_transform)
    return active_obs_transforms
