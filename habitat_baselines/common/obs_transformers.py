#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import copy
import numbers
from typing import List

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
    def __init__(self, *args, **kwargs):
        super().__init__()

    def transform_observation_space(
        self, observation_space: SpaceDict, **kwargs
    ):
        return observation_space

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, envs):
        pass

    def forward(self, *args):
        return args


@baseline_registry.register_obs_transformer()
class ResizeShortestEdge(ObservationTransformer):
    def __init__(self, size: int, channels_last: bool = False):
        r"""An nn module the resizes your the shortest edge of the input while maintaining aspect ratio.
        This module assumes that all images in the batch are of the same size.
        Args:
            size: The size you want to resize the shortest edge to
            channels_list: indicates if channels is the last dimension
        """
        super().__init__()
        self._size = size
        self.channels_last = channels_last

    def transform_observation_space(
        self,
        observation_space: SpaceDict,
        trans_keys=("rgb", "depth", "semantic"),
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in trans_keys:
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
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return image_resize_shortest_edge(
            input, self._size, channels_last=self.channels_last
        )

    @classmethod
    def from_config(cls, config, envs):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE)


@baseline_registry.register_obs_transformer()
class CenterCropper(ObservationTransformer):
    def __init__(self, size, channels_last: bool = False):
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

    def transform_observation_space(
        self,
        observation_space: SpaceDict,
        trans_keys=("rgb", "depth", "semantic"),
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in trans_keys
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
        self.observation_space = observation_space
        return observation_space

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return input

        return center_crop(
            input,
            self._size,
            channels_last=self.channels_last,
        )

    @classmethod
    def from_config(cls, config, envs):
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
