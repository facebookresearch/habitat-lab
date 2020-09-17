#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the

# LICENSE file in the root directory of this source tree.

r"""This module defines various ObservationTransformers that can be used
to transform the output of the simulator before they are fed into the
policy of the neural network. This can include various useful preprocessing
including faking a semantic sensor using RGB input and MaskRCNN or faking
a depth sensor using RGB input. You can also stich together multiple sensors.
This code runs on the batched of inputs to these networks efficiently.
ObservationTransformer all run as nn.modules and can be used for encoders or
any other neural networks preprocessing steps.
Assumes the input is on CUDA.

They also implement a function that transforms that observation space so help
fake or modify sensor input from the simulator.

This module API is experimental and likely to change
"""
import abc
import copy
import math
import numbers
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from gym.spaces.dict_space import Dict as SpaceDict
from torch import nn

from habitat.config import Config
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    image_resize_shortest_edge,
    overwrite_gym_box_shape,
)


class ObservationTransformer(nn.Module, metaclass=abc.ABCMeta):
    """This is the base ObservationTransformer class that all other observation
    Transformers should extend. from_config must be implemented by the transformer.
    transform_observation_space is only needed if the observation_space ie.
    (resolution, range, or num of channels change)."""

    def transform_observation_space(
        self, observation_space: SpaceDict, **kwargs
    ):
        return observation_space

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return observations


@baseline_registry.register_obs_transformer()
class ResizeShortestEdge(ObservationTransformer):
    r"""An nn module the resizes your the shortest edge of the input while maintaining aspect ratio.
    This module assumes that all images in the batch are of the same size.
    """

    def __init__(
        self,
        size: int,
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        """Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension
        """
        super(ResizeShortestEdge, self).__init__()
        self._size: int = size
        self.channels_last: bool = channels_last
        self.trans_keys: Tuple[str] = trans_keys

    def transform_observation_space(
        self,
        observation_space: SpaceDict,
    ):
        size = self._size
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

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE)


@baseline_registry.register_obs_transformer()
class CenterCropper(ObservationTransformer):
    """An observation transformer is a simple nn module that center crops your input."""

    def __init__(
        self,
        size: Union[int, Tuple[int]],
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):
        """Args:
        size: A sequence (h, w) or int of the size you wish to resize/center_crop.
                If int, assumes square crop
        channels_list: indicates if channels is the last dimension
        trans_keys: The list of sensors it will try to centercrop.
        """
        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (h, w)"
        self._size = size
        self.channels_last = channels_last
        self.trans_keys = trans_keys  # TODO: Add to from_config constructor

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

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER
        return cls(
            (
                cc_config.HEIGHT,
                cc_config.WIDTH,
            )
        )


class Cube2Equirec(nn.Module):
    """This is the backend Cube2Equirec nn.module that does the stiching.
    Inspired from https://github.com/fuenwang/PanoramaUtility and
    optimized for modern PyTorch."""

    def __init__(
        self, equ_h: int, equ_w: int, cube_length: int, fov: int = 90
    ):
        """Args:
        equ_h: (int) the height of the generated equirect
        equ_w: (int) the width of the generated equirect
        cube_length: (int) the length of each side of the cubemap
        fov: (int) the FOV of each camera making the cubemap
        """
        super(Cube2Equirec, self).__init__()
        self.cube_h = cube_length
        self.cube_w = cube_length
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.fov = fov
        self.fov_rad = self.fov * np.pi / 180

        # Compute the parameters for projection
        assert self.cube_w == self.cube_h
        self.radius = int(0.5 * cube_length)

        # Map equirectangular pixel to longitude and latitude
        # NOTE: Make end a full length since arange have a right open bound [a, b)
        theta_start = math.pi - (math.pi / equ_w)
        theta_end = -math.pi
        theta_step = 2 * math.pi / equ_w
        theta_range = torch.arange(theta_start, theta_end, -theta_step)

        phi_start = 0.5 * math.pi - (0.5 * math.pi / equ_h)
        phi_end = -0.5 * math.pi
        phi_step = math.pi / equ_h
        phi_range = torch.arange(phi_start, phi_end, -phi_step)

        # Stack to get the longitude latitude map
        self.theta_map = theta_range.unsqueeze(0).repeat(equ_h, 1)
        self.phi_map = phi_range.unsqueeze(-1).repeat(1, equ_w)
        self.lonlat_map = torch.stack([self.theta_map, self.phi_map], dim=-1)

        # Get mapping relation (h, w, face) (orientation map)
        # [back, down, front, left, right, up] => [0, 1, 2, 3, 4, 5]

        # Project each face to 3D cube and convert to pixel coordinates
        self.grid, self.orientation_mask = self.get_grid2()

    def get_grid2(self):
        # Get the point of equirectangular on 3D ball
        x_3d = (
            self.radius * torch.cos(self.phi_map) * torch.sin(self.theta_map)
        ).view(self.equ_h, self.equ_w, 1)
        y_3d = (self.radius * torch.sin(self.phi_map)).view(
            self.equ_h, self.equ_w, 1
        )
        z_3d = (
            self.radius * torch.cos(self.phi_map) * torch.cos(self.theta_map)
        ).view(self.equ_h, self.equ_w, 1)

        self.grid_ball = torch.cat([x_3d, y_3d, z_3d], 2).view(
            self.equ_h, self.equ_w, 3
        )

        # Compute the down grid
        radius_ratio_down = torch.abs(y_3d / self.radius)
        grid_down_raw = self.grid_ball / radius_ratio_down.view(
            self.equ_h, self.equ_w, 1
        ).expand(-1, -1, 3)
        grid_down_w = (
            -grid_down_raw[:, :, 0].clone() / self.radius
        ).unsqueeze(-1)
        grid_down_h = (
            -grid_down_raw[:, :, 2].clone() / self.radius
        ).unsqueeze(-1)
        grid_down = torch.cat([grid_down_w, grid_down_h], 2).unsqueeze(0)
        mask_down = (
            ((grid_down_w <= 1) * (grid_down_w >= -1))
            * ((grid_down_h <= 1) * (grid_down_h >= -1))
            * (grid_down_raw[:, :, 1] == -self.radius).unsqueeze(2)
        ).float()

        # Compute the up grid
        radius_ratio_up = torch.abs(y_3d / self.radius)
        grid_up_raw = self.grid_ball / radius_ratio_up.view(
            self.equ_h, self.equ_w, 1
        ).expand(-1, -1, 3)
        grid_up_w = (-grid_up_raw[:, :, 0].clone() / self.radius).unsqueeze(-1)
        grid_up_h = (grid_up_raw[:, :, 2].clone() / self.radius).unsqueeze(-1)
        grid_up = torch.cat([grid_up_w, grid_up_h], 2).unsqueeze(0)
        mask_up = (
            ((grid_up_w <= 1) * (grid_up_w >= -1))
            * ((grid_up_h <= 1) * (grid_up_h >= -1))
            * (grid_up_raw[:, :, 1] == self.radius).unsqueeze(2)
        ).float()

        # Compute the front grid
        radius_ratio_front = torch.abs(z_3d / self.radius)
        grid_front_raw = self.grid_ball / radius_ratio_front.view(
            self.equ_h, self.equ_w, 1
        ).expand(-1, -1, 3)
        grid_front_w = (
            -grid_front_raw[:, :, 0].clone() / self.radius
        ).unsqueeze(-1)
        grid_front_h = (
            -grid_front_raw[:, :, 1].clone() / self.radius
        ).unsqueeze(-1)
        grid_front = torch.cat([grid_front_w, grid_front_h], 2).unsqueeze(0)
        mask_front = (
            ((grid_front_w <= 1) * (grid_front_w >= -1))
            * ((grid_front_h <= 1) * (grid_front_h >= -1))
            * (torch.round(grid_front_raw[:, :, 2]) == self.radius).unsqueeze(
                2
            )
        ).float()

        # Compute the back grid
        radius_ratio_back = torch.abs(z_3d / self.radius)
        grid_back_raw = self.grid_ball / radius_ratio_back.view(
            self.equ_h, self.equ_w, 1
        ).expand(-1, -1, 3)
        grid_back_w = (grid_back_raw[:, :, 0].clone() / self.radius).unsqueeze(
            -1
        )
        grid_back_h = (
            -grid_back_raw[:, :, 1].clone() / self.radius
        ).unsqueeze(-1)
        grid_back = torch.cat([grid_back_w, grid_back_h], 2).unsqueeze(0)
        mask_back = (
            ((grid_back_w <= 1) * (grid_back_w >= -1))
            * ((grid_back_h <= 1) * (grid_back_h >= -1))
            * (torch.round(grid_back_raw[:, :, 2]) == -self.radius).unsqueeze(
                2
            )
        ).float()

        # Compute the right grid
        radius_ratio_right = torch.abs(x_3d / self.radius)
        grid_right_raw = self.grid_ball / radius_ratio_right.view(
            self.equ_h, self.equ_w, 1
        ).expand(-1, -1, 3)
        grid_right_w = (
            -grid_right_raw[:, :, 2].clone() / self.radius
        ).unsqueeze(-1)
        grid_right_h = (
            -grid_right_raw[:, :, 1].clone() / self.radius
        ).unsqueeze(-1)
        grid_right = torch.cat([grid_right_w, grid_right_h], 2).unsqueeze(0)
        mask_right = (
            ((grid_right_w <= 1) * (grid_right_w >= -1))
            * ((grid_right_h <= 1) * (grid_right_h >= -1))
            * (torch.round(grid_right_raw[:, :, 0]) == -self.radius).unsqueeze(
                2
            )
        ).float()

        # Compute the left grid
        radius_ratio_left = torch.abs(x_3d / self.radius)
        grid_left_raw = self.grid_ball / radius_ratio_left.view(
            self.equ_h, self.equ_w, 1
        ).expand(-1, -1, 3)
        grid_left_w = (grid_left_raw[:, :, 2].clone() / self.radius).unsqueeze(
            -1
        )
        grid_left_h = (
            -grid_left_raw[:, :, 1].clone() / self.radius
        ).unsqueeze(-1)
        grid_left = torch.cat([grid_left_w, grid_left_h], 2).unsqueeze(0)
        mask_left = (
            ((grid_left_w <= 1) * (grid_left_w >= -1))
            * ((grid_left_h <= 1) * (grid_left_h >= -1))
            * (torch.round(grid_left_raw[:, :, 0]) == self.radius).unsqueeze(2)
        ).float()

        # Face map contains numbers correspond to that face
        orientation_mask = (
            mask_back * 0
            + mask_down * 1
            + mask_front * 2
            + mask_left * 3
            + mask_right * 4
            + mask_up * 5
        )

        return (
            torch.cat(
                [
                    grid_back,
                    grid_down,
                    grid_front,
                    grid_left,
                    grid_right,
                    grid_up,
                ],
                0,
            ),
            orientation_mask,
        )

    # Convert cubic images to equirectangular
    def _to_equirec(self, batch: torch.Tensor):
        batch_size, ch, _H, _W = batch.shape
        if batch_size != 6:
            raise ValueError("Batch size mismatch!!")

        output = torch.zeros(
            1, ch, self.equ_h, self.equ_w, device=batch.device
        )

        for ori in range(6):
            grid = self.grid[ori, :, :, :].unsqueeze(
                0
            )  # 1, self.equ_h, self.equ_w, 2
            mask = (self.orientation_mask == ori).unsqueeze(
                0
            )  # 1, self.equ_h, self.equ_w, 1

            masked_grid = grid * mask.float().expand(
                -1, -1, -1, 2
            )  # 1, self.equ_h, self.equ_w, 2

            source_image = batch[ori].unsqueeze(0)  # 1, ch, H, W

            sampled_image = torch.nn.functional.grid_sample(
                source_image,
                masked_grid,
                align_corners=False,
                padding_mode="border",
            )  # 1, ch, self.equ_h, self.equ_w

            sampled_image_masked = sampled_image * (
                mask.float()
                .view(1, 1, self.equ_h, self.equ_w)
                .expand(1, ch, -1, -1)
            )
            output = (
                output + sampled_image_masked
            )  # 1, ch, self.equ_h, self.equ_w

        return output

    # Convert input cubic tensor to output equirectangular image
    def to_equirec_tensor(self, batch: torch.Tensor):
        # Move the params to the right device. NOOP after first call
        self.grid = self.grid.to(batch.device)
        self.orientation_mask = self.orientation_mask.to(batch.device)
        # Check whether batch size is 6x
        batch_size = batch.size()[0]
        if batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")

        processed = []
        for idx in range(int(batch_size / 6)):
            target = batch[idx * 6 : (idx + 1) * 6, :, :, :]
            target_processed = self._to_equirec(target)
            processed.append(target_processed)

        output = torch.cat(processed, 0)
        return output

    def forward(self, batch: torch.Tensor):
        return self.to_equirec_tensor(batch)


@baseline_registry.register_obs_transformer()
class CubeMap2Equirec(ObservationTransformer):
    r"""This is an experimental use of ObservationTransformer that converts a cubemap
    output to an equirectangular one through projection. This needs to be fed
    a list of 6 cameras at various orientations but will be able to stitch a
    360 sensor out of these inputs. The code below will generate a config that
    has the 6 sensors in the proper orientations. This code also assumes a 90
    FOV.

    Sensor order for cubemap stiching is Back, Down, Front, Left, Right, Up.
    The output will be writen the UUID of the first sensor.
    """

    def __init__(
        self,
        sensor_uuids: List[str],
        eq_shape: Tuple[int],
        cubemap_length: int,
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
    ):
        r""":param sensor: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param eq_shape: The shape of the equirectangular output (height, width)
        :param cubemap_length: int length of the each side of the cubemap
        """
        super(CubeMap2Equirec, self).__init__()
        num_sensors = len(sensor_uuids)
        assert (
            num_sensors % 6 == 0 and num_sensors != 0
        ), f"{len(sensor_uuids)}: length of sensors is not a multiple of 6"
        # TODO verify attributes of the sensors in the config if possible. Think about API design
        assert (
            len(eq_shape) == 2
        ), f"eq_shape must be a tuple of (height, width), given: {eq_shape}"
        assert (
            cubemap_length > 0
        ), f"cubemap_length must be greater than 0: provided {cubemap_length}"
        self.sensor_uuids: List[str] = sensor_uuids
        self.eq_shape: Tuple[int] = eq_shape
        self.cubemap_length: int = cubemap_length
        self.channels_last: bool = channels_last
        self.c2eq: nn.Module = Cube2Equirec(
            eq_shape[0], eq_shape[1], cubemap_length
        )
        if target_uuids == None:
            self.target_uuids: List[str] = self.sensor_uuids[::6]
        else:
            self.target_uuids: List[str] = target_uuids
        # TODO support and test different FOVs than just 90

    def transform_observation_space(
        self,
        observation_space: SpaceDict,
    ):
        r"""Transforms the target UUID's sensor obs_space so it matches the new shape (EQ_H, EQ_W)"""
        # Transforms the observation space to of the target UUID
        observation_space = copy.deepcopy(observation_space)
        for i, key in enumerate(self.target_uuids):
            assert (
                key in observation_space.spaces
            ), f"{key} not found in observation space: {observation_space.spaces}"
            c = self.cubemap_length
            logger.info(
                f"Overwrite sensor: {key} from size of ({c}, {c}) to equirect image of {self.eq_shape} from sensors: {self.sensor_uuids[i*6:(i+1)*6]}"
            )
            if (c, c) != self.eq_shape:
                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.eq_shape
                )
        return observation_space

    @classmethod
    def from_config(cls, config):
        cube2eq_config = config.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ
        if hasattr(cube2eq_config, "TARGET_UUIDS"):
            # Optional Config Value to specify target UUID
            target_uuids = cube2eq_config.TARGET_UUIDS
        else:
            target_uuids = None
        return cls(
            cube2eq_config.SENSOR_UUIDS,
            eq_shape=(
                cube2eq_config.HEIGHT,
                cube2eq_config.WIDTH,
            ),
            cubemap_length=cube2eq_config.CUBE_LENGTH,
            target_uuids=target_uuids,
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for i in range(0, len(self.target_uuids), 6):
            # The UUID we are overwriting
            target_sensor_uuid = self.target_uuids[i // 6]
            assert target_sensor_uuid in self.sensor_uuids[i : i + 6]
            sensor_obs = [
                observations[sensor] for sensor in self.sensor_uuids[i : i + 6]
            ]
            sensor_dtype = observations[target_sensor_uuid].dtype
            # Stacking along axis makes the flattening go in the right order.
            imgs = torch.stack(sensor_obs, axis=1)
            imgs = torch.flatten(imgs, end_dim=1)
            if not self.channels_last:
                imgs = imgs.permute((0, 3, 1, 2))  # NHWC => NCHW
            imgs = imgs.float()
            equirect = self.c2eq(imgs)  # Here is where the stiching happens
            equirect = equirect.to(dtype=sensor_dtype)
            if not self.channels_last:
                equirect = equirect.permute((0, 2, 3, 1))  # NCHW => NHWC
            observations[target_sensor_uuid] = equirect
        return observations


def get_active_obs_transforms(config: Config) -> List[ObservationTransformer]:
    active_obs_transforms = []
    if hasattr(config.RL.POLICY, "OBS_TRANSFORMS"):
        obs_transform_names = (
            config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS
        )
        for obs_transform_name in obs_transform_names:
            obs_trans_cls = baseline_registry.get_obs_transformer(
                obs_transform_name
            )
            obs_transform = obs_trans_cls.from_config(config)
            active_obs_transforms.append(obs_transform)
    return active_obs_transforms


def apply_obs_transforms_batch(
    batch: Dict[str, torch.Tensor],
    obs_transforms: Iterable[ObservationTransformer],
) -> Dict[str, torch.Tensor]:
    for obs_transform in obs_transforms:
        batch = obs_transform(batch)
    return batch


def apply_obs_transforms_obs_space(
    obs_space: SpaceDict, obs_transforms: Iterable[ObservationTransformer]
) -> SpaceDict:
    for obs_transform in obs_transforms:
        obs_space = obs_transform.transform_observation_space(obs_space)
    return obs_space
