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
import numbers
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
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
        self, observation_space: spaces.Dict, **kwargs
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
        observation_space: spaces.Dict,
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
        observation_space: spaces.Dict,
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

    def __init__(self, equ_h: int, equ_w: int):
        """Args:
        equ_h: (int) the height of the generated equirect
        equ_w: (int) the width of the generated equirect
        """
        super(Cube2Equirec, self).__init__()
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.grids = self.generate_grid(equ_h, equ_w)
        self._grids_cache = None

    def generate_grid(self, equ_h: int, equ_w: int) -> torch.Tensor:
        # Project on sphere
        theta_map, phi_map = self.get_theta_phi_map(equ_h, equ_w)
        xyz_on_sphere = self.angle2sphere(theta_map, phi_map)

        # Rotate so that each face will be in front of camera
        rotations = [
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Back
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # Down
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Front
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # Left
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # Right
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # Up
        ]
        # Generate grid
        grids = []
        for rot in rotations:
            R = torch.from_numpy(rot.T).float()
            rotate_on_sphere = torch.matmul(
                xyz_on_sphere.view((-1, 3)), R
            ).view(equ_h, equ_w, 3)

            # Project points on z=1 plane
            grid = rotate_on_sphere / torch.abs(rotate_on_sphere[..., 2:3])
            mask = torch.abs(grid).max(-1)[0] <= 1  # -1 <= grid.xy <= 1
            mask *= grid[..., 2] == 1
            grid[
                ~mask
            ] = 2  # values bigger than one will be ignored by grid_sample
            grid_xy = -grid[..., :2].unsqueeze(0)
            grids.append(grid_xy)
        grids = torch.cat(grids, dim=0)
        return grids

    def _to_equirec(self, batch: torch.Tensor) -> torch.Tensor:
        """Takes a batch of cubemaps stacked in proper order and converts thems to equirects, reduces batch size by 6"""
        batch_size, ch, _H, _W = batch.shape
        if batch_size == 0 or batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")
        output = torch.nn.functional.grid_sample(
            batch,
            self._grids_cache,
            align_corners=True,
            padding_mode="zeros",
        )
        output = output.view(
            batch_size // 6, 6, ch, self.equ_h, self.equ_w
        ).sum(dim=1)
        return output  # batch_size // 6, ch, self.equ_h, self.equ_w

    # Convert input cubic tensor to output equirectangular image
    def to_equirec_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.size()[0]

        # Check whether batch size is 6x
        if batch_size == 0 or batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")

        # to(device) is a NOOP after the first call
        self.grids = self.grids.to(batch.device)

        # Cache the repeated grids for subsequent batches
        if (
            self._grids_cache is None
            or self._grids_cache.size()[0] != batch_size
        ):
            self._grids_cache = self.grids.repeat(batch_size // 6, 1, 1, 1)
            assert self._grids_cache.size()[0] == batch_size
        self._grids_cache = self._grids_cache.to(batch.device)
        return self._to_equirec(batch)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.to_equirec_tensor(batch)

    # Get theta and phi map
    def get_theta_phi_map(self, equ_h: int, equ_w: int) -> torch.Tensor:
        phi, theta = torch.meshgrid(torch.arange(equ_h), torch.arange(equ_w))
        theta_map = -(theta + 0.5) * 2 * np.pi / equ_w + np.pi
        phi_map = -(phi + 0.5) * np.pi / equ_h + np.pi / 2
        return theta_map, phi_map

    # Project on unit sphere
    def angle2sphere(
        self, theta_map: torch.Tensor, phi_map: torch.Tensor
    ) -> torch.Tensor:
        sin_theta = torch.sin(theta_map)
        cos_theta = torch.cos(theta_map)
        sin_phi = torch.sin(phi_map)
        cos_phi = torch.cos(phi_map)
        return torch.stack(
            [cos_phi * sin_theta, sin_phi, cos_phi * cos_theta], dim=-1
        )


# TODO Measure Inheritance of CubeMap2Equirec + CubeMap2FishEye into same abstract class
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
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
    ):
        r""":param sensor: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param eq_shape: The shape of the equirectangular output (height, width)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
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
        self.sensor_uuids: List[str] = sensor_uuids
        self.eq_shape: Tuple[int] = eq_shape
        self.channels_last: bool = channels_last
        self.c2eq: nn.Module = Cube2Equirec(eq_shape[0], eq_shape[1])
        if target_uuids is None:
            self.target_uuids: List[str] = self.sensor_uuids[::6]
        else:
            self.target_uuids: List[str] = target_uuids
        # TODO support and test different FOVs than just 90

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        r"""Transforms the target UUID's sensor obs_space so it matches the new shape (EQ_H, EQ_W)"""
        # Transforms the observation space to of the target UUID
        for i, key in enumerate(self.target_uuids):
            assert (
                key in observation_space.spaces
            ), f"{key} not found in observation space: {observation_space.spaces}"
            h, w = get_image_height_width(
                observation_space.spaces[key], channels_last=True
            )
            assert (
                h == w
            ), f"cubemap height and width must be the same, but is {h} and {w}"
            logger.info(
                f"Overwrite sensor: {key} from size of ({h}, {w}) to equirect image of {self.eq_shape} from sensors: {self.sensor_uuids[i*6:(i+1)*6]}"
            )
            if (h, w) != self.eq_shape:
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
            target_uuids=target_uuids,
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for i, target_sensor_uuid in enumerate(self.target_uuids):
            # The UUID we are overwriting
            assert target_sensor_uuid in self.sensor_uuids[i * 6 : (i + 1) * 6]
            sensor_obs = [
                observations[sensor]
                for sensor in self.sensor_uuids[i * 6 : (i + 1) * 6]
            ]
            target_obs = observations[target_sensor_uuid]
            sensor_dtype = target_obs.dtype
            # Stacking along axis makes the flattening go in the right order.
            imgs = torch.stack(sensor_obs, axis=1)
            imgs = torch.flatten(imgs, end_dim=1)
            if not self.channels_last:
                imgs = imgs.permute((0, 3, 1, 2))  # NHWC => NCHW
            imgs = imgs.float()  # NCHW
            equirect = self.c2eq(imgs)  # Here is where the stiching happens
            # for debugging
            # torchvision.utils.save_image(equirect, f'sample_eqr_{target_sensor_uuid}.jpg', normalize=True, range=(0, 255) if 'rgb' in target_sensor_uuid else (0, 1))
            equirect = equirect.to(dtype=sensor_dtype)
            if not self.channels_last:
                equirect = equirect.permute((0, 2, 3, 1))  # NCHW => NHWC
            observations[target_sensor_uuid] = equirect
        return observations


class Cube2Fisheye(nn.Module):
    r"""This is the implementation to generate fisheye images from cubemap images.
    The camera model is based on the Double Sphere Camera Model (Usenko et. al.;3DV 2018).
    Paper: https://arxiv.org/abs/1807.08957
    """

    def __init__(
        self,
        fish_h: int,
        fish_w: int,
        fish_fov: float,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        xi: float,
        alpha: float,
    ):
        """Args:
        fish_h: (int) the height of the generated fisheye
        fish_w: (int) the width of the generated fisheye
        fish_fov: (float) the fov of the generated fisheye in degrees
        cx, cy: (float) the optical center of the generated fisheye
        fx, fy, xi, alpha: (float) the fisheye camera model parameters
        """
        super(Cube2Fisheye, self).__init__()
        self.fish_h = fish_h
        self.fish_w = fish_w
        self.fish_fov = fish_fov
        self.fish_param = [cx, cy, fx, fy, xi, alpha]
        self.grids = self.generate_grid(
            fish_h, fish_w, fish_fov, self.fish_param
        )
        self._grids_cache = None

    def generate_grid(
        self,
        fish_h: int,
        fish_w: int,
        fish_fov: float,
        fish_param: List[float],
    ) -> torch.Tensor:
        # Project on sphere
        xyz_on_sphere, fov_mask = self.get_points_on_sphere(
            fish_h, fish_w, fish_fov, fish_param
        )

        # Rotate so that each face will be in front of camera
        rotations = [
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Back
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # Down
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Front
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # Left
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # Right
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # Up
        ]
        # Generate grid
        grids = []
        not_assigned_mask = torch.full(
            (fish_h, fish_w), True, dtype=torch.bool
        )
        _h, _w, _ = xyz_on_sphere.shape
        for rot in rotations:
            R = torch.from_numpy(rot.T).float()
            rotate_on_sphere = torch.matmul(
                xyz_on_sphere.view((-1, 3)), R
            ).view(_h, _w, 3)

            # Project points on z=1 plane
            grid = rotate_on_sphere / torch.abs(rotate_on_sphere[..., 2:3])
            mask = torch.abs(grid).max(-1)[0] <= 1  # -1 <= grid.xy <= 1
            mask *= grid[..., 2] == 1
            # Take care of FoV
            mask *= fov_mask
            # Make sure each point is only assigned to single face
            mask *= not_assigned_mask
            # Values bigger than one will be ignored by grid_sample
            grid[~mask] = 2
            # Update not_assigned_mask
            not_assigned_mask *= ~mask
            grid_xy = -grid[..., :2].unsqueeze(0)
            grids.append(grid_xy)
        grids = torch.cat(grids, dim=0)
        return grids

    def _to_fisheye(self, batch: torch.Tensor) -> torch.Tensor:
        """Takes a batch of cubemaps stacked in proper order and converts thems to fisheye, reduces batch size by 6"""
        batch_size, ch, _H, _W = batch.shape
        if batch_size == 0 or batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")
        output = torch.nn.functional.grid_sample(
            batch,
            self._grids_cache,
            align_corners=True,
            padding_mode="zeros",
        )
        output = output.view(
            batch_size // 6, 6, ch, self.fish_h, self.fish_w
        ).sum(dim=1)
        return output  # batch_size // 6, ch, self.fish_h, self.fish_w

    # Convert input cubic tensor to output fisheye image
    def to_fisheye_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.size()[0]

        # Check whether batch size is 6x
        if batch_size == 0 or batch_size % 6 != 0:
            raise ValueError("Batch size should be 6x")

        # to(device) is a NOOP after the first call
        self.grids = self.grids.to(batch.device)

        # Cache the repeated grids for subsequent batches
        if (
            self._grids_cache is None
            or self._grids_cache.size()[0] != batch_size
        ):
            self._grids_cache = self.grids.repeat(batch_size // 6, 1, 1, 1)
            assert self._grids_cache.size()[0] == batch_size
        self._grids_cache = self._grids_cache.to(batch.device)
        return self._to_fisheye(batch)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.to_fisheye_tensor(batch)

    def get_points_on_sphere(
        self,
        fish_h: int,
        fish_w: int,
        fish_fov: float,
        fish_param: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unpack parameters
        cx, cy, fx, fy, xi, alpha = fish_param
        fov_rad = fish_fov / 180 * np.pi
        fov_cos = np.cos(fov_rad / 2)

        # Calculate unprojection
        v, u = torch.meshgrid([torch.arange(fish_h), torch.arange(fish_w)])
        mx = (u - cx) / fx
        my = (v - cy) / fy
        r2 = mx * mx + my * my
        mz = (1 - alpha * alpha * r2) / (
            alpha * torch.sqrt(1 - (2 * alpha - 1) * r2) + 1 - alpha
        )
        mz2 = mz * mz

        k1 = mz * xi + torch.sqrt(mz2 + (1 - xi * xi) * r2)
        k2 = mz2 + r2
        k = k1 / k2

        # Unprojected unit vectors
        unprojected_unit = k.unsqueeze(-1) * torch.stack([mx, my, mz], dim=-1)
        unprojected_unit[..., 2] -= xi
        # Coordinate transformation between camera and habitat
        unprojected_unit[..., 0] *= -1
        unprojected_unit[..., 1] *= -1

        # Calculate fov
        unprojected_fov_cos = unprojected_unit[
            ..., 2
        ]  # unprojected_unit @ z_axis
        fov_mask = unprojected_fov_cos >= fov_cos
        if alpha > 0.5:
            fov_mask *= r2 <= (1 / (2 * alpha - 1))

        return unprojected_unit, fov_mask


@baseline_registry.register_obs_transformer()
class CubeMap2Fisheye(ObservationTransformer):
    r"""This is an experimental use of ObservationTransformer that converts a cubemap
    output to a fisheye one through projection. This needs to be fed
    a list of 6 cameras at various orientations but will be able to stitch a
    fisheye image out of these inputs. The code below will generate a config that
    has the 6 sensors in the proper orientations. This code also assumes a 90
    FOV.

    Sensor order for cubemap stiching is Back, Down, Front, Left, Right, Up.
    The output will be writen the UUID of the first sensor.
    """

    def __init__(
        self,
        sensor_uuids: List[str],
        fish_shape: Tuple[int],
        fish_fov: float,
        fish_params: Tuple[float],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
    ):
        r""":param sensor: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param fish_shape: The shape of the fisheye output (height, width)
        :param fish_fov: The FoV of the fisheye output in degrees
        :param fish_params: The camera parameters of fisheye output (f, xi, alpha)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
        """
        super(CubeMap2Fisheye, self).__init__()
        num_sensors = len(sensor_uuids)
        assert (
            num_sensors % 6 == 0 and num_sensors != 0
        ), f"{len(sensor_uuids)}: length of sensors is not a multiple of 6"
        # TODO verify attributes of the sensors in the config if possible. Think about API design
        assert (
            len(fish_shape) == 2
        ), f"fish_shape must be a tuple of (height, width), given: {fish_shape}"
        assert len(fish_params) == 3
        self.sensor_uuids: List[str] = sensor_uuids
        self.fish_shape: Tuple[int] = fish_shape
        self.channels_last: bool = channels_last
        # fisheye camera parameters
        fx = fish_params[0] * min(fish_shape)
        fy = fx
        cx = fish_shape[1] / 2
        cy = fish_shape[0] / 2
        xi = fish_params[1]
        alpha = fish_params[2]
        self.c2fish: nn.Module = Cube2Fisheye(
            fish_shape[0], fish_shape[1], fish_fov, cx, cy, fx, fy, xi, alpha
        )

        if target_uuids is None:
            self.target_uuids: List[str] = self.sensor_uuids[::6]
        else:
            self.target_uuids: List[str] = target_uuids
        # TODO support and test different FOVs than just 90

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        r"""Transforms the target UUID's sensor obs_space so it matches the new shape (FISH_H, FISH_W)"""
        # Transforms the observation space to of the target UUID
        for i, key in enumerate(self.target_uuids):
            assert (
                key in observation_space.spaces
            ), f"{key} not found in observation space: {observation_space.spaces}"
            h, w = get_image_height_width(
                observation_space.spaces[key], channels_last=True
            )
            assert (
                h == w
            ), f"cubemap height and width must be the same, but is {h} and {w}"
            logger.info(
                f"Overwrite sensor: {key} from size of ({h}, {w}) to fisheye image of {self.fish_shape} from sensors: {self.sensor_uuids[i*6:(i+1)*6]}"
            )
            if (h, w) != self.fish_shape:
                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.fish_shape
                )
        return observation_space

    @classmethod
    def from_config(cls, config):
        cube2fish_config = config.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH
        if hasattr(cube2fish_config, "TARGET_UUIDS"):
            # Optional Config Value to specify target UUID
            target_uuids = cube2fish_config.TARGET_UUIDS
        else:
            target_uuids = None
        return cls(
            cube2fish_config.SENSOR_UUIDS,
            fish_shape=(
                cube2fish_config.HEIGHT,
                cube2fish_config.WIDTH,
            ),
            fish_fov=cube2fish_config.FOV,
            fish_params=cube2fish_config.PARAMS,
            target_uuids=target_uuids,
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for i, target_sensor_uuid in enumerate(self.target_uuids):
            # The UUID we are overwriting
            assert target_sensor_uuid in self.sensor_uuids[i * 6 : (i + 1) * 6]
            sensor_obs = [
                observations[sensor]
                for sensor in self.sensor_uuids[i * 6 : (i + 1) * 6]
            ]
            target_obs = observations[target_sensor_uuid]
            sensor_dtype = target_obs.dtype
            # Stacking along axis makes the flattening go in the right order.
            imgs = torch.stack(sensor_obs, axis=1)
            imgs = torch.flatten(imgs, end_dim=1)
            if not self.channels_last:
                imgs = imgs.permute((0, 3, 1, 2))  # NHWC => NCHW
            imgs = imgs.float()  # NCHW
            fisheye = self.c2fish(imgs)  # Here is where the stiching happens
            fisheye = fisheye.to(dtype=sensor_dtype)
            if not self.channels_last:
                fisheye = fisheye.permute((0, 2, 3, 1))  # NCHW => NHWC
            observations[target_sensor_uuid] = fisheye
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
    obs_space: spaces.Dict, obs_transforms: Iterable[ObservationTransformer]
) -> spaces.Dict:
    for obs_transform in obs_transforms:
        obs_space = obs_transform.transform_observation_space(obs_space)
    return obs_space
