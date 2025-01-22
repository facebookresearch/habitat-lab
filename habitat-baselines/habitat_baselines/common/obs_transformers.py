#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
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
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch import nn

from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    image_resize_shortest_edge,
    overwrite_gym_box_shape,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


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
    def from_config(cls, config: "DictConfig"):
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
        trans_keys: Tuple[str, ...] = ("rgb", "depth", "semantic"),
        semantic_key: str = "semantic",
    ):
        """Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension
        """
        super(ResizeShortestEdge, self).__init__()
        self._size: int = size
        self.channels_last: bool = channels_last
        self.trans_keys: Tuple[str, ...] = trans_keys
        self.semantic_key = semantic_key

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

    def _transform_obs(
        self, obs: torch.Tensor, interpolation_mode: str
    ) -> torch.Tensor:
        return image_resize_shortest_edge(
            obs,
            self._size,
            channels_last=self.channels_last,
            interpolation_mode=interpolation_mode,
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            for sensor in self.trans_keys:
                if sensor in observations:
                    interpolation_mode = "area"
                    if self.semantic_key in sensor:
                        interpolation_mode = "nearest"
                    observations[sensor] = self._transform_obs(
                        observations[sensor], interpolation_mode
                    )
        return observations

    @classmethod
    def from_config(cls, config: "DictConfig"):
        return cls(
            config.size,
            config.channels_last,
            config.trans_keys,
            config.semantic_key,
        )


@baseline_registry.register_obs_transformer()
class CenterCropper(ObservationTransformer):
    """An observation transformer is a simple nn module that center crops your input."""

    def __init__(
        self,
        size: Union[numbers.Integral, Tuple[int, int]],
        channels_last: bool = True,
        trans_keys: Tuple[str, ...] = ("rgb", "depth", "semantic"),
    ):
        """Args:
        size: A sequence (h, w) or int of the size you wish to resize/center_crop.
                If int, assumes square crop
        channels_list: indicates if channels is the last dimension
        trans_keys: The list of sensors it will try to centercrop.
        """
        super().__init__()
        if isinstance(size, numbers.Integral):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (h, w)"
        self._size = size
        self.channels_last = channels_last
        self.trans_keys = trans_keys

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
    def from_config(cls, config: "DictConfig"):
        return cls(
            (config.height, config.width),
            config.channels_last,
            config.trans_keys,
        )


class _DepthFrom(Enum):
    Z_VAL = 0
    OPTI_CENTER = 1


class CameraProjection(metaclass=abc.ABCMeta):
    """This is the base CameraProjection class that converts
    projection model of images into different one. This can be used for
    conversion between cubemap, equirect, fisheye images, etc.
    projection that project 3D points onto the image plane and
    unprojection that project image points onto unit sphere
    must be implemented."""

    def __init__(
        self,
        img_h: int,
        img_w: int,
        R: Optional[torch.Tensor] = None,
        depth_from: _DepthFrom = _DepthFrom.OPTI_CENTER,
    ):
        """Args:
        img_h: (int) the height of camera image
        img_w: (int) the width of camera image
        R: (torch.Tensor) 3x3 rotation matrix of camera
        depth_from: (_DepthFrom) the depth from z value or optical center
        """
        self.img_h = img_h
        self.img_w = img_w
        self.depth_from = depth_from

        # Camera rotation: points in world coord = R @ points in camera coord
        if R is not None:
            self.R = R.float()
        else:
            self.R = None

    @abc.abstractmethod
    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project points in world coord onto image planes.
        Args:
            world_pts: 3D points in world coord
        Returns:
            proj_pts: Projected points for grid_sample, -1 <= proj_pts <= 1
            valid_mask: True if the point is valid (inside FoV)
        """

    @abc.abstractmethod
    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unproject 2D image points onto unit sphere.
        Args:
            with_rotation: If True, unprojected points is in world coord.
                           If False, unprojected points is in camera coord.
        Returns:
            unproj_pts: Unprojected 3D points on unit sphere
            valid_mask: True if the point is valid (inside FoV)
        """

    @property
    def rotation(self):
        """Camera rotation: points in world coord = R @ points in camera coord"""
        if self.R is None:
            return torch.eye(3, dtype=torch.float32)
        else:
            return self.R

    @property
    def shape(self):
        """Camera image shape: (img_h, img_w)"""
        return (self.img_h, self.img_w)

    def size(self):
        """Camera image shape: (img_h, img_w)"""
        return self.shape

    def camcoord2worldcoord(self, pts: torch.Tensor):
        """Convert points in camera coords into points in world coords.
        Args:
            pts: 3D points in camera coords
        Returns:
            rotated_pts: 3D points in world coords
        """
        if self.R is None:
            return pts
        else:
            # Rotate points according to camera rotation
            _h, _w, _ = pts.shape
            # points in world coord = R @ points in camera coord
            rotated_pts = torch.matmul(pts.view((-1, 3)), self.R.T)  # type: ignore
            return rotated_pts.view(_h, _w, 3)

    def worldcoord2camcoord(self, pts: torch.Tensor):
        """Convert points in world coords into points in camera coords.
        Args:
            pts: 3D points in world coords
        Returns:
            rotated_pts: 3D points in camera coords
        """
        if self.R is None:
            return pts
        else:
            # Rotate points according to camera rotation
            _h, _w, _ = pts.shape
            # points in camera coord = R.T @ points in world coord
            rotated_pts = torch.matmul(pts.view((-1, 3)), self.R)
            return rotated_pts.view(_h, _w, 3)


class PerspectiveProjection(CameraProjection):
    """This is the perspective camera projection class."""

    def __init__(
        self,
        img_h: int,
        img_w: int,
        f: Optional[float] = None,
        R: Optional[torch.Tensor] = None,
    ):
        """Args:
        img_h: (int) the height of camera image
        img_w: (int) the width of camera image
        f: (float) the focal length of camera
        R: (torch.Tensor) 3x3 rotation matrix of camera
        """
        super(PerspectiveProjection, self).__init__(
            img_h, img_w, R, _DepthFrom.Z_VAL
        )
        if f is None:
            self.f = max(img_h, img_w) / 2
        else:
            self.f = f

    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rotate world points according to camera rotation
        world_pts = self.worldcoord2camcoord(world_pts)

        # Project points onto image plane
        img_pts = self.f * world_pts / torch.abs(world_pts[..., 2:3])
        cx = self.img_w / 2
        cy = self.img_h / 2
        u = img_pts[..., 0] + cx
        v = img_pts[..., 1] + cy

        # For grid_sample, -1 <= proj_pts <= 1
        mapx = 2 * u / self.img_w - 1.0
        mapy = 2 * v / self.img_h - 1.0
        proj_pts = torch.stack([mapx, mapy], dim=-1)

        # Valid mask
        valid_mask = torch.abs(proj_pts).max(-1)[0] <= 1  # -1 <= grid.xy <= 1
        valid_mask *= img_pts[..., 2] > 0
        return proj_pts, valid_mask

    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v, u = torch.meshgrid(
            torch.arange(self.img_h), torch.arange(self.img_w)
        )
        x = (u + 0.5) - self.img_w / 2
        y = (v + 0.5) - self.img_h / 2
        z = torch.full_like(x, self.f, dtype=torch.float)
        unproj_pts = torch.stack([x, y, z], dim=-1)
        # Project on unit shpere
        unproj_pts /= torch.linalg.norm(unproj_pts, dim=-1, keepdim=True)
        # All points in image are valid
        valid_mask = torch.full(unproj_pts.shape[:2], True, dtype=torch.bool)

        # Rotate unproj_pts points according to camera rotation
        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)

        return unproj_pts, valid_mask


class EquirectProjection(CameraProjection):
    """This is the equirectanglar camera projection class."""

    def __init__(
        self, img_h: int, img_w: int, R: Optional[torch.Tensor] = None
    ):
        """Args:
        img_h: (int) the height of equirectanglar camera image
        img_w: (int) the width of equirectanglar camera image
        R: (torch.Tensor) 3x3 rotation matrix of camera
        """
        super(EquirectProjection, self).__init__(img_h, img_w, R)

    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rotate world points according to camera rotation
        world_pts = self.worldcoord2camcoord(world_pts)

        x, y, z = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]
        # x,y,z to theta, phi
        theta = torch.atan2(x, z)
        c = torch.sqrt(x * x + z * z)
        phi = torch.atan2(y, c)

        # For grid_sample, -1 <= proj_pts <= 1
        mapx = theta / np.pi
        mapy = phi / (np.pi / 2)
        proj_pts = torch.stack([mapx, mapy], dim=-1)

        # All points in image are valid
        valid_mask = torch.full(proj_pts.shape[:2], True, dtype=torch.bool)
        return proj_pts, valid_mask

    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_map, phi_map = self.get_theta_phi_map(self.img_h, self.img_w)
        unproj_pts = self.angle2sphere(theta_map, phi_map)
        # All points in image are valid
        valid_mask = torch.full(unproj_pts.shape[:2], True, dtype=torch.bool)
        # Rotate unproj_pts points according to camera rotation
        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)
        return unproj_pts, valid_mask

    def get_theta_phi_map(
        self, img_h: int, img_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get theta and phi map for equirectangular image.
        PI < theta_map < PI,  PI/2 < phi_map < PI/2
        """
        phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
        theta_map = (theta + 0.5) * 2 * np.pi / img_w - np.pi
        phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
        return theta_map, phi_map

    def angle2sphere(
        self, theta_map: torch.Tensor, phi_map: torch.Tensor
    ) -> torch.Tensor:
        """Project points on unit sphere based on theta and phi map."""
        sin_theta = torch.sin(theta_map)
        cos_theta = torch.cos(theta_map)
        sin_phi = torch.sin(phi_map)
        cos_phi = torch.cos(phi_map)
        return torch.stack(
            [cos_phi * sin_theta, sin_phi, cos_phi * cos_theta], dim=-1
        )


class FisheyeProjection(CameraProjection):
    r"""This is the fisheye camera projection class.
    The camera model is based on the Double Sphere Camera Model (Usenko et. al.;3DV 2018).
    Paper: https://arxiv.org/abs/1807.08957
    Implementation: https://github.com/matsuren/dscamera
    """

    def __init__(
        self,
        img_h: int,
        img_w: int,
        fish_fov: float,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        xi: float,
        alpha: float,
        R: Optional[torch.Tensor] = None,
    ):
        """Args:
        img_h: (int) the height of fisheye camera image
        img_w: (int) the width of fisheye camera image
        fish_fov: (float) the fov of fisheye camera in degrees
        cx, cy: (float) the optical center of the fisheye camera
        fx, fy, xi, alpha: (float) the fisheye camera model parameters
        R: (torch.Tensor) 3x3 rotation matrix of camera
        """
        super(FisheyeProjection, self).__init__(img_h, img_w, R)

        self.fish_fov = fish_fov  # FoV in degrees
        fov_rad = self.fish_fov / 180 * np.pi  # FoV in radians
        self.fov_cos = np.cos(fov_rad / 2)
        self.fish_param = [cx, cy, fx, fy, xi, alpha]

    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rotate world points according to camera rotation
        world_pts = self.worldcoord2camcoord(world_pts)

        # Unpack parameters
        cx, cy, fx, fy, xi, alpha = self.fish_param
        # Unpack 3D world points
        x, y, z = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]

        # Calculate fov
        world_pts_fov_cos = z  # point3D @ z_axis
        fov_mask = world_pts_fov_cos >= self.fov_cos

        # Calculate projection
        x2 = x * x
        y2 = y * y
        z2 = z * z
        d1 = torch.sqrt(x2 + y2 + z2)
        zxi = xi * d1 + z
        d2 = torch.sqrt(x2 + y2 + zxi * zxi)

        div = alpha * d2 + (1 - alpha) * zxi
        u = fx * x / div + cx
        v = fy * y / div + cy

        # Projected points on image plane
        # For grid_sample, -1 <= proj_pts <= 1
        mapx = 2 * u / self.img_w - 1.0
        mapy = 2 * v / self.img_h - 1.0
        proj_pts = torch.stack([mapx, mapy], dim=-1)

        # Check valid area
        if alpha <= 0.5:
            w1 = alpha / (1 - alpha)
        else:
            w1 = (1 - alpha) / alpha
        w2 = w1 + xi / np.sqrt(2 * w1 * xi + xi * xi + 1)
        valid_mask = z > -w2 * d1
        valid_mask *= fov_mask

        return proj_pts, valid_mask

    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unpack parameters
        cx, cy, fx, fy, xi, alpha = self.fish_param

        # Calculate unprojection
        v, u = torch.meshgrid(
            [torch.arange(self.img_h), torch.arange(self.img_w)]
        )
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
        unproj_pts = k.unsqueeze(-1) * torch.stack([mx, my, mz], dim=-1)
        unproj_pts[..., 2] -= xi

        # Calculate fov
        unproj_fov_cos = unproj_pts[..., 2]  # unproj_pts @ z_axis
        fov_mask = unproj_fov_cos >= self.fov_cos
        if alpha > 0.5:
            fov_mask *= r2 <= (1 / (2 * alpha - 1))

        # Rotate unproj_pts points according to camera rotation
        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)

        return unproj_pts, fov_mask


class ProjectionConverter(nn.Module):
    r"""This is the implementation to convert {cubemap, equirect, fisheye} images
    into {perspective, equirect, fisheye} images.
    """

    def __init__(
        self,
        input_projections: Union[List[CameraProjection], CameraProjection],
        output_projections: Union[List[CameraProjection], CameraProjection],
    ):
        """Args:
        input_projections: input images of projection models
        output_projections: generated image of projection models
        """
        super(ProjectionConverter, self).__init__()
        # Convert to list
        if not isinstance(input_projections, list):
            input_projections = [input_projections]
        if not isinstance(output_projections, list):
            output_projections = [output_projections]

        self.input_models = input_projections
        self.output_models = output_projections
        self.input_len = len(self.input_models)
        self.output_len = len(self.output_models)

        # Check image size
        input_size = self.input_models[0].size()
        for it in self.input_models:
            assert (
                input_size == it.size()
            ), "All input models must have the same image size"

        output_size = self.output_models[0].size()
        for it in self.output_models:
            assert (
                output_size == it.size()
            ), "All output models must have the same image size"

        # Check if depth conversion is required
        # If depth is in z value in input, conversion is required
        self.input_zfactor = self.calculate_zfactor(self.input_models)
        # If depth is in z value in output, inverse conversion is required
        self.output_zfactor = self.calculate_zfactor(
            self.output_models, inverse=True
        )

        # grids shape: (output_len, input_len, output_img_h, output_img_w, 2)
        self.grids = self.generate_grid()
        # _grids_cache shape: (batch_size*output_len*input_len, output_img_h, output_img_w, 2)
        self._grids_cache: Optional[torch.Tensor] = None

    def _generate_grid_one_output(
        self, output_model: CameraProjection
    ) -> torch.Tensor:
        # Obtain points on unit sphere
        world_pts, not_assigned_mask = output_model.unprojection()
        # Generate grid
        grids = []
        for input_model in self.input_models:
            grid, input_mask = input_model.projection(world_pts)
            # Make sure each point is only assigned to single input
            input_mask *= not_assigned_mask
            # Values bigger than one will be ignored by grid_sample
            grid[~input_mask] = 2
            # Update not_assigned_mask
            not_assigned_mask *= ~input_mask
            grids.append(grid)
        grids = torch.stack(grids, dim=0)
        return grids

    def generate_grid(self) -> torch.Tensor:
        multi_output_grids = []
        for output_model in self.output_models:
            grids = self._generate_grid_one_output(output_model)
            multi_output_grids.append(grids.unsqueeze(1))
        multi_output_grids = torch.cat(multi_output_grids, dim=1)
        return multi_output_grids  # input_len, output_len, output_img_h, output_img_w, 2

    def _convert(self, batch: torch.Tensor) -> torch.Tensor:
        """Takes a batch of images stacked in proper order and converts thems,
        reduces batch size by input_len."""
        batch_size, ch, _H, _W = batch.shape
        out_h, out_w = self.output_models[0].size()
        if batch_size == 0 or batch_size % self.input_len != 0:
            raise ValueError(f"Batch size should be {self.input_len}x")
        output = torch.nn.functional.grid_sample(
            batch,
            self._grids_cache,
            align_corners=True,
            padding_mode="zeros",
        )
        output = output.view(
            batch_size // self.input_len,
            self.input_len,
            ch,
            out_h,
            out_w,
        ).sum(dim=1)
        return output  # output_len * batch_size, ch, output_model.img_h, output_model.img_w

    def to_converted_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert tensors based on projection models. If there are two
        batches from two envs (R_1st, G_1st, B_1st) and (R_2nd, G_2nd, B_2nd),
        the input order is [R_1st, G_1st, B_1st, R_2nd, G_2nd, B_2nd]
        """
        # batch tensor order should be NCHW
        batch_size, ch, in_h, in_w = batch.size()

        out_h, out_w = self.output_models[0].size()

        # Check whether batch size is len(self.input_models) x
        if batch_size == 0 or batch_size % self.input_len != 0:
            raise ValueError(f"Batch size should be {self.input_len}x")

        # How many sets of input.
        num_input_set = batch_size // self.input_len

        # to(device) is a NOOP after the first call
        self.grids = self.grids.to(batch.device)

        # Adjust batch for multiple outputs
        # batch must be [1st batch * output_len, 2nd batch * output_len, ...]
        # not that [1st batch, 2nd batch, ...] * output_len
        multi_out_batch = (
            batch.view(num_input_set, self.input_len, ch, in_h, in_w)
            .repeat(1, self.output_len, 1, 1, 1)
            .view(self.output_len * batch_size, ch, in_h, in_w)
        )

        # Cache the repeated grids for subsequent batches
        if (
            self._grids_cache is None
            or self._grids_cache.size()[0] != multi_out_batch.size()[0]
        ):
            # batch size is more than one
            self._grids_cache = self.grids.repeat(
                num_input_set, 1, 1, 1, 1
            ).view(batch_size * self.output_len, out_h, out_w, 2)
        self._grids_cache = self._grids_cache.to(batch.device)

        return self._convert(multi_out_batch)

    def calculate_zfactor(
        self, projections: List[CameraProjection], inverse: bool = False
    ) -> Optional[torch.Tensor]:
        """Calculate z factor based on camera projection models. z_factor is
        used for converting depth in z value to depth from optical center
        (for input_models) or conversion of depth from optical center to depth
        in z value (inverse = True, for output_models). Whether the conversion
        is required or not is decided based on depth_from property of
        CameraProjection class.
        Args:
            projections: input or output projection models
            inverse: True to convert depth from optical center to z value
                     False to convert z value to depth from optical center
        Returns:
            z_factors: z factor. Return None if conversion is not required.
        """
        z_factors = []
        for cam in projections:
            if cam.depth_from == _DepthFrom.Z_VAL:
                pts_on_sphere, _ = cam.unprojection(with_rotation=False)
                zval_to_optcenter = 1 / pts_on_sphere[..., 2]
                z_factors.append(zval_to_optcenter.unsqueeze(0))
            else:
                all_one = torch.full(
                    (1, cam.img_h, cam.img_w), 1.0, dtype=torch.float
                )
                z_factors.append(all_one)
        z_factors = torch.stack(z_factors)

        if (z_factors == 1.0).all():
            # All input cameras have depth from optical center
            return None
        else:
            if not inverse:
                # for input_models
                return z_factors
            else:
                # for output_models
                return 1 / z_factors

    def forward(
        self, batch: torch.Tensor, is_depth: bool = False
    ) -> torch.Tensor:
        # Depth conversion for input tensors
        if is_depth and self.input_zfactor is not None:
            input_b = batch.size()[0] // self.input_len
            self.input_zfactor = self.input_zfactor.to(batch.device)
            batch = batch * self.input_zfactor.repeat(input_b, 1, 1, 1)

        # Common operator to convert projection models
        out = self.to_converted_tensor(batch)

        # Depth conversion for output tensors
        if is_depth and self.output_zfactor is not None:
            output_b = out.size()[0] // self.output_len
            self.output_zfactor = self.output_zfactor.to(batch.device)
            out = out * self.output_zfactor.repeat(output_b, 1, 1, 1)

        return out


def get_cubemap_projections(
    img_h: int = 256, img_w: int = 256
) -> List[CameraProjection]:
    """Get cubemap camera projections that consist of six PerspectiveCameras.
    The orders are 'BACK', 'DOWN', 'FRONT', 'LEFT', 'RIGHT', 'UP'.
    Args:
        img_h: (int) the height of camera image
        img_w: (int) the width of camera image

    The rotation matrices are equivalent to
    .. code-block:: python
        from scipy.spatial.transform import Rotation
        rotations = [
            Rotation.from_euler("y", 180, degrees=True),  # Back
            Rotation.from_euler("x", -90, degrees=True),  # Down
            Rotation.from_euler("x", 0, degrees=True),  # Front
            Rotation.from_euler("y", -90, degrees=True),  # Left
            Rotation.from_euler("y", 90, degrees=True),  # Right
            Rotation.from_euler("x", 90, degrees=True)  # Up
        ]
    """
    rotations = [
        torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Back
        torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # Down
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Front
        torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),  # Left
        torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # Right
        torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),  # Up
    ]

    projections: List[CameraProjection] = []
    for rot in rotations:
        cam = PerspectiveProjection(img_h, img_w, R=rot)
        projections.append(cam)
    return projections


class Cube2Equirect(ProjectionConverter):
    """This is the backend Cube2Equirect nn.module that does the stitching.
    Inspired from https://github.com/fuenwang/PanoramaUtility and
    optimized for modern PyTorch."""

    def __init__(self, equ_h: int, equ_w: int):
        """Args:
        equ_h: (int) the height of the generated equirect
        equ_w: (int) the width of the generated equirect
        """

        # Cubemap input
        input_projections = get_cubemap_projections()

        # Equirectangular output
        output_projection = EquirectProjection(equ_h, equ_w)
        super(Cube2Equirect, self).__init__(
            input_projections, output_projection
        )


class ProjectionTransformer(ObservationTransformer):
    r"""
    ProjectionTransformer base class. It can be used to  convert {cubemap, equirect, fisheye} images
    into {perspective, equirect, fisheye} images in ObservationTransformer.
    """

    def __init__(
        self,
        converter: ProjectionConverter,
        sensor_uuids: List[str],
        image_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):
        r""":param converter: ProjectionConverter class
        :param sensor_uuids: List of sensor_uuids
        :param image_shape: The shape of the output image (height, width)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
        :param depth_key: If sensor_uuids has depth_key substring, they are processed as depth
        """
        super(ProjectionTransformer, self).__init__()
        num_sensors = len(sensor_uuids)
        assert (
            num_sensors % converter.input_len == 0 and num_sensors != 0
        ), f"{len(sensor_uuids)}: length of sensors is not a multiple of {converter.input_len}"
        # TODO verify attributes of the sensors in the config if possible. Think about API design
        assert (
            len(image_shape) == 2
        ), f"image_shape must be a tuple of (height, width), given: {image_shape}"
        self.sensor_uuids: List[str] = sensor_uuids
        self.img_shape: Tuple[int, int] = image_shape
        self.channels_last: bool = channels_last
        self.converter = converter
        if target_uuids is None:
            target_uuids = self.sensor_uuids[::6]
        self.target_uuids: List[str] = target_uuids
        self.depth_key = depth_key

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        r"""Transforms the target uuid's sensor obs_space so it matches the new shape (H, W)"""
        # Transforms the observation space to of the target uuid
        observation_space = copy.deepcopy(observation_space)
        for i, key in enumerate(self.target_uuids):
            assert (
                key in observation_space.spaces
            ), f"{key} not found in observation space: {observation_space.spaces}"
            h, w = get_image_height_width(
                observation_space.spaces[key], channels_last=True
            )
            in_len = self.converter.input_len
            logger.info(
                f"Overwrite sensor: {key} from size of ({h}, {w}) to image of"
                f" {self.img_shape} from sensors: {self.sensor_uuids[i*in_len:(i+1)*in_len]}"
            )
            if (h, w) != self.img_shape:
                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.img_shape
                )
        return observation_space

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for i, target_sensor_uuid in enumerate(self.target_uuids):
            # number of input and input sensor uuids
            in_len = self.converter.input_len
            in_sensor_uuids = self.sensor_uuids[i * in_len : (i + 1) * in_len]

            # If the sensor is depth
            is_depth = any(self.depth_key in s for s in in_sensor_uuids)

            # The UUID we are overwriting
            assert target_sensor_uuid in in_sensor_uuids
            sensor_obs = [observations[sensor] for sensor in in_sensor_uuids]
            target_obs = observations[target_sensor_uuid]
            sensor_dtype = target_obs.dtype
            # Stacking along axis makes the flattening go in the right order.
            imgs = torch.stack(sensor_obs, dim=1)
            imgs = torch.flatten(imgs, end_dim=1)
            if not self.channels_last:
                imgs = imgs.permute((0, 3, 1, 2))  # NHWC => NCHW
            imgs = imgs.float()  # NCHW
            # Here is where the projection conversion happens
            output = self.converter(imgs, is_depth=is_depth)

            # for debugging
            # torchvision.utils.save_image(output, f'sample_eqr_{target_sensor_uuid}.jpg', normalize=True, range=(0, 255) if 'rgb' in target_sensor_uuid else (0, 1))
            output = output.to(dtype=sensor_dtype)
            if not self.channels_last:
                output = output.permute((0, 2, 3, 1))  # NCHW => NHWC
            observations[target_sensor_uuid] = output
        return observations


@baseline_registry.register_obs_transformer()
class CubeMap2Equirect(ProjectionTransformer):
    r"""This is an experimental use of ObservationTransformer that converts a cubemap
    output to an equirectangular one through projection. This needs to be fed
    a list of 6 cameras at various orientations but will be able to stitch a
    360 sensor out of these inputs. The code below will generate a config that
    has the 6 sensors in the proper orientations. This code also assumes a 90
    FOV.

    Sensor order for cubemap stitching is Back, Down, Front, Left, Right, Up.
    The output will be written the UUID of the first sensor.
    """

    def __init__(
        self,
        sensor_uuids: List[str],
        eq_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):
        r""":param sensor_uuids: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param eq_shape: The shape of the equirectangular output (height, width)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
        :param depth_key: If sensor_uuids has depth_key substring, they are processed as depth
        """

        converter = Cube2Equirect(eq_shape[0], eq_shape[1])
        super(CubeMap2Equirect, self).__init__(
            converter,
            sensor_uuids,
            eq_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        if hasattr(config, "target_uuids"):
            # Optional Config Value to specify target uuid
            target_uuids = config.target_uuids
        else:
            target_uuids = None
        return cls(
            config.sensor_uuids,
            eq_shape=(
                config.height,
                config.width,
            ),
            target_uuids=target_uuids,
        )


class Cube2Fisheye(ProjectionConverter):
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

        # Cubemap input
        input_projections = get_cubemap_projections(fish_h, fish_w)

        # Fisheye output
        output_projection = FisheyeProjection(
            fish_h, fish_w, fish_fov, cx, cy, fx, fy, xi, alpha
        )
        super(Cube2Fisheye, self).__init__(
            input_projections, output_projection
        )


@baseline_registry.register_obs_transformer()
class CubeMap2Fisheye(ProjectionTransformer):
    r"""This is an experimental use of ObservationTransformer that converts a cubemap
    output to a fisheye one through projection. This needs to be fed
    a list of 6 cameras at various orientations but will be able to stitch a
    fisheye image out of these inputs. The code below will generate a config that
    has the 6 sensors in the proper orientations. This code also assumes a 90
    FOV.

    Sensor order for cubemap stitching is Back, Down, Front, Left, Right, Up.
    The output will be written the UUID of the first sensor.
    """

    def __init__(
        self,
        sensor_uuids: List[str],
        fish_shape: Tuple[int, int],
        fish_fov: float,
        fish_params: Tuple[float, float, float],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):
        r""":param sensor_uuids: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param fish_shape: The shape of the fisheye output (height, width)
        :param fish_fov: The FoV of the fisheye output in degrees
        :param fish_params: The camera parameters of fisheye output (f, xi, alpha)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
        :param depth_key: If sensor_uuids has depth_key substring, they are processed as depth
        """

        assert (
            len(fish_params) == 3
        ), "fish_params must have three parameters (f, xi, alpha)"
        # fisheye camera parameters
        fx = fish_params[0] * min(fish_shape)
        fy = fx
        cx = fish_shape[1] / 2
        cy = fish_shape[0] / 2
        xi = fish_params[1]
        alpha = fish_params[2]
        converter: ProjectionConverter = Cube2Fisheye(
            fish_shape[0], fish_shape[1], fish_fov, cx, cy, fx, fy, xi, alpha
        )

        super(CubeMap2Fisheye, self).__init__(
            converter,
            sensor_uuids,
            fish_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        if hasattr(config, "target_uuids"):
            # Optional Config Value to specify target uuid
            target_uuids = config.target_uuids
        else:
            target_uuids = None
        return cls(
            config.sensor_uuids,
            fish_shape=(
                config.height,
                config.width,
            ),
            fish_fov=config.fov,
            fish_params=config.params,
            target_uuids=target_uuids,
        )


class Equirect2Cube(ProjectionConverter):
    """This is the backend Equirect2CubeMap that converts equirectangular image
    to cubemap images."""

    def __init__(self, img_h: int, img_w: int):
        """Args:
        img_h: (int) the height of the generated cubemap
        img_w: (int) the width of the generated cubemap
        """

        # Equirectangular input
        input_projection = EquirectProjection(256, 512)

        #  Cubemap output
        output_projections = get_cubemap_projections(img_h, img_w)
        super(Equirect2Cube, self).__init__(
            input_projection, output_projections
        )


@baseline_registry.register_obs_transformer()
class Equirect2CubeMap(ProjectionTransformer):
    r"""This is an experimental use of ObservationTransformer that converts
    an equirectangular image to cubemap images.
    Cubemap order is Back, Down, Front, Left, Right, Up.
    The output will be written the UUID of the first sensor.
    """

    def __init__(
        self,
        sensor_uuids: List[str],
        img_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):
        r""":param sensor_uuids: List of sensor_uuids: Back, Down, Front, Left, Right, Up.
        :param img_shape: The shape of the equirectangular output (height, width)
        :param channels_last: Are the channels last in the input
        :param target_uuids: Optional List of which of the sensor_uuids to overwrite
        :param depth_key: If sensor_uuids has depth_key substring, they are processed as depth
        """

        converter = Equirect2Cube(img_shape[0], img_shape[1])
        super(Equirect2CubeMap, self).__init__(
            converter,
            sensor_uuids,
            img_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        if hasattr(config, "target_uuids"):
            # Optional Config Value to specify target uuid
            target_uuids = config.target_uuids
        else:
            target_uuids = None
        return cls(
            config.sensor_uuids,
            img_shape=(
                config.height,
                config.width,
            ),
            target_uuids=target_uuids,
        )


def get_active_obs_transforms(
    config: "DictConfig", agent_name: str = None
) -> List[ObservationTransformer]:
    active_obs_transforms = []

    # When using observation transformations, we
    # assume for now that the observation space is shared among agents
    agent_name = list(config.habitat_baselines.rl.policy.keys())[0]
    obs_trans_conf = config.habitat_baselines.rl.policy[
        agent_name
    ].obs_transforms
    if hasattr(
        config.habitat_baselines.rl.policy[agent_name], "obs_transforms"
    ):
        for obs_transform_config in obs_trans_conf.values():
            obs_trans_cls = baseline_registry.get_obs_transformer(
                obs_transform_config.type
            )
            if obs_trans_cls is None:
                raise ValueError(
                    f"Unkown ObservationTransform with name {obs_transform_config.type}."
                )
            obs_transform = obs_trans_cls.from_config(obs_transform_config)
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


@baseline_registry.register_obs_transformer()
class AddVirtualKeys(ObservationTransformer):
    """
    Will add sensor values with all 0s to the observation. This is used in the
    TP-SRL method since skills are trained with sensors not available during
    evaluation (such as the target object to navigate to for the navigation
    policy). The method is responsible for filling out these 0 value
    observations before passing them into the policy.
    """

    def __init__(self, virtual_keys):
        super().__init__()
        self._virtual_keys = virtual_keys

    def transform_observation_space(
        self, observation_space: spaces.Dict, **kwargs
    ):
        for k, obs_dim in self._virtual_keys.virtual_keys.items():
            observation_space[k] = spaces.Box(
                shape=(obs_dim,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )
        return observation_space

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        first_obs = next(iter(observations.values()))
        device = first_obs.device
        batch_dim = first_obs.shape[0]
        for k, obs_dim in self._virtual_keys.virtual_keys.items():
            observations[k] = torch.zeros((batch_dim, obs_dim), device=device)
        return observations

    @classmethod
    def from_config(cls, config):
        return cls(config)
