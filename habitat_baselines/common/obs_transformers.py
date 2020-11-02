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
from enum import Enum
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


class _DepthFrom(Enum):
    Z_VAL = 0
    OPTI_CENTER = 1


class CameraProjection(metaclass=abc.ABCMeta):
    """This is the base CameraProjection class that converts
    projection model of images into different one. This can be used for
    conversion between cubemap, equirec, fisheye images, etc.
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
        pass

    @abc.abstractmethod
    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @property
    def rotation(self):
        # Camera rotation: points in world coord = R @ points in camera coord
        if self.R is None:
            return torch.eye(3, dtype=torch.float32)
        else:
            return self.R

    def camcoord2worldcoord(self, pts: torch.Tensor):
        if self.R is None:
            return pts
        else:
            # Rotate points according to camera rotation
            _h, _w, _ = pts.shape
            # points in world coord = R @ points in camera coord
            rotated_pts = torch.matmul(pts.view((-1, 3)), self.R.T)
            return rotated_pts.view(_h, _w, 3)

    def worldcoord2camcoord(self, pts: torch.Tensor):
        if self.R is None:
            return pts
        else:
            # Rotate points according to camera rotation
            _h, _w, _ = pts.shape
            # points in camera coord = R.T @ points in world coord
            rotated_pts = torch.matmul(pts.view((-1, 3)), self.R)
            return rotated_pts.view(_h, _w, 3)


class PerspectiveCamera(CameraProjection):
    """This is the perspective camera class."""

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
        super(PerspectiveCamera, self).__init__(
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
        unproj_pts /= torch.norm(unproj_pts, dim=-1, keepdim=True)
        # All points in image are valid
        valid_mask = torch.full(unproj_pts.shape[:2], True, dtype=torch.bool)

        # Rotate unproj_pts points according to camera rotation
        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)

        return unproj_pts, valid_mask


class EquirecCamera(CameraProjection):
    """This is the equirectanglar camera class."""

    def __init__(
        self, img_h: int, img_w: int, R: Optional[torch.Tensor] = None
    ):
        """Args:
        img_h: (int) the height of equirectanglar camera image
        img_w: (int) the width of equirectanglar camera image
        R: (torch.Tensor) 3x3 rotation matrix of camera
        """
        super(EquirecCamera, self).__init__(img_h, img_w, R)

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

    # Get theta and phi map
    def get_theta_phi_map(self, img_h: int, img_w: int) -> torch.Tensor:
        phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
        theta_map = (theta + 0.5) * 2 * np.pi / img_w - np.pi
        phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
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


class FisheyeCamera(CameraProjection):
    r"""This is the fisheye camera class.
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
        super(FisheyeCamera, self).__init__(img_h, img_w, R)

        self.fish_fov = fish_fov
        fov_rad = self.fish_fov / 180 * np.pi
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
    r"""This is the implementation to convert {cubemap, equirec, fisheye} images
    into {perspective, equirec, fisheye} images.
    """

    def __init__(
        self,
        input_projections: List[CameraProjection],
        output_projection: CameraProjection,
    ):
        """Args:
        input_projections: (list of CameraProjection) input images of projection models
        output_projection: (CameraProjection) generated image of projection model
        """
        super(ProjectionConverter, self).__init__()
        self.input_models = input_projections
        self.output_model = output_projection

        # Check if depth conversion is required
        # If depth is in z value in input, conversion is required
        self.input_zfactors = self.calculate_zfactor(self.input_models)
        # If depth is in z value in output, inverse conversion is required
        output_invzfactors = self.calculate_zfactor([self.output_model])
        if output_invzfactors is not None:
            self.output_zfactors = 1 / output_invzfactors
        else:
            self.output_zfactors = None

        self.grids = self.generate_grid()
        self._grids_cache = None

    def generate_grid(self) -> torch.Tensor:
        # Obtain points on unit sphere
        world_pts, not_assigned_mask = self.output_model.unprojection()
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

    def _convert(self, batch: torch.Tensor) -> torch.Tensor:
        """Takes a batch of images stacked in proper order and converts thems, reduces batch size by input_len"""
        batch_size, ch, _H, _W = batch.shape
        input_len = len(self.input_models)
        if batch_size == 0 or batch_size % input_len != 0:
            raise ValueError(f"Batch size should be {input_len}x")
        output = torch.nn.functional.grid_sample(
            batch,
            self._grids_cache,
            align_corners=True,
            padding_mode="zeros",
        )
        output = output.view(
            batch_size // input_len,
            input_len,
            ch,
            self.output_model.img_h,
            self.output_model.img_w,
        ).sum(dim=1)
        return output  # batch_size // input_len, ch, output_model.img_h, output_model.img_w

    def to_converted_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size, ch = batch.size()[:2]
        input_len = len(self.input_models)

        # Check whether batch size is len(self.input_models) x
        if batch_size == 0 or batch_size % input_len != 0:
            raise ValueError(f"Batch size should be {input_len}x")

        # to(device) is a NOOP after the first call
        self.grids = self.grids.to(batch.device)

        # Depth conversion
        if ch == 1 and self.input_zfactors is not None:
            self.input_zfactors = self.input_zfactors.to(batch.device)
            batch = batch * self.input_zfactors.repeat(
                batch_size // input_len, 1, 1, 1
            )

        # Cache the repeated grids for subsequent batches
        if (
            self._grids_cache is None
            or self._grids_cache.size()[0] != batch_size
        ):
            self._grids_cache = self.grids.repeat(
                batch_size // input_len, 1, 1, 1
            )
            assert self._grids_cache.size()[0] == batch_size
        self._grids_cache = self._grids_cache.to(batch.device)

        out = self._convert(batch)
        # Depth conversion
        if ch == 1 and self.output_zfactors is not None:
            output_b = batch_size // input_len
            self.output_zfactors = self.output_zfactors.to(batch.device)
            out = out * self.output_zfactors.repeat(output_b, 1, 1, 1)
        return out

    def calculate_zfactor(self, projections: List[CameraProjection]):
        # z factor to convert depth in z value to depth from optical center
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
            return z_factors

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.to_converted_tensor(batch)


class _RotationMat(object):
    BACK = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    DOWN = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    FRONT = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    LEFT = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    RIGHT = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    UP = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    @classmethod
    def for_cubemap(cls) -> List[torch.Tensor]:
        """Get rotation matrix for cubemap. The orders are
        'BACK', 'DOWN', 'FRONT', 'LEFT', 'RIGHT', 'UP'
        """
        face_orders = ["BACK", "DOWN", "FRONT", "LEFT", "RIGHT", "UP"]
        return [getattr(cls, face) for face in face_orders]


def get_cubemap_projections(
    img_h: int = 256, img_w: int = 256
) -> List[CameraProjection]:
    """Get cubemap camera projections that consist of six PerspectiveCameras.
    The orders are 'BACK', 'DOWN', 'FRONT', 'LEFT', 'RIGHT', 'UP'.
    img_h: (int) the height of camera image
    img_w: (int) the width of camera image
    """
    projections = []
    for rot in _RotationMat.for_cubemap():
        cam = PerspectiveCamera(img_h, img_w, R=rot)
        projections.append(cam)
    return projections


class Cube2Equirec(ProjectionConverter):
    """This is the backend Cube2Equirec nn.module that does the stiching.
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
        output_projection = EquirecCamera(equ_h, equ_w)
        super(Cube2Equirec, self).__init__(
            input_projections, output_projection
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
        input_projections = get_cubemap_projections()

        # Fisheye output
        output_projection = FisheyeCamera(
            fish_h, fish_w, fish_fov, cx, cy, fx, fy, xi, alpha
        )
        super(Cube2Fisheye, self).__init__(
            input_projections, output_projection
        )


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
