#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn as nn

from habitat_baselines.slambased.reprojection import (
    get_map_size_in_cells,
    project2d_pcl_into_worldmap,
    reproject_local_to_global,
)


def depth2local3d(depth, fx, fy, cx, cy):
    r"""Projects depth map to 3d point cloud
    with origin in the camera focus
    """
    device = depth.device
    h, w = depth.squeeze().size()
    x = torch.linspace(0, w - 1, w).to(device)
    y = torch.linspace(0, h - 1, h).to(device)
    xv, yv = torch.meshgrid([x, y])
    dfl = depth.t().flatten()
    return torch.cat(
        [
            (dfl * (xv.flatten() - cx) / fx).unsqueeze(-1),  # x
            (dfl * (yv.flatten() - cy) / fy).unsqueeze(-1),  # y
            dfl.unsqueeze(-1),
        ],
        dim=1,
    )  # z


def pcl_to_obstacles(pts3d, map_size=40, cell_size=0.2, min_pts=10):
    r"""Counts number of 3d points in 2d map cell.
    Height is sum-pooled.
    """
    device = pts3d.device
    map_size_in_cells = get_map_size_in_cells(map_size, cell_size) - 1
    init_map = torch.zeros(
        (map_size_in_cells, map_size_in_cells), device=device
    )
    if len(pts3d) <= 1:
        return init_map
    num_pts, dim = pts3d.size()
    pts2d = torch.cat([pts3d[:, 2:3], pts3d[:, 0:1]], dim=1)
    data_idxs = torch.round(
        project2d_pcl_into_worldmap(pts2d, map_size, cell_size)
    )
    if len(data_idxs) > min_pts:
        u, counts = np.unique(
            data_idxs.detach().cpu().numpy(), axis=0, return_counts=True
        )
        init_map[u[:, 0], u[:, 1]] = torch.from_numpy(counts).to(
            dtype=torch.float32, device=device
        )
    return init_map


class DirectDepthMapper(nn.Module):
    r"""Estimates obstacle map given the depth image
    ToDo: replace numpy histogram counting with differentiable
    pytorch soft count like in
    https://papers.nips.cc/paper/7545-unsupervised-learning-of-shape-and-pose-with-differentiable-point-clouds.pdf
    """

    def __init__(
        self,
        camera_height=0,
        near_th=0.1,
        far_th=4.0,
        h_min=0.0,
        h_max=1.0,
        map_size=40,
        map_cell_size=0.1,
        device=torch.device("cpu"),  # noqa: B008
        **kwargs
    ):
        super(DirectDepthMapper, self).__init__()
        self.device = device
        self.near_th = near_th
        self.far_th = far_th
        self.h_min_th = h_min
        self.h_max_th = h_max
        self.camera_height = camera_height
        self.map_size_meters = map_size
        self.map_cell_size = map_cell_size
        return

    def forward(self, depth, pose=torch.eye(4).float()):  # noqa: B008
        self.device = depth.device
        # Works for FOV = 90 degrees
        # Should be adjusted, if FOV changed
        self.fx = float(depth.size(1)) / 2.0
        self.fy = float(depth.size(0)) / 2.0
        self.cx = int(self.fx) - 1
        self.cy = int(self.fy) - 1
        pose = pose.to(self.device)
        local_3d_pcl = depth2local3d(depth, self.fx, self.fy, self.cx, self.cy)
        idxs = (torch.abs(local_3d_pcl[:, 2]) < self.far_th) * (
            torch.abs(local_3d_pcl[:, 2]) >= self.near_th
        )
        survived_points = local_3d_pcl[idxs]
        if len(survived_points) < 20:
            map_size_in_cells = (
                get_map_size_in_cells(self.map_size_meters, self.map_cell_size)
                - 1
            )
            init_map = torch.zeros(
                (map_size_in_cells, map_size_in_cells), device=self.device
            )
            return init_map
        global_3d_pcl = reproject_local_to_global(survived_points, pose)[:, :3]
        # Because originally y looks down and from agent camera height
        global_3d_pcl[:, 1] = -global_3d_pcl[:, 1] + self.camera_height
        idxs = (global_3d_pcl[:, 1] > self.h_min_th) * (
            global_3d_pcl[:, 1] < self.h_max_th
        )
        global_3d_pcl = global_3d_pcl[idxs]
        obstacle_map = pcl_to_obstacles(
            global_3d_pcl, self.map_size_meters, self.map_cell_size
        )
        return obstacle_map
