#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.nn import functional as F

from habitat_baselines.slambased.utils import generate_2dgrid


def safe_roi_2d(array2d, ymin, ymax, xmin, xmax):
    (h, w) = array2d.shape
    return max(0, ymin), min(ymax, h), max(0, xmin), min(xmax, w)


def f2ind(ten, i):
    # Float to index
    return torch.round(ten[i]).long()


def init_neights_to_channels(ks=3):
    r"""Convolutional kernel,
    which maps nighborhood into channels
    """
    weights = np.zeros((ks * ks, 1, ks, ks), dtype=np.float32)
    for y in range(ks):
        for x in range(ks):
            weights[x * ks + y, 0, y, x] = 1.0
    return weights


class SoftArgMin(nn.Module):
    def __init__(self, beta=5):
        super(SoftArgMin, self).__init__()
        self.beta = beta
        return

    def forward(self, x, coords2d=None):
        bx_sm = F.softmax(self.beta * (-x).view(1, -1), dim=1)
        if coords2d is None:
            coords2d = generate_2dgrid(x.size(2), x.size(3), False)
        coords2d_flat = coords2d.view(2, -1)
        return (bx_sm.expand_as(coords2d_flat) * coords2d_flat).sum(
            dim=1
        ) / bx_sm.sum(dim=1)


class HardArgMin(nn.Module):
    def __init__(self):
        super(HardArgMin, self).__init__()
        return

    def forward(self, x, coords2d=None):
        val, idx = x.view(-1).min(dim=0)
        if coords2d is None:
            coords2d = generate_2dgrid(x.size(2), x.size(3), False)
        coords2d_flat = coords2d.view(2, -1)
        return coords2d_flat[:, idx].view(2)


class DifferentiableStarPlanner(nn.Module):
    def __init__(
        self,
        max_steps=500,
        visualize=False,
        preprocess=False,
        beta=100,
        connectivity="eight",
        device=torch.device("cpu"),  # noqa: B008
        **kwargs
    ):
        super(DifferentiableStarPlanner, self).__init__()
        self.eps = 1e-12
        self.max_steps = max_steps
        self.visualize = visualize
        self.inf = 1e7
        self.ob_cost = 10000.0
        self.device = device
        self.beta = beta
        self.preprocess = preprocess
        # self.argmin = SoftArgMin(beta)
        self.argmin = HardArgMin()
        self.neights2channels = nn.Conv2d(1, 9, kernel_size=(3, 3), bias=False)
        self.neights2channels.weight.data = torch.from_numpy(
            init_neights_to_channels(3)
        )
        self.neights2channels.to(device)
        self.preprocessNet = nn.Conv2d(
            1, 1, kernel_size=(3, 3), padding=1, bias=False
        )
        self.preprocessNet.weight.data = torch.from_numpy(
            np.array(
                [
                    [
                        [
                            [0.00001, 0.0001, 0.00001],
                            [0.0001, 1, 0.0001],
                            [0.00001, 0.0001, 0.00001],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        )
        self.preprocessNet.to(device)
        if connectivity == "eight":
            self.gx_to_right = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
            self.gx_to_right.weight.data = torch.from_numpy(
                np.array([[[[0, 1, -1]]]], dtype=np.float32)
            )
            self.gx_to_right.to(device)

            self.gx_to_left = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
            self.gx_to_left.weight.data = torch.from_numpy(
                np.array([[[[-1, 1, 0]]]], dtype=np.float32)
            )
            self.gx_to_left.to(device)

            self.gy_to_up = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
            self.gy_to_up.weight.data = torch.from_numpy(
                np.array([[[[0], [1], [-1]]]], dtype=np.float32)
            )
            self.gy_to_up.to(device)

            self.gy_to_down = nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
            self.gy_to_down.weight.data = torch.from_numpy(
                np.array([[[[-1], [1], [0]]]], dtype=np.float32)
            )
            self.gy_to_down.to(device)
        else:
            raise ValueError('Only "eight" connectivity now supported')
        return

    def preprocess_obstacle_map(self, obstacle_map):
        if self.preprocess:
            return self.preprocessNet(obstacle_map)
        return obstacle_map

    def coords2grid(self, node_coords, h, w):
        grid = node_coords.squeeze() - torch.FloatTensor(
            (h / 2.0, w / 2.0)
        ).to(self.device)
        grid = grid / torch.FloatTensor((h / 2.0, w / 2.0)).to(self.device)
        return grid.view(1, 1, 1, 2).flip(3)

    def init_closelistmap(self):
        return torch.zeros_like(self.start_map).float()

    def init_openlistmap(self):
        return self.start_map.clone()

    def init_g_map(self):
        return torch.clamp(
            self.inf
            * (torch.ones_like(self.start_map) - self.start_map.clone()),
            min=0,
            max=self.inf,
        )

    def safe_roi_2d(self, ymin, ymax, xmin, xmax):
        return (
            int(max(0, torch.round(ymin).item())),
            int(min(torch.round(ymax).item(), self.height)),
            int(max(0, torch.round(xmin).item())),
            int(min(torch.round(xmax).item(), self.width)),
        )

    def forward(
        self,
        obstacles,
        coords,
        start_map,
        goal_map,
        non_obstacle_cost_map=None,
        additional_steps=50,
        return_path=True,
    ):
        self.trav_init_time = 0
        self.trav_mask_time = 0
        self.trav_soft_time = 0
        self.conv_time = 0
        self.close_time = 0

        self.obstacles = self.preprocess_obstacle_map(
            obstacles.to(self.device)
        )
        self.start_map = start_map.to(self.device)
        self.been_there = torch.zeros_like(self.start_map).to(
            torch.device("cpu")
        )
        self.coords = coords.to(self.device)
        self.goal_map = goal_map.to(self.device)
        self.been_there = torch.zeros_like(self.goal_map).to(self.device)
        self.height = obstacles.size(2)
        self.width = obstacles.size(3)
        m, goal_idx = torch.max(self.goal_map.view(-1), 0)
        c_map = self.calculate_local_path_costs(non_obstacle_cost_map)
        # c_map might be non persistent in map update
        self.g_map = self.init_g_map()
        self.close_list_map = self.init_closelistmap()
        self.open_list_map = self.init_openlistmap()
        not_done = False
        step = 0
        stopped_by_max_iter = False
        if self.visualize:
            self.fig, self.ax = plt.subplots(1, 1)
            self.image = self.ax.imshow(
                self.g_map.squeeze().cpu().detach().numpy().astype(np.float32),
                animated=True,
            )
            self.fig.canvas.draw()
        not_done = (self.close_list_map.view(-1)[goal_idx].item() < 1.0) or (
            self.g_map.view(-1)[goal_idx].item() >= 0.9 * self.ob_cost
        )
        rad = 1
        self.start_coords = (
            (self.coords * self.start_map.expand_as(self.coords))
            .sum(dim=2)
            .sum(dim=2)
            .squeeze()
        )
        node_coords = self.start_coords
        self.goal_coords = (
            (self.coords * self.goal_map.expand_as(self.coords))
            .sum(dim=2)
            .sum(dim=2)
            .squeeze()
        )
        self.max_steps = 4 * int(
            torch.sqrt(
                ((self.start_coords - self.goal_coords) ** 2).sum() + 1e-6
            ).item()
        )
        while not_done:
            ymin, ymax, xmin, xmax = self.safe_roi_2d(
                node_coords[0] - rad,
                node_coords[0] + rad + 1,
                node_coords[1] - rad,
                node_coords[1] + rad + 1,
            )
            if (
                (ymin - 1 > 0)
                and (xmin - 1 > 0)
                and (ymax + 1 < self.height)
                and (xmax + 1 < self.width)
            ):
                n2c = self.neights2channels(
                    self.g_map[:, :, ymin - 1 : ymax + 1, xmin - 1 : xmax + 1]
                )
                self.g_map[:, :, ymin:ymax, xmin:xmax] = torch.min(
                    self.g_map[:, :, ymin:ymax, xmin:xmax].clone(),
                    (n2c + c_map[:, :, ymin:ymax, xmin:xmax]).min(
                        dim=1, keepdim=True
                    )[0],
                )
                self.close_list_map[:, :, ymin:ymax, xmin:xmax] = torch.max(
                    self.close_list_map[:, :, ymin:ymax, xmin:xmax],
                    self.open_list_map[:, :, ymin:ymax, xmin:xmax],
                )
                self.open_list_map[:, :, ymin:ymax, xmin:xmax] = F.relu(
                    F.max_pool2d(
                        self.open_list_map[
                            :, :, ymin - 1 : ymax + 1, xmin - 1 : xmax + 1
                        ],
                        3,
                        stride=1,
                        padding=0,
                    )
                    - self.close_list_map[:, :, ymin:ymax, xmin:xmax]
                    - self.obstacles[:, :, ymin:ymax, xmin:xmax]
                )
            else:
                self.g_map = torch.min(
                    self.g_map,
                    (
                        self.neights2channels(
                            F.pad(self.g_map, (1, 1, 1, 1), "replicate")
                        )
                        + c_map
                    ).min(dim=1, keepdim=True)[0],
                )
                self.close_list_map = torch.max(
                    self.close_list_map, self.open_list_map
                )
                self.open_list_map = F.relu(
                    F.max_pool2d(self.open_list_map, 3, stride=1, padding=1)
                    - self.close_list_map
                    - self.obstacles
                )
            step += 1
            if step >= self.max_steps:
                stopped_by_max_iter = True
                break
            not_done = (
                self.close_list_map.view(-1)[goal_idx].item() < 1.0
            ) or (self.g_map.view(-1)[goal_idx].item() >= 0.1 * self.inf)
            rad += 1
        if not stopped_by_max_iter:
            for _ in range(additional_steps):
                # now propagating beyong start point
                self.g_map = torch.min(
                    self.g_map,
                    (
                        self.neights2channels(
                            F.pad(self.g_map, (1, 1, 1, 1), "replicate")
                        )
                        + c_map
                    ).min(dim=1, keepdim=True)[0],
                )
                self.close_list_map = torch.max(
                    self.close_list_map, self.open_list_map
                )
                self.open_list_map = F.relu(
                    F.max_pool2d(self.open_list_map, 3, stride=1, padding=1)
                    - self.close_list_map
                    - self.obstacles
                )
        if return_path:
            out_path, cost = self.reconstruct_path()
            return out_path, cost
        return None

    def calculate_local_path_costs(self, non_obstacle_cost_map=None):
        coords = self.coords
        h = coords.size(2)
        w = coords.size(3)
        obstacles_pd = F.pad(self.obstacles, (1, 1, 1, 1), "replicate")
        if non_obstacle_cost_map is None:
            learned_bias = torch.ones_like(self.obstacles).to(
                obstacles_pd.device
            )
        else:
            learned_bias = non_obstacle_cost_map.to(obstacles_pd.device)
        left_diff_sq = (
            self.gx_to_left(
                F.pad(coords[:, 1:2, :, :], (1, 1, 0, 0), "replicate")
            )
            ** 2
        )
        right_diff_sq = (
            self.gx_to_right(
                F.pad(coords[:, 1:2, :, :], (1, 1, 0, 0), "replicate")
            )
            ** 2
        )
        up_diff_sq = (
            self.gy_to_up(
                F.pad(coords[:, 0:1, :, :], (0, 0, 1, 1), "replicate")
            )
            ** 2
        )
        down_diff_sq = (
            self.gy_to_down(
                F.pad(coords[:, 0:1, :, :], (0, 0, 1, 1), "replicate")
            )
            ** 2
        )
        out = torch.cat(
            [
                # Order in from up to down, from left to right
                # hopefully same as in PyTorch
                torch.sqrt(left_diff_sq + up_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 0:h, 0:w],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                torch.sqrt(left_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 0:h, 1 : w + 1],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                torch.sqrt(left_diff_sq + down_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 2 : h + 2, 0:w],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                torch.sqrt(up_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 0:h, 1 : w + 1],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                0 * right_diff_sq
                + self.ob_cost
                * obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],  # current center
                torch.sqrt(down_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 2 : h + 2, 1 : w + 1],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                torch.sqrt(right_diff_sq + up_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 0:h, 2 : w + 2],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                torch.sqrt(right_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 1 : h + 1, 2 : w + 2],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
                torch.sqrt(right_diff_sq + down_diff_sq + self.eps)
                + self.ob_cost
                * torch.max(
                    obstacles_pd[:, :, 2 : h + 2, 2 : w + 2],
                    obstacles_pd[:, :, 1 : h + 1, 1 : w + 1],
                ),
            ],
            dim=1,
        )
        return out + torch.clamp(
            learned_bias.expand_as(out), min=0, max=self.ob_cost
        )

    def propagate_traversal(self, node_coords, close, g, coords):
        ymin, ymax, xmin, xmax = self.safe_roi_2d(
            node_coords[0] - 1,
            node_coords[0] + 2,
            node_coords[1] - 1,
            node_coords[1] + 2,
        )
        mask = close[:, :, ymin:ymax, xmin:xmax] > 0
        mask[
            :, :, f2ind(node_coords, 0) - ymin, f2ind(node_coords, 1) - xmin
        ] = 0
        mask = mask > 0
        current_g_cost = g[:, :, ymin:ymax, xmin:xmax][mask].clone()
        if len(current_g_cost.view(-1)) == 0:
            # we are kind surrounded by obstacles,
            # but still need to output something
            mask = torch.relu(
                1.0 - self.been_there[:, :, ymin:ymax, xmin:xmax]
            )
            mask[
                :,
                :,
                f2ind(node_coords, 0) - ymin,
                f2ind(node_coords, 1) - xmin,
            ] = 0
            mask = mask > 0
            current_g_cost = g[:, :, ymin:ymax, xmin:xmax][mask].clone()
        if len(current_g_cost.view(-1)) > 1:
            current_g_cost = current_g_cost - torch.min(current_g_cost).item()
            current_g_cost = current_g_cost + 0.41 * torch.randperm(
                len(current_g_cost),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ) / (len(current_g_cost))
        #
        coords_roi = coords[:, :, ymin:ymax, xmin:xmax]
        out = self.argmin(
            current_g_cost, coords_roi[mask.expand_as(coords_roi)]
        )
        return out

    def get_clean_costmap_and_goodmask(self):
        good_mask = 1 - F.max_pool2d(self.obstacles, 3, stride=1, padding=1)
        costmap = self.g_map
        obstacle_cost_corrected = 10000.0
        sampling_map = torch.clamp(costmap, min=0, max=obstacle_cost_corrected)
        return sampling_map, good_mask

    def reconstruct_path(self):
        out_path = []
        goal_coords = self.goal_coords.cpu()
        start_coords = self.start_coords.cpu()

        cost = self.g_map[:, :, f2ind(goal_coords, 0), f2ind(goal_coords, 1)]
        # Traversing
        done = False
        node_coords = goal_coords.cpu()
        out_path.append(node_coords)
        self.been_there = 0 * self.been_there.cpu()
        self.been_there[
            :, :, f2ind(node_coords, 0), f2ind(node_coords, 1)
        ] = 1.0
        self.close_list_map = self.close_list_map.cpu()
        self.g_map = self.g_map.cpu()
        self.coords = self.coords.cpu()
        count1 = 0
        while not done:
            node_coords = self.propagate_traversal(
                node_coords, self.close_list_map, self.g_map, self.coords
            )
            self.been_there[
                :, :, f2ind(node_coords, 0), f2ind(node_coords, 1)
            ] = 1.0
            if torch.norm(node_coords - out_path[-1], 2).item() < 0.3:
                y = node_coords.flatten()[0].long()
                x = node_coords.flatten()[1].long()
                print(self.g_map[0, 0, y - 2 : y + 3, x - 2 : x + 3])
                print("loop in out_path", node_coords)
                raise ValueError("loop in out_path")
            out_path.append(node_coords)
            done = torch.norm(node_coords - start_coords.cpu(), 2).item() < 0.3
            count1 += 1
            if count1 > 250:
                break
        return out_path, cost
