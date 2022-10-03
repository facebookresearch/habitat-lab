#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil, floor

import numpy as np
import torch


def p_zx(p):
    return p[(0, 2), 3]


def get_map_size_in_cells(map_size_in_meters, cell_size_in_meters):
    return int(ceil(map_size_in_meters / cell_size_in_meters)) + 1


def get_pos_diff(p_init, p_fin):
    return p_zx(p_fin) - p_zx(p_init)


def get_distance(p_init, p_fin):
    return torch.norm(get_pos_diff(p_init, p_fin))


def get_pos_diffs(ps):
    return ps[1:, (0, 2), 3] - ps[: (ps.size(0) - 1), (0, 2), 3]


def angle_to_pi_2_minus_pi_2(angle):
    if angle < -np.pi:
        angle = 2.0 * np.pi + angle
    if angle > np.pi:
        angle = -2.0 * np.pi + angle
    return angle


def get_direction(p_init, p_fin, ang_th=0.2, pos_th=0.1):
    pos_diff = get_pos_diff(p_init, p_fin)
    if torch.norm(pos_diff, 2).item() < pos_th:
        return 0
    else:
        needed_angle = torch.atan2(pos_diff[1], pos_diff[0])
        current_angle = torch.atan2(p_init[2, 0], p_init[0, 0])
    to_rotate = angle_to_pi_2_minus_pi_2(
        -np.pi / 2.0 + needed_angle - current_angle
    )
    if torch.abs(to_rotate).item() < ang_th:
        return 0
    return to_rotate


def reproject_local_to_global(xyz_local, p):
    device = xyz_local.device
    num, dim = xyz_local.size()
    if dim == 3:
        xyz = torch.cat(
            [
                xyz_local,
                torch.ones((num, 1), dtype=torch.float32, device=device),
            ],
            dim=1,
        )
    elif dim == 4:
        xyz = xyz_local
    else:
        raise ValueError(
            "3d point cloud dim is neighter 3, or 4 (homogeneous)"
        )
    # print(xyz.shape, P.shape)
    xyz_global = torch.mm(p.squeeze(), xyz.t())
    return xyz_global.t()


def project2d_pcl_into_worldmap(zx, map_size, cell_size):
    device = zx.device
    shift = int(floor(get_map_size_in_cells(map_size, cell_size) / 2.0))
    topdown2index = torch.tensor(
        [[1.0 / cell_size, 0, shift], [0, 1.0 / cell_size, shift], [0, 0, 1]],
        device=device,
    )
    world_coords_h = torch.cat(
        [zx.view(-1, 2), torch.ones((len(zx), 1), device=device)], dim=1
    )
    world_coords = torch.mm(topdown2index, world_coords_h.t())
    return world_coords.t()[:, :2]


def get_pose2d(poses6d):
    poses6d = poses6d.view(-1, 4, 4)
    poses2d = poses6d[:, (0, 2)]
    poses2d = poses2d[:, :, (0, 2, 3)]
    return poses2d


def get_rotation_matrix(angle_in_radians):
    angle_in_radians = angle_in_radians.view(-1, 1, 1)
    sin_a = torch.sin(angle_in_radians)
    cos_a = torch.cos(angle_in_radians)
    a1x = torch.cat([cos_a, sin_a], dim=2)
    a2x = torch.cat([-sin_a, cos_a], dim=2)
    transform = torch.cat([a1x, a2x], dim=1)
    return transform


def normalize_zx_ori(p):
    p2d = get_pose2d(p)
    norms = torch.norm(p2d[:, 0, :2], dim=1).view(-1, 1, 1)
    out = torch.cat(
        [
            torch.cat(
                [p[:, :3, :3] / norms.expand(p.size(0), 3, 3), p[:, 3:, :3]],
                dim=1,
            ),
            p[:, :, 3:],
        ],
        dim=2,
    )
    return out


def add_rot_wps(p):
    planned_tps_norm = normalize_zx_ori(p)
    pos_diffs = get_pos_diffs(planned_tps_norm)

    angles = torch.atan2(pos_diffs[:, 1], pos_diffs[:, 0])
    rotmats = get_rotation_matrix(angles)
    planned_tps_norm[: p.size(0) - 1, 0, 0] = rotmats[:, 0, 0]
    planned_tps_norm[: p.size(0) - 1, 0, 2] = rotmats[:, 0, 1]
    planned_tps_norm[: p.size(0) - 1, 2, 0] = rotmats[:, 1, 0]
    planned_tps_norm[: p.size(0) - 1, 2, 2] = rotmats[:, 1, 1]

    planned_points2 = planned_tps_norm.clone()

    planned_points2[1:, 0, 0] = planned_tps_norm[: p.size(0) - 1, 0, 0]
    planned_points2[1:, 0, 2] = planned_tps_norm[: p.size(0) - 1, 0, 2]
    planned_points2[1:, 2, 0] = planned_tps_norm[: p.size(0) - 1, 2, 0]
    planned_points2[1:, 2, 2] = planned_tps_norm[: p.size(0) - 1, 2, 2]
    out = torch.stack(
        (planned_points2.unsqueeze(0), planned_tps_norm.unsqueeze(0)), dim=0
    ).squeeze()
    out = out.permute(1, 0, 2, 3).contiguous().view(-1, 4, 4)
    return out


def planned_path2tps(path, cell_size, map_size, agent_h, add_rot=False):
    r"""Path is list of 2d coordinates from planner, in map cells.
    tp is trajectory pose, 4x4 matrix - same format,
    as in localization module
    """
    path = torch.cat(path).view(-1, 2)
    # print(path.size())
    num_pts = len(path)
    planned_tps = torch.eye(4).unsqueeze(0).repeat((num_pts, 1, 1))
    planned_tps[:, 0, 3] = path[:, 1]  # switch back x and z
    planned_tps[:, 1, 3] = agent_h
    planned_tps[:, 2, 3] = path[:, 0]  # switch back x and z
    shift = int(floor(get_map_size_in_cells(map_size, cell_size) / 2.0))
    planned_tps[:, 0, 3] = planned_tps[:, 0, 3] - shift
    planned_tps[:, 2, 3] = planned_tps[:, 2, 3] - shift
    p = torch.tensor(
        [
            [1.0 / cell_size, 0, 0, 0],
            [0, 1.0 / cell_size, 0, 0],
            [0, 0, 1.0 / cell_size, 0],
            [0, 0, 0, 1],
        ]
    )
    planned_tps = torch.bmm(
        p.inverse().unsqueeze(0).expand(num_pts, 4, 4), planned_tps
    )
    if add_rot:
        return add_rot_wps(planned_tps)
    return planned_tps


def habitat_goalpos_to_tp(ro_phi, p_curr):
    r"""Convert distance and azimuth to
    trajectory pose, 4x4 matrix - same format,
    as in localization module
    """
    device = ro_phi.device
    offset = torch.tensor(
        [
            -ro_phi[0] * torch.sin(ro_phi[1]),
            0,
            ro_phi[0] * torch.cos(ro_phi[1]),
        ]
    ).to(device)
    if p_curr.size(1) == 3:
        p_curr = homogenize_p(p_curr)
    goal_tp = torch.mm(
        p_curr.to(device),
        torch.cat(
            [
                offset
                * torch.tensor(
                    [1.0, 1.0, 1.0], dtype=torch.float32, device=device
                ),
                torch.tensor([1.0], device=device),
            ]
        ).reshape(4, 1),
    )
    return goal_tp


def habitat_goalpos_to_mapgoal_pos(offset, p_curr, cell_size, map_size):
    r"""Convert distance and azimuth to
    map cell coordinates
    """
    device = offset.device
    goal_tp = habitat_goalpos_to_tp(offset, p_curr)
    goal_tp1 = torch.eye(4).to(device)
    goal_tp1[:, 3:] = goal_tp
    projected_p = project_tps_into_worldmap(
        goal_tp1.view(1, 4, 4), cell_size, map_size
    )
    return projected_p


def homogenize_p(tps):
    device = tps.device
    tps = tps.view(-1, 3, 4)
    return torch.cat(
        [
            tps.float(),
            torch.tensor([0, 0, 0, 1.0])
            .view(1, 1, 4)
            .expand(tps.size(0), 1, 4)
            .to(device),
        ],
        dim=1,
    )


def project_tps_into_worldmap(tps, cell_size, map_size, do_floor=True):
    r"""Convert 4x4 pose matrices (trajectory poses) to
    map cell coordinates
    """
    if len(tps) == 0:
        return []
    if isinstance(tps, list):
        return []
    device = tps.device
    topdown_p = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]).to(device)
    world_coords = torch.bmm(
        topdown_p.view(1, 2, 4).expand(tps.size(0), 2, 4),
        tps[:, :, 3:].view(-1, 4, 1),
    )
    shift = int(floor(get_map_size_in_cells(map_size, cell_size) / 2.0))
    topdown2index = torch.tensor(
        [[1.0 / cell_size, 0, shift], [0, 1.0 / cell_size, shift], [0, 0, 1]]
    ).to(device)
    world_coords_h = torch.cat(
        [world_coords, torch.ones((len(world_coords), 1, 1)).to(device)], dim=1
    )
    world_coords = torch.bmm(
        topdown2index.unsqueeze(0).expand(world_coords_h.size(0), 3, 3),
        world_coords_h,
    )[:, :2, 0]
    if do_floor:
        return (
            torch.floor(world_coords.flip(1)) + 1
        )  # for having revesrve (z,x) ordering
    return world_coords.flip(1)


def project_tps_into_worldmap_numpy(tps, slam_to_world, cell_size, map_size):
    if len(tps) == 0:
        return []
    if isinstance(tps, list):
        return []
    # tps is expected in [n,4,4] format
    topdown_p = np.array([[slam_to_world, 0, 0, 0], [0, 0, slam_to_world, 0]])
    try:
        world_coords = np.matmul(
            topdown_p.reshape(1, 2, 4), tps[:, :, 3:].reshape(-1, 4, 1)
        )
    except BaseException:
        return []
    shift = int(floor(get_map_size_in_cells(map_size, cell_size) / 2.0))
    topdown2index = np.array(
        [[1.0 / cell_size, 0, shift], [0, 1.0 / cell_size, shift], [0, 0, 1]]
    )
    world_coords_h = np.concatenate(
        [world_coords, np.ones((len(world_coords), 1, 1))], axis=1
    )
    world_coords = np.matmul(topdown2index, world_coords_h)[:, :2, 0]
    return (
        world_coords[:, ::-1].astype(np.int32) + 1
    )  # for having revesrve (z,x) ordering
