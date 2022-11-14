import numpy as np
import torch
import quaternion


def get_l2_distance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_pose(position, rotation):
    x = -position[2]
    y = -position[0]
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2
    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1
    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change
    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.0:
        o -= 360.0
    return x, y, o


def get_new_pose_batch(pose, rel_pose_change):
    const = 57.29577951308232
    pose[:, 1] += rel_pose_change[:, 0] * torch.sin(
        pose[:, 2] / const
    ) + rel_pose_change[:, 1] * torch.cos(pose[:, 2] / const)
    pose[:, 0] += rel_pose_change[:, 0] * torch.cos(
        pose[:, 2] / const
    ) - rel_pose_change[:, 1] * torch.sin(pose[:, 2] / const)
    pose[:, 2] += rel_pose_change[:, 2] * const
    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0
    return pose


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords
