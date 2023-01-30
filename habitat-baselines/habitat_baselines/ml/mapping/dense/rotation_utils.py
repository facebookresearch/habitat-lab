import numpy as np
import torch
from torch.nn import functional as F

ANGLE_EPS = 0.001


def normalize(v):
    return v / np.linalg.norm(v)


def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32,
        )
        R = (
            np.eye(3)
            + np.sin(angle) * S_hat
            + (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
        )
    else:
        R = np.eye(3)
    return R


def r_between(v_from_, v_to_):
    v_from = normalize(v_from_)
    v_to = normalize(v_to_)
    ax = normalize(np.cross(v_from, v_to))
    angle = np.arccos(np.dot(v_from, v_to))
    return get_r_matrix(ax, angle)


def rotate_camera_to_point_at(up_from, lookat_from, up_to, lookat_to):
    inputs = [up_from, lookat_from, up_to, lookat_to]
    for i in range(4):
        inputs[i] = normalize(np.array(inputs[i]).reshape((-1,)))
    up_from, lookat_from, up_to, lookat_to = inputs
    r1 = r_between(lookat_from, lookat_to)

    new_x = np.dot(r1, np.array([1, 0, 0]).reshape((-1, 1))).reshape((-1))
    to_x = normalize(np.cross(lookat_to, up_to))
    angle = np.arccos(np.dot(new_x, to_x))
    if angle > ANGLE_EPS:
        if angle < np.pi - ANGLE_EPS:
            ax = normalize(np.cross(new_x, to_x))
            flip = np.dot(lookat_to, ax)
            if flip > 0:
                r2 = get_r_matrix(lookat_to, angle)
            elif flip < 0:
                r2 = get_r_matrix(lookat_to, -1.0 * angle)
        else:
            # Angle of rotation is too close to 180 degrees, direction of
            # rotation does not matter.
            r2 = get_r_matrix(lookat_to, angle)
    else:
        r2 = np.eye(3)
    return np.dot(r2, r1)


def get_grid(pose, grid_size, precision):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)

    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    t = t * np.pi / 180.0
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t, torch.zeros_like(cos_t)], 1)
    theta12 = torch.stack([sin_t, cos_t, torch.zeros_like(cos_t)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones_like(x), torch.zeros_like(x), x], 1)
    theta22 = torch.stack([torch.zeros_like(x), torch.ones_like(x), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size)).to(precision)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size)).to(precision)

    return rot_grid, trans_grid
