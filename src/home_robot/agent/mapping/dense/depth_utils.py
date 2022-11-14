from argparse import Namespace
import itertools
import numpy as np
import torch

from . import rotation_utils as ru


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    camera_matrix = {"xc": xc, "zc": zc, "f": f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(
        torch.arange(Y_t.shape[-1], device=device),
        torch.arange(Y_t.shape[-2] - 1, -1, -1, device=device),
    )
    grid_x = grid_x.transpose(1, 0)
    grid_z = grid_z.transpose(1, 0)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (
        (grid_x[:, ::scale, ::scale] - camera_matrix.xc)
        * Y_t[:, ::scale, ::scale]
        / camera_matrix.f
    )
    Z_t = (
        (grid_z[:, ::scale, ::scale] - camera_matrix.zc)
        * Y_t[:, ::scale, ::scale]
        / camera_matrix.f
    )

    XYZ = torch.stack((X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))

    return XYZ


def transform_camera_view_t(XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([1.0, 0.0, 0.0], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(
        XYZ.reshape(-1, 3), torch.from_numpy(R).float().transpose(1, 0).to(device)
    ).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose_t(XYZ, current_pose, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0.0, 0.0, 1.0], angle=current_pose[2] - np.pi / 2.0)
    XYZ = torch.matmul(
        XYZ.reshape(-1, 3), torch.from_numpy(R).float().transpose(1, 0).to(device)
    ).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]
    return XYZ


def splat_feat_nd(init_grid, feat, coords):
    """
    Args:
        init_grid: B X nF X W X H X D X ..
        feat: B X nF X nPt
        coords: B X nDims X nPt in [-1, 1]
    Returns:
        grid: B X nF X W X H X D X ..
    """
    wts_dim = []
    pos_dim = []
    grid_dims = init_grid.shape[2:]

    B = init_grid.shape[0]
    F = init_grid.shape[1]

    n_dims = len(grid_dims)

    grid_flat = init_grid.view(B, F, -1)

    for d in range(n_dims):
        pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2
        pos_d = []
        wts_d = []

        for ix in [0, 1]:
            pos_ix = torch.floor(pos) + ix
            safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d])
            safe_ix = safe_ix.type(pos.dtype)

            wts_ix = 1 - torch.abs(pos - pos_ix)

            wts_ix = wts_ix * safe_ix
            pos_ix = pos_ix * safe_ix

            pos_d.append(pos_ix)
            wts_d.append(wts_ix)

        pos_dim.append(pos_d)
        wts_dim.append(wts_d)

    l_ix = [[0, 1] for _ in range(n_dims)]

    for ix_d in itertools.product(*l_ix):
        wts = torch.ones_like(wts_dim[0][0])
        index = torch.zeros_like(wts_dim[0][0])
        for d in range(n_dims):
            index = index * grid_dims[d] + pos_dim[d][ix_d[d]]
            wts = wts * wts_dim[d][ix_d[d]]

        index = index.long().expand(-1, F, -1)
        src = (feat * wts).to(grid_flat.dtype)
        grid_flat.scatter_add_(2, index, src)

    grid_flat = torch.round(grid_flat)

    return grid_flat.view(init_grid.shape)
