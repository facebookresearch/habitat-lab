import pathlib
import random
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def to_world_frame(
    points: torch.Tensor,
    frame_rot: torch.Tensor,
    frame_pos: torch.Tensor,
    quaternions: Optional[torch.Tensor] = None,
    quat_first: bool = False,
) -> torch.Tensor:
    """Converts points and (optional) quaternions from a reference frame to world frame.

    Args:
        points (torch.Tensor): a (B, nPoints, 3)-shaped tensor with xyz.
        frame_rot (torch.Tensor): the xyzw rotation quaternion for the reference frame,
            with shape (B, 4).
        frame_pos (torch.Tensor): the xyz position of the reference frame, with
            shape (B, 3).
        quaternions (Optional[torch.Tensor], optional): (B, nPoints, 4)-shaped
            tensor with xyzw rotations to convert. Defaults to None.
        quat_first (bool): if True and quaternions is given, the return tensor
            has xyzw quaternion before xyz position. If false, xyz then xyzw.

    Returns:
        torch.Tensor: the transformed
            points and (optional) transformed quaternions.
            If only points, the return has shape
            (B, nPoints, 3). Otherwise it is (B, nPoints, 7).
    """
    assert points.ndim == 3, "points tensor must be (B, nPoints, 3)"
    assert frame_rot.ndim == 2, "object rot must be (B, 4)"
    assert frame_pos.ndim == 2, "object pos must be (B, 3)"
    points__world = (
        quat_apply(
            frame_rot[:, None].repeat(1, points.shape[1], 1),
            points,
        )
        + frame_pos[:, None]
    )
    if quaternions is None:
        return points__world

    assert quaternions.ndim == 3, "points tensor must be (B, nPoints, 4)"
    quat__world = quat_mul(
        frame_rot.unsqueeze(1).repeat(1, points.shape[1], 1), quaternions
    )
    ret_list = (
        [quat__world, points__world]
        if quat_first
        else [points__world, quat__world]
    )
    return torch.cat(ret_list, dim=-1)


def sample_point_cloud_from_urdf(
    assets_root: Union[pathlib.Path, str],
    urdf_fname: str,
    sampled_dim: int,
    noise: float = 0.0,
    seed: int = 0,
    element_filter: Optional[Tuple[str, str]] = None,
) -> np.ndarray:
    """Computes a point cloud from mesh at the given folder.

    Args:
        assets_root (Union[pathlib.Path, str]): the root folder for all assets.
        urdf_fname (str): the name of the urdf defining the mesh.
        sampled_dim (int): the number of points to sample.
        noise (float, optional): if given, Gaussian noise N(0, noise) is added.
            Defaults to 0.0.
        seed (int, optional): a seed for trimesh. Defaults to 0.
        element_filter (Optional[Tuple[str, str]], optional): if given, it describes
            (element type, element name), and the function will find the element
            in the URDF matching this filter and compute the point cloud for this
            element. Defaults to None.

    Returns:
        np.ndarray: the sampled point cloud.
    """

    assets_root = pathlib.Path(assets_root)
    urdf_path = pathlib.Path(urdf_fname)

    def read_xml(filename):
        import xml.etree.ElementTree as Et

        with open(filename, "r") as file:
            # Read the file content and strip leading whitespace
            content = file.read().lstrip()  # Parse the XML content
        root = Et.fromstring(content)
        return root

    src_urdf = read_xml(str(assets_root / urdf_fname))

    if element_filter:
        el_type, el_name = element_filter
        src_urdf = [
            x for x in src_urdf.findall(el_type) if el_name in x.get("name")
        ][0]

    col_el = src_urdf.findall(f".//collision/geometry/")[0]
    if col_el.tag == "sphere":
        # Sample points uniformly from a sphere
        # (X, Y, Z) ~ N(0, 1) ---> (X/norm, Y/norm, Z/norm) ~ U(S^2)
        # see  https://mathworld.wolfram.com/SpherePointPicking.html
        # (and Muller 1959, Marsaglia 1972).

        radius = float(col_el.attrib["radius"].split(" ")[0])
        mesh = np.zeros((sampled_dim, 3), dtype=float)
        mesh_norm = np.linalg.norm(mesh, axis=1, keepdims=True)
        while True:
            resample = mesh_norm[:, 0] < 1e-8
            if not resample.any():
                break
            mesh[resample] = np.random.normal(
                loc=0.0, scale=1.0, size=(resample.shape[0], 3)
            )
            mesh_norm = np.linalg.norm(mesh, axis=1, keepdims=True)
        mesh = radius * mesh / mesh_norm
    else:
        scale = (
            float(col_el.attrib["scale"].split(" ")[0])
            if "scale" in col_el.attrib
            else 1.0
        )
        try:
            mesh_file = urdf_path.parent / col_el.attrib["filename"]
            mesh = trimesh.load(assets_root / mesh_file, force="mesh")
        except:
            # Try from assets root directly
            mesh = trimesh.load(assets_root / col_el.attrib["filename"])
        samples, faces = trimesh.sample.sample_surface(
            mesh, sampled_dim, seed=seed
        )
        normals = mesh.face_normals[faces]
        mesh = samples * scale
    if noise > 0.0:
        mesh += np.random.normal(scale=noise, size=mesh.shape)
    return mesh, normals
