# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import functools
from typing import Optional, Tuple
from warnings import warn

import torch
import torch.nn.functional as F

"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def xyzw_to_wxyz(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [quaternions[..., None, -1], quaternions[..., :3]], dim=-1
    )


def wxyz_to_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.cat([quaternions[..., 1:], quaternions[..., None, 0]], dim=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4). as w,x,y,z
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(
    euler_angles: torch.Tensor, convention: str
) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(
        _axis_angle_rotation, convention, torch.unbind(euler_angles, -1)
    )
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(
    axis: str,
    other_axis: str,
    data: torch.Tensor,
    horizontal: bool,
    tait_bryan: bool,
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError(f"Invalid letter {letter}")


def matrix_to_euler_angles(
    matrix: torch.Tensor, convention: str
) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn(
        (n, 4), dtype=dtype, device=device, requires_grad=requires_grad
    )
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
        requires_grad: Whether the resulting tensor should have the gradient
            flag set

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(
    quaternion: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_and_d_to_pris_matrix(
    axis: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """
    Creates a 4x4 matrix that represents a translation along an axis of a distance d
    Works with any number of batch dimensions.

    Args:
        axis: [..., 3]
        d: [ ...]

    Returns: [..., 4, 4]

    """
    batch_axes = axis.shape[:-1]
    mat33 = torch.eye(3).to(axis).expand(*batch_axes, 3, 3)
    pos = axis * d.unsqueeze(-1)
    mat44 = torch.cat((mat33, pos.unsqueeze(-1)), -1)
    mat44 = torch.cat(
        (
            mat44,
            torch.tensor([0.0, 0.0, 0.0, 1.0])
            .expand(*batch_axes, 1, 4)
            .to(axis),
        ),
        -2,
    )
    return mat44


def axis_and_angle_to_matrix_44(
    axis: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """
    Creates a 4x4 matrix that represents a rotation around an axis by an angle theta.
    Works with any number of batch dimensions.

    Args:
        axis: [..., 3]
        theta: [ ...]

    Returns: [..., 4, 4]

    """
    rot = axis_and_angle_to_matrix_33(axis, theta)
    batch_shape = axis.shape[:-1]
    mat44 = torch.cat((rot, torch.zeros(*batch_shape, 3, 1).to(axis)), -1)
    mat44 = torch.cat(
        (
            mat44,
            torch.tensor([0.0, 0.0, 0.0, 1.0])
            .expand(*batch_shape, 1, 4)
            .to(axis),
        ),
        -2,
    )
    return mat44


def fast_axisangle_to_matrix(ax_ang: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(ax_ang, p=2, dim=1, keepdim=True)  # L2 norm along rows
    axis = ax_ang / (angle + 1e-8)
    return axis_and_angle_to_matrix_33(axis, angle[:, 0])


def axis_and_angle_to_matrix_33(
    axis: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """
    Creates a 3x3 matrix that represents a rotation around an axis by an angle theta.
    Works with any number of batch dimensions.

    Argsaxis.sh:
        axis: [..., 3]
        theta: [ ...]

    Returns: [..., 3, 3]

    """
    # based on https://ai.stackexchange.com/questions/14041/, and checked against wikipedia
    c = torch.cos(
        theta
    )  # NOTE: cos is not that precise for float32, you may want to use float64
    one_minus_c = 1 - c
    s = torch.sin(theta)
    kx, ky, kz = torch.unbind(axis, -1)
    r00 = c + kx * kx * one_minus_c
    r01 = kx * ky * one_minus_c - kz * s
    r02 = kx * kz * one_minus_c + ky * s
    r10 = ky * kx * one_minus_c + kz * s
    r11 = c + ky * ky * one_minus_c
    r12 = ky * kz * one_minus_c - kx * s
    r20 = kz * kx * one_minus_c - ky * s
    r21 = kz * ky * one_minus_c + kx * s
    r22 = c + kz * kz * one_minus_c
    rot = torch.stack(
        [
            torch.stack([r00, r01, r02], -1),
            torch.stack([r10, r11, r12], -1),
            torch.stack([r20, r21, r22], -1),
        ],
        -2,
    )
    return rot


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.
    This uses quaternions as an intermediate representation,
    and is a little slower than the other version

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    warn(
        "This is deprecated because it is slow. Use axis_and_angle_to_matrix_33 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    axis_angle =  torch.tensor(axis_angle,dtype=torch.float32) if not isinstance(axis_angle,torch.Tensor) else axis_angle
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def matrix44_to_se3_9d(matrix: torch.Tensor) -> torch.Tensor:
    r = matrix_to_rotation_6d(matrix[..., :3, :3])
    t = matrix[..., :3, 3]
    return torch.cat([r, t], dim=-1)


def se3_9d_to_matrix44(se3: torch.Tensor) -> torch.Tensor:
    r = rotation_6d_to_matrix(se3[..., :6])
    t = se3[..., 6:]
    H = torch.eye(4, device=r.device, dtype=r.dtype).repeat(
        r.shape[:-2] + (1, 1)
    )
    H[..., :3, :3] = r
    H[..., :3, 3] = t
    return H


def matrix_to_pos_rot(m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 4x4 transformation matrix to (position, xyzw quatnerion) used by pybullet and RViz"""
    pos = m[..., :3, 3]
    rot = matrix_to_quaternion(m[..., :3, :3])
    rot = wxyz_to_xyzw(rot)
    return pos, rot


def pos_rot_to_matrix(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    rot = xyzw_to_wxyz(rot)
    rot = quaternion_to_matrix(rot)
    m = torch.eye(4, device=pos.device, dtype=pos.dtype).repeat(
        pos.shape[:-1] + (1, 1)
    )
    m[..., :3, 3] = pos
    m[..., :3, :3] = rot
    return m


# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = {v: k for k, v in _AXES2TUPLE.items()}


# def quaternion_from_euler(rpy: torch.Tensor, axes: str = "sxyz") -> torch.Tensor:
#     """
#     Return quaternion from Euler angles and axis sequence.
#     Taken from
#     https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py#L1238

#     ai, aj, ak : Euler's roll, pitch and yaw angles
#     axes : One of 24 axis sequences as string or encoded tuple

#     """
#     try:
#         firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
#     except (AttributeError, KeyError):
#         _TUPLE2AXES[axes]  # noqa: validation
#         firstaxis, parity, repetition, frame = axes

#     ai, aj, ak = torch.unbind(rpy, -1)
#     i = firstaxis + 1
#     j = _NEXT_AXIS[i + parity - 1] + 1
#     k = _NEXT_AXIS[i - parity] + 1

#     if frame:
#         ai, ak = ak, ai
#     if parity:
#         aj = -aj

#     ai /= 2.0
#     aj /= 2.0
#     ak /= 2.0
#     ci = torch.cos(ai)
#     si = torch.sin(ai)
#     cj = torch.cos(aj)
#     sj = torch.sin(aj)
#     ck = torch.cos(ak)
#     sk = torch.sin(ak)
#     cc = ci * ck
#     cs = ci * sk
#     sc = si * ck
#     ss = si * sk

#     q = torch.zeros([*rpy.shape[:-1], 4]).to(rpy)
#     if repetition:
#         q[..., 0] = cj * (cc - ss)
#         q[..., i] = cj * (cs + sc)
#         q[..., j] = sj * (cc + ss)
#         q[..., k] = sj * (cs - sc)
#     else:
#         q[..., 0] = cj * cc + sj * ss
#         q[..., i] = cj * sc - sj * cs
#         q[..., j] = cj * ss + sj * cc
#         q[..., k] = cj * cs - sj * sc
#     if parity:
#         q[..., j] *= -1.0

#     return q
