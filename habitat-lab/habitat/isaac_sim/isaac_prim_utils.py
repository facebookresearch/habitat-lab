# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def usd_to_habitat_rotation(rotation_wxyz):
    """
    Assume wxyz quaternions. Apply 90-degree rotation about X (from Isaac z-up to Habitat y-up) and 180-degree rotation about Y (from Isaac +z-forward to Habitat -z-forward).
    """
    w_o, x_o, y_o, z_o = rotation_wxyz

    HALF_SQRT2 = 0.70710678  # √0.5

    w_h = -HALF_SQRT2 * (y_o + z_o)
    x_h =  HALF_SQRT2 * (z_o - y_o)
    y_h =  HALF_SQRT2 * (w_o + x_o)
    z_h =  HALF_SQRT2 * (w_o - x_o)

    rotation_habitat = w_h, x_h, y_h, z_h

    return rotation_habitat


def usd_to_habitat_position(position):

    x, y, z = position
    return [-x, z, y]


def isaac_to_habitat(positions, orientations):
    """
    Convert from Isaac (Z-up) to Habitat (Y-up) coordinate system for positions and orientations.
    """

    # Positions:
    # From Isaac (Z-up) to Habitat (Y-up): Isaac (x, y, z) → Habitat (-x, z, y)
    # We can do this in a single step with array slicing:
    new_positions = positions[:, [0, 2, 1]]  # Rearrange: (x, y, z) -> (x, z, y)
    new_positions[:, 0] *= -1  # Negate the X-axis

    # Orientations:
    # Apply the fixed transform derived above
    new_orientations = apply_isaac_to_habitat_orientation(orientations)

    return new_positions, new_orientations

def habitat_to_usd_position(position):
    """
    Convert a position from Habitat (Y-up) to USD (Z-up) coordinate system.

    Habitat (-x, z, y) -> Isaac (x, y, z)
    """
    x, y, z = position
    return [-x, z, y]

def habitat_to_usd_rotation(rotation_wxyz):
    """
    Convert a quaternion rotation from Habitat to USD coordinate system.

    Parameters
    ----------
    rotation : list[float]
        Quaternion in Habitat coordinates [w, x, y, z] (wxyz).

    Returns
    -------
    list[float]
        Quaternion in USD coordinates [w, x, y, z] (wxyz).
    """
    HALF_SQRT2 = 0.70710678  # √0.5

    # Combined inverse quaternion transform: (inverse of q_trans)
    # q_x90_inv = [√0.5, √0.5, 0, 0] (wxyz format)
    # q_y180_inv = [0, 0, -1, 0] (wxyz format)
    q_y90_inv = [HALF_SQRT2, 0.0, HALF_SQRT2, 0.0]

    q_x90_inv = [HALF_SQRT2, HALF_SQRT2, 0.0, 0.0]
    q_z90_inv = [HALF_SQRT2, 0.0, 0.0, HALF_SQRT2]
    q_y180_inv = [0.0, 0.0, -1.0, 0.0]
    q_z180_inv = [0.0, 0.0, 0.0, 1.0]

    # todo: revise this to get the 180-degree rotation about y from the object_config.json

    # Multiply q_y180_inv * q_x90_inv to get the combined quaternion
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return [w, x, y, z]

    q_trans_inv = quat_multiply(q_x90_inv, q_y180_inv)

    # Multiply q_trans_inv with the input rotation quaternion
    w, x, y, z = rotation_wxyz
    rotation_usd = quat_multiply(q_trans_inv, [w, x, y, z])

    return rotation_usd


def set_translation(prim, translation):
    """Note that accessing a rigid object through USD prim API is not preferred. See RigidPrim."""

    from pxr import UsdGeom, Gf
    xformable = UsdGeom.Xformable(prim)

    translate_op = next(
        (op for op in xformable.GetOrderedXformOps() if op.GetName() == "xformOp:translate"),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3f(*translation))


def set_rotation(prim, rotation_quat_wxyz):

    from pxr import UsdGeom, Gf
    xformable = UsdGeom.Xformable(prim)

    orient_op = next(
        (op for op in xformable.GetOrderedXformOps() if op.GetName() == "xformOp:orient"),
        None,
    )
    if orient_op is None:
        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op.Set(Gf.Quatd(*rotation_quat_wxyz))    
