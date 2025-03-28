import importlib
from typing import Callable

import magnum as mn
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def process_obs_img(obs):
    im = obs["third_rgb"]
    im2 = obs["articulated_agent_arm_rgb"]
    im3 = (255 * obs["articulated_agent_arm_depth"]).astype(np.uint8)
    imt = np.zeros(im.shape, dtype=np.uint8)
    imt[: im2.shape[0], : im2.shape[1], :] = im2
    imt[im2.shape[0] :, : im2.shape[1], 0] = im3[:, :, 0]
    imt[im2.shape[0] :, : im2.shape[1], 1] = im3[:, :, 0]
    imt[im2.shape[0] :, : im2.shape[1], 2] = im3[:, :, 0]
    im = np.concatenate([im, imt], 1)
    return im


def import_fn(func_name: str) -> Callable:
    args_ = func_name.split(".")
    module = importlib.import_module(".".join(args_[:-1]))
    return getattr(module, args_[-1])


def create_T_matrix(pos, rot, use_rotvec=False):
    T_mat = np.eye(4)
    # check dtype if it is magnum quaternion
    if isinstance(rot, mn.Quaternion):
        rot_quat = R.from_quat(np.array([rot.scalar, *rot.vector]))
    elif isinstance(rot, np.ndarray):
        # check if two dim or one dim
        if rot.ndim == 2:
            rot_shape = rot.shape[-1]
        else:
            rot_shape = rot.shape
        if rot_shape == (4,):
            rot_quat = R.from_quat(rot)
        elif rot_shape == (3,):
            if use_rotvec:
                rot_quat = R.from_rotvec(rot)
            else:
                rot_quat = R.from_euler("xyz", rot)

    T_mat[:3, :3] = rot_quat.as_matrix()
    T_mat[:, -1] = np.array([*pos, 1])
    return T_mat
