import json

import numpy as np
import torch


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float):
    intr = np.eye(3, dtype=np.float64)
    intr[0, 0] = fx
    intr[1, 1] = fy
    intr[0, 2] = cx
    intr[1, 2] = cy
    return torch.from_numpy(intr)


with open(
    "/Users/jimmytyyang/Downloads/seqs/replicaCAD_info.json",
    "r",
    encoding="utf-8",
) as f:
    info = json.load(f)
cam_params = info["camera"]
intrinsics = intrinsics_matrix(
    cam_params["fx"], cam_params["fy"], cam_params["cx"], cam_params["cy"]
)
depth_scale = info["depth_scale"]

import pdb

pdb.set_trace()
