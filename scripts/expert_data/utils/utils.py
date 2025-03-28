import importlib
from typing import Callable

import numpy as np


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
