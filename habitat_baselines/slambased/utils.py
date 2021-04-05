#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import numpy as np
import torch
from PIL import Image


def generate_2dgrid(h, w, centered=False):
    if centered:
        x = torch.linspace(-w / 2 + 1, w / 2, w)
        y = torch.linspace(-h / 2 + 1, h / 2, h)
    else:
        x = torch.linspace(0, w - 1, w)
        y = torch.linspace(0, h - 1, h)
    grid2d = torch.stack(
        [y.repeat(w, 1).t().contiguous().view(-1), x.repeat(h)], 1
    )
    return grid2d.view(1, h, w, 2).permute(0, 3, 1, 2)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"{v} cannot be converted to a bool")


def resize_pil(np_img, size=128):
    im1 = Image.fromarray(np_img)
    im1.thumbnail((size, size))
    return np.array(im1)


def find_map_size(h, w):
    map_size_in_meters = int(0.1 * 3 * max(h, w))
    if map_size_in_meters % 10 != 0:
        map_size_in_meters = map_size_in_meters + (
            10 - (map_size_in_meters % 10)
        )
    return map_size_in_meters


def gettimestr():
    return time.strftime("%Y-%m-%d--%H_%M_%S", time.gmtime())
