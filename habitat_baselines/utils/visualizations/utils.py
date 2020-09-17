#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch

from habitat_baselines.utils.common import (
    tensor_to_bgr_images,
    tensor_to_depth_images,
)
from habitat_sim.utils.common import d3_40_colors_rgb


def save_rgb_results(
    gt_rgb: torch.Tensor, pred_rgb: torch.Tensor, path: str
) -> None:
    r"""For saving RGB reconstruction results during EQA-CNN-Pretrain eval.

    Args:
        gt_rgb: RGB ground truth tensor
        pred_rgb: RGB reconstruction tensor
        path: to save images
    """
    path = path.format(split="val", type="rgb")
    gt_bgr, pred_bgr = tensor_to_bgr_images([gt_rgb, pred_rgb])
    cv2.imwrite(path + "_gt.jpg", gt_bgr)
    cv2.imwrite(path + "_pred.jpg", pred_bgr)


def save_seg_results(
    gt_seg: torch.Tensor, pred_seg: torch.Tensor, path: str
) -> None:
    r"""For saving predicted and ground truth seg maps during
    EQA-CNN-Pretrain eval.

    Args:
        gt_seg: ground truth segmentation tensor
        pred_seg: ouput segmentation tensor
        path: to save images
    """

    path = path.format(split="val", type="seg")

    gt_seg = gt_seg.cpu().numpy() % 40
    pred_seg = torch.argmax(pred_seg, 0).cpu().numpy() % 40

    gt_seg_colored = d3_40_colors_rgb[gt_seg]
    pred_seg_colored = d3_40_colors_rgb[pred_seg]

    cv2.imwrite(path + "_gt.jpg", gt_seg_colored)
    cv2.imwrite(path + "_pred.jpg", pred_seg_colored)


def save_depth_results(
    gt_depth: torch.Tensor, pred_depth: torch.Tensor, path: str
) -> None:
    r"""For saving predicted and ground truth depth maps during
    EQA-CNN-Pretrain eval.

    Args:
        gt_depth: ground truth depth tensor
        pred_depth: ouput depth tensor
        path: to save images
    """
    path = path.format(split="val", type="depth")

    gt_depth, pred_depth = tensor_to_depth_images([gt_depth, pred_depth])

    cv2.imwrite(path + "_gt.jpg", gt_depth)
    cv2.imwrite(path + "_pred.jpg", pred_depth)
