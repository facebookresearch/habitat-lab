#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch

from habitat_baselines.utils.common import (
    tensor_to_bgr_images,
    tensor_to_depth_images,
)

try:
    from habitat_sim.utils.common import d3_40_colors_rgb
except ImportError:
    d3_40_colors_rgb = None
    import cv2


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
    gt_bgr_o, pred_bgr = tensor_to_bgr_images([gt_rgb, pred_rgb])
    cv2.imwrite(path + "_gt.jpg", gt_bgr_o)
    cv2.imwrite(path + "_pred.jpg", pred_bgr)


def save_seg_results(
    gt_seg: torch.Tensor, pred_seg: torch.Tensor, path: str
) -> None:
    r"""For saving predicted and ground truth seg maps during
    EQA-CNN-Pretrain eval.

    Args:
        gt_seg: ground truth segmentation tensor
        pred_seg: output segmentation tensor
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
        pred_depth: output depth tensor
        path: to save images
    """
    path = path.format(split="val", type="depth")

    gt_depth, pred_depth = tensor_to_depth_images([gt_depth, pred_depth])

    cv2.imwrite(path + "_gt.jpg", gt_depth)
    cv2.imwrite(path + "_pred.jpg", pred_depth)


def put_vqa_text_on_image(
    image: np.ndarray,
    question: str,
    prediction: str,
    ground_truth: str,
) -> np.ndarray:
    r"""For writing VQA question, prediction and ground truth answer
        on image.
    Args:
        image: image on which text has to be written
        question: input question to model
        prediction: model's answer prediction
        ground_truth: ground truth answer
    Returns:
        image with text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    scale = 0.4
    thickness = 1

    cv2.putText(
        image,
        "Question: " + question,
        (10, 15),
        font,
        scale,
        color,
        thickness,
    )
    cv2.putText(
        image,
        "Prediction: " + prediction,
        (10, 30),
        font,
        scale,
        color,
        thickness,
    )
    cv2.putText(
        image,
        "Ground truth: " + ground_truth,
        (10, 45),
        font,
        scale,
        color,
        thickness,
    )

    return image


def save_vqa_image_results(
    images_tensor: torch.Tensor,
    question: str,
    prediction: str,
    ground_truth: str,
    path: str,
) -> None:
    r"""For saving VQA input images with input question and predicted answer.
    Being used to save model predictions during eval.
    Args:
        images_tensor: images' tensor containing input frames
        question: input question to model
        prediction: model's answer prediction
        ground_truth: ground truth answer
        path: to save images
    Returns:
        None
    """

    images = tensor_to_bgr_images(images_tensor)

    collage_image = cv2.hconcat(images)
    collage_image = cv2.copyMakeBorder(
        collage_image,
        55,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    image = put_vqa_text_on_image(
        collage_image, question, prediction, ground_truth
    )

    cv2.imwrite(path, image)
