#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box

from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r""" Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
    (sorted by time of last modification).

    Args:
        checkpoint_folder: directory to look for checkpoints.
        previous_ckpt_ind: index of checkpoint last returned.

    Returns:
        return checkpoint path if (previous_ckpt_ind + 1)th checkpoint is found
        else return None.
    """
    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def image_resize_shortest_edge(img, size: int, channels_first: bool = False):
    """Resizes an img so that the shortest side is length of size.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels_first: a boolean that specifies the img is (CHW) or (NCHW)
    Returns:
        The resized array as a torch tensor.
    """
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) != 3 and len(img.shape) != 4:
        raise NotImplementedError()
    img = _to_tensor(img)
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_first:
        # NCHW
        h, w = img.shape[-2:]
    else:
        # NHWC
        h, w = img.shape[1:3]
        img = img.permute(0, 3, 1, 2)

    if w > h:
        percent = size / h
    else:
        percent = size / w
    h *= percent
    w *= percent
    h = int(h)
    w = int(w)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if not channels_first:
        img = img.permute(0, 2, 3, 1)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(img, cropx: int, cropy: int, channels_first: bool = False):
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: the size that you want the shortest edge to be resize to
        channels_first: If it's in NCHW
    Returns:
        the resized array
    """
    if channels_first:
        # NCHW
        y, x = img.shape[-2:]
    else:
        # NHWC
        y, x = img.shape[-3:-1]

    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    if channels_first:
        return img[..., starty : starty + cropy, startx : startx + cropx]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]


def apply_ppo_data_augs(
    observations: Dict[str, Any],
    resize: int,
    center_crop_size: int,
    channels_first: bool = False,
) -> Dict[str, Any]:
    for k, obs in observations.items():
        if k in ["rgb", "depth", "semantic"]:
            if resize != 0:
                obs = image_resize_shortest_edge(
                    obs, resize, channels_first=channels_first
                )
            if center_crop_size != 0:
                obs = center_crop(
                    obs,
                    center_crop_size,
                    center_crop_size,
                    channels_first=channels_first,
                )
            observations[k] = obs
    return observations


def overwrite_gym_box(box: Box, shape: tuple) -> Box:
    # if len(shape) < len(box.shape):
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low
    if not np.isscalar(low):
        low = np.min(low)  # low.flatten()[0]
    high = box.high
    if not np.isscalar(high):
        high = np.max(high)  # high.flatten()[0]
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)
