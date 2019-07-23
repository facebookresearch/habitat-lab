#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from habitat import Config
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import (
    DummyWriter,
    TensorboardWriter,
)

# TODO distribute utilities in this file to separate files


def get_trainer(trainer_name: str, trainer_cfg: Config) -> BaseTrainer:
    r"""
    Create specific trainer instance according to name.
    Args:
        trainer_name: name of registered trainer .
        trainer_cfg: config file for trainer.

    Returns:
        an instance of the specified trainer.
    """
    trainer = baseline_registry.get_trainer(trainer_name)
    assert trainer is not None, f"{trainer_name} is not supported"
    return trainer(trainer_cfg)


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


# TODO make this a  LRScheduler class
def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    r"""Decreases the learning rate linearly
    """
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def batch_obs(observations: List[Dict]) -> Dict:
    r"""
    Transpose a batch of observation dicts to a dict of batched
    observations.
    Args:
        observations:  list of dicts of observations.

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    for sensor in batch:
        batch[sensor] = torch.tensor(
            np.array(batch[sensor]), dtype=torch.float
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
    assert os.path.isdir(checkpoint_folder), "invalid checkpoint folder path"
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    config: Config,
    images: List[np.ndarray],
    episode_id: int,
    checkpoint_idx: int,
    spl: float,
    tb_writer: Union[DummyWriter, TensorboardWriter],
    fps: int = 10,
) -> None:
    r"""
    Generate video according to specified information.
    Args:
        config: config object that contains video_option and video_dir.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        spl: SPL for this episode for video naming.
        tb_writer: tensorboard writer object for uploading video
        fps: fps for generated video
    Returns:
        None
    """
    if config.video_option and len(images) > 0:
        video_name = f"episode{episode_id}_ckpt{checkpoint_idx}_spl{spl:.2f}"
        if "disk" in config.video_option:
            images_to_video(images, config.video_dir, video_name)
        if "tensorboard" in config.video_option:
            tb_writer.add_video_from_np_images(
                f"episode{episode_id}", checkpoint_idx, images, fps=fps
            )
