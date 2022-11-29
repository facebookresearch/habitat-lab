#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List

import torch

from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseILTrainer(BaseTrainer):
    r"""Base trainer class for IL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: "DictConfig"
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: "DictConfig"):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30
        self._make_dirs()

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def _make_dirs(self) -> None:
        r"""Makes directories for log files, checkpoints & results."""
        self._make_log_dir()
        self._make_ckpt_dir()
        if self.config.habitat_baselines.il.eval_save_results:
            self._make_results_dir()

    def _make_log_dir(self) -> None:
        r"""Makes directory for writing log files."""
        if self.config.habitat_baselines.il.log_metrics and not os.path.isdir(
            self.config.habitat_baselines.il.output_log_dir
        ):
            os.makedirs(self.config.habitat_baselines.il.output_log_dir)

    def _make_ckpt_dir(self) -> None:
        r"""Makes directory for saving model checkpoints."""
        if not os.path.isdir(self.config.habitat_baselines.checkpoint_folder):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

    def _make_results_dir(self) -> None:
        r"""Makes directory for saving eval results."""
        dir_name = self.config.habitat_baselines.il.results_dir.format(
            split="val"
        )
        os.makedirs(dir_name, exist_ok=True)

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, state_dict: OrderedDict, file_name: str) -> None:
        r"""Save checkpoint with specified name.

        Args:
            state_dict: model's state_dict
            file_name: file name for checkpoint

        Returns:
            None
        """
        torch.save(
            state_dict,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, file_name
            ),
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError
