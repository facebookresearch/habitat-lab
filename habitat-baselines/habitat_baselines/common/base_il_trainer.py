#!/usr/bin/env python3

import os
from collections import OrderedDict
from typing import Dict, List

import torch

from habitat import Config
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter


class BaseILTrainer(BaseTrainer):
    r"""Base trainer class for IL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config):
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
        if self.config.EVAL_SAVE_RESULTS:
            self._make_results_dir()

    def _make_log_dir(self) -> None:
        r"""Makes directory for writing log files."""
        if self.config.LOG_METRICS and not os.path.isdir(
            self.config.OUTPUT_LOG_DIR
        ):
            os.makedirs(self.config.OUTPUT_LOG_DIR)

    def _make_ckpt_dir(self) -> None:
        r"""Makes directory for saving model checkpoints."""
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

    def _make_results_dir(self) -> None:
        r"""Makes directory for saving eval results."""
        dir_name = self.config.RESULTS_DIR.format(split="val")
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
            state_dict, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError
