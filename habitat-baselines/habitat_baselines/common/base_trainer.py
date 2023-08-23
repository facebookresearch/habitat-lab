#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import TYPE_CHECKING, ClassVar, Dict, List

import torch

from habitat import logger
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.ddp_utils import (
    SAVE_STATE,
    add_signal_handlers,
    is_slurm_batch_job,
    load_resume_state,
    save_resume_state,
)
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """
    config: "DictConfig"
    flush_secs: float
    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def _get_resume_state_config_or_new_config(
        self, resume_state_config: "DictConfig"
    ):
        if self.config.habitat_baselines.load_resume_state_config:
            if self.config != resume_state_config:
                logger.warning(
                    "\n##################\n"
                    "You are attempting to resume training with a different "
                    "configuration than the one used for the original training run. "
                    "Since load_resume_state_config=True, the ORIGINAL configuration "
                    "will be used and the new configuration will be IGNORED."
                    "##################\n"
                )
            return resume_state_config
        return self.config.copy()

    def _add_preemption_signal_handlers(self):
        if is_slurm_batch_job():
            add_signal_handlers()

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """

        self._add_preemption_signal_handlers()

        resume_state = load_resume_state(self.config, filename_key="eval")
        if resume_state is not None:
            # If we have a resume state saved, that means
            # we are resuming an evaluation session that got
            # preempted. We grab the config and the prev_ckpt_ind
            # so that we pick-up from the checkpoint we left off with
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )
            prev_ckpt_ind = resume_state["prev_ckpt_ind"]
        else:
            prev_ckpt_ind = -1

        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.habitat_baselines.eval.video_option:
            assert (
                len(self.config.habitat_baselines.tensorboard_dir) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(
                self.config.habitat_baselines.tensorboard_dir, exist_ok=True
            )
        if "disk" in self.config.habitat_baselines.eval.video_option:
            assert (
                len(self.config.habitat_baselines.video_dir) > 0
            ), "Must specify a directory for storing videos on disk"

        with get_writer(self.config, flush_secs=self.flush_secs) as writer:
            if (
                os.path.isfile(
                    self.config.habitat_baselines.eval_ckpt_path_dir
                )
                or not self.config.habitat_baselines.eval.should_load_ckpt
            ):
                # evaluate single checkpoint. If `should_load_ckpt=False` then
                # the `eval_ckpt_path_dir` will be ignored.

                if self.config.habitat_baselines.eval.should_load_ckpt:
                    proposed_index = get_checkpoint_id(
                        self.config.habitat_baselines.eval_ckpt_path_dir
                    )
                else:
                    proposed_index = None

                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.habitat_baselines.eval_ckpt_path_dir,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.config.habitat_baselines.eval_ckpt_path_dir,
                            prev_ckpt_ind,
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")  # type: ignore
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

                    # We save a resume state during evaluation so that
                    # we can resume evaluating incase the job gets
                    # preempted.
                    save_resume_state(
                        {
                            "config": self.config,
                            "prev_ckpt_ind": prev_ckpt_ind,
                        },
                        self.config,
                        filename_key="eval",
                    )

                    if (
                        prev_ckpt_ind + 1
                    ) == self.config.habitat_baselines.num_checkpoints:
                        break

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device  # type: ignore
    config: "DictConfig"
    video_option: List[str]
    num_updates_done: int
    num_steps_done: int
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self, config: "DictConfig") -> None:
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30
        self.num_updates_done = 0
        self.num_steps_done = 0
        self._last_checkpoint_percent = -1.0

        if (
            config.habitat_baselines.num_updates != -1
            and config.habitat_baselines.total_num_steps != -1
        ):
            raise RuntimeError(
                "num_updates and total_num_steps are both specified.  One must be -1.\n"
                " num_updates: {} total_num_steps: {}".format(
                    config.habitat_baselines.num_updates,
                    config.habitat_baselines.total_num_steps,
                )
            )

        if (
            config.habitat_baselines.num_updates == -1
            and config.habitat_baselines.total_num_steps == -1
        ):
            raise RuntimeError(
                "One of num_updates and total_num_steps must be specified.\n"
                " num_updates: {} total_num_steps: {}".format(
                    config.habitat_baselines.num_updates,
                    config.habitat_baselines.total_num_steps,
                )
            )

        if (
            config.habitat_baselines.num_checkpoints != -1
            and config.habitat_baselines.checkpoint_interval != -1
        ):
            raise RuntimeError(
                "num_checkpoints and checkpoint_interval are both specified."
                "  One must be -1.\n"
                " num_checkpoints: {} checkpoint_interval: {}".format(
                    config.habitat_baselines.num_checkpoints,
                    config.habitat_baselines.checkpoint_interval,
                )
            )

        if (
            config.habitat_baselines.num_checkpoints == -1
            and config.habitat_baselines.checkpoint_interval == -1
        ):
            raise RuntimeError(
                "One of num_checkpoints and checkpoint_interval must be specified"
                " num_checkpoints: {} checkpoint_interval: {}".format(
                    config.habitat_baselines.num_checkpoints,
                    config.habitat_baselines.checkpoint_interval,
                )
            )

    def percent_done(self) -> float:
        if self.config.habitat_baselines.num_updates != -1:
            return (
                self.num_updates_done
                / self.config.habitat_baselines.num_updates
            )
        else:
            return (
                self.num_steps_done
                / self.config.habitat_baselines.total_num_steps
            )

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.config.habitat_baselines.num_checkpoints != -1:
            checkpoint_every = (
                1 / self.config.habitat_baselines.num_checkpoints
            )
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_updates_done
                % self.config.habitat_baselines.checkpoint_interval
            ) == 0

        return needs_checkpoint

    def _should_save_resume_state(self) -> bool:
        return SAVE_STATE.is_set() or (
            (
                not self.config.habitat_baselines.rl.preemption.save_state_batch_only
                or is_slurm_batch_job()
            )
            and (
                (
                    int(self.num_updates_done + 1)
                    % self.config.habitat_baselines.rl.preemption.save_resume_state_interval
                )
                == 0
            )
        )

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

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

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError
