#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from habitat.config import Config


class TensorboardWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def add_video_from_np_images(
        self,
        video_name: str,
        step_idx: int,
        images: List[np.ndarray],
        fps: int = 10,
    ) -> None:
        r"""Write video into tensorboard from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if not self.writer:
            return
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [
            torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.writer.add_video(
            video_name, video_tensor, fps=fps, global_step=step_idx
        )

    def add_config(self, config: Config) -> None:
        if self.writer:
            self.writer.add_text(
                "config", TensorboardWriter._config_to_tb_string(config, 0)
            )
            self.writer.flush()

    @staticmethod
    def _config_to_tb_string(config: Union[Any, Config], indent: int) -> str:
        """
        Args:
            config: A nested yacs Config or the contents of a yacs Config.
            indent: The indentation level of the config.
        Returns:
            A string version of the config.
        """
        if not isinstance(config, Config):
            return str(config)
        else:
            return ("\n" if indent > 0 else "") + "\n".join(
                [
                    "\t"
                    + "  " * indent
                    + f"{k}:\t{TensorboardWriter._config_to_tb_string(v, indent + 1)}"
                    for k, v in config.items()
                ]
            )
