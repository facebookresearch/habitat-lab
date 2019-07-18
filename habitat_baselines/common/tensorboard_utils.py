#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
import torch


# TODO Add checks to replace DummyWriter
class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        pass

    def __getattr__(self, item):
        return lambda *args, **kwargs: None


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = DummyWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
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
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [
            torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.add_video(video_name, video_tensor, fps=fps, global_step=step_idx)


def get_tensorboard_writer(
    log_dir: str, *args, **kwargs
) -> Union[DummyWriter, TensorboardWriter]:
    r"""Get tensorboard writer if log_dir is specified, otherwise,
        return dummy writer instead.

    Args:
        log_dir: log directory path for tensorboard SummaryWriter.
        *args: additional positional args.
        **kwargs: additional keyword args.

    Returns:
        either the created tensorboard writer or a dummy writer.
    """
    if log_dir:
        return TensorboardWriter(log_dir, *args, **kwargs)
    else:
        return DummyWriter()
