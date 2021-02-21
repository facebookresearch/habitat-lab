#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import re
import shutil
import tarfile
from collections import defaultdict
from io import BytesIO
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import attr
import numpy as np
import torch
from gym.spaces import Box
from PIL import Image
from torch import Size, Tensor
from torch import nn as nn

from habitat import logger
from habitat.core.dataset import Episode
from habitat.core.utils import try_cv2_import
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

cv2 = try_cv2_import()


class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
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
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


@attr.s(auto_attribs=True, slots=True)
class ObservationBatchingCache:
    r"""Helper for batching observations that maintains a cpu-side tensor
    that is the right size and is pinned to cuda memory
    """
    _pool: Dict[Any, torch.Tensor] = attr.Factory(dict)

    def get(
        self,
        num_obs: int,
        sensor_name: str,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """
        key = (
            num_obs,
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            return self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            cache = cache.pin_memory()

        self._pool[key] = cache
        return cache


@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
    cache: Optional[ObservationBatchingCache] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
        cache: An ObservationBatchingCache.  This enables faster
            stacking of observations and cpu-gpu transfer as it
            maintains a correctly sized tensor for the batched
            observations that is pinned to cuda memory.

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch_t: TensorDict = TensorDict()
    if cache is None:
        batch: DefaultDict[str, List] = defaultdict(list)

    for i, obs in enumerate(observations):
        for sensor_name, sensor in obs.items():
            sensor = torch.as_tensor(sensor)
            if cache is None:
                batch[sensor_name].append(sensor)
            else:
                if sensor_name not in batch_t:
                    batch_t[sensor_name] = cache.get(
                        len(observations), sensor_name, sensor, device
                    )

                batch_t[sensor_name][i].copy_(sensor)

    if cache is None:
        for sensor in batch:
            batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device, non_blocking=True))


def get_checkpoint_id(ckpt_path: str) -> Optional[int]:
    r"""Attempts to extract the ckpt_id from the filename of a checkpoint.
    Assumes structure of ckpt.ID.path .

    Args:
        ckpt_path: the path to the ckpt file

    Returns:
        returns an int if it is able to extract the ckpt_path else None
    """
    ckpt_path = os.path.basename(ckpt_path)
    nums: List[int] = [int(s) for s in ckpt_path.split(".") if s.isdigit()]
    if len(nums) > 0:
        return nums[-1]
    return None


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:
    r"""Return (previous_ckpt_ind + 1)th checkpoint in checkpoint folder
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
    episode_id: Union[int, str],
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


def tensor_to_depth_images(tensor: Union[torch.Tensor, List]) -> np.ndarray:
    r"""Converts tensor (or list) of n image tensors to list of n images.
    Args:
        tensor: tensor containing n image tensors
    Returns:
        list of images
    """
    images = []

    for img_tensor in tensor:
        image = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        images.append(image)

    return images


def tensor_to_bgr_images(
    tensor: Union[torch.Tensor, Iterable[torch.Tensor]]
) -> List[np.ndarray]:
    r"""Converts tensor of n image tensors to list of n BGR images.
    Args:
        tensor: tensor containing n image tensors
    Returns:
        list of images
    """
    images = []

    for img_tensor in tensor:
        img = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    return images


def image_resize_shortest_edge(
    img: Tensor, size: int, channels_last: bool = False
) -> torch.Tensor:
    """Resizes an img so that the shortest side is length of size while
        preserving aspect ratio.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want the shortest edge to be resize to
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    h, w = get_image_height_width(img, channels_last=channels_last)
    if channels_last:
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)

    # Percentage resize
    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


def center_crop(
    img: Tensor, size: Union[int, Tuple[int, int]], channels_last: bool = False
) -> Tensor:
    """Performs a center crop on an image.

    Args:
        img: the array object that needs to be resized (either batched or unbatched)
        size: A sequence (h, w) or a python(int) that you want cropped
        channels_last: If the channels are the last dimension.
    Returns:
        the resized array
    """
    h, w = get_image_height_width(img, channels_last=channels_last)

    if isinstance(size, int):
        size_tuple: Tuple[int, int] = (int(size), int(size))
    else:
        size_tuple = size
    assert len(size_tuple) == 2, "size should be (h,w) you wish to resize to"
    cropy, cropx = size_tuple

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def get_image_height_width(
    img: Union[Box, np.ndarray, torch.Tensor], channels_last: bool = False
) -> Tuple[int, int]:
    if img.shape is None or len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if channels_last:
        # NHWC
        h, w = img.shape[-3:-1]
    else:
        # NCHW
        h, w = img.shape[-2:]
    return h, w


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


def get_scene_episode_dict(episodes: List[Episode]) -> Dict:
    scene_ids = []
    scene_episode_dict = {}

    for episode in episodes:
        if episode.scene_id not in scene_ids:
            scene_ids.append(episode.scene_id)
            scene_episode_dict[episode.scene_id] = [episode]
        else:
            scene_episode_dict[episode.scene_id].append(episode)

    return scene_episode_dict


def base_plus_ext(path: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    """Helper method that splits off all extension.
    Returns base, allext.
    path: path with extensions
    returns: path with all extensions removed
    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample: Optional[Any]) -> bool:
    """Check whether a webdataset sample is valid.
    sample: sample to be checked
    """
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


def img_bytes_2_np_array(
    x: Tuple[int, torch.Tensor, bytes]
) -> Tuple[int, torch.Tensor, bytes, np.ndarray]:
    """Mapper function to convert image bytes in webdataset sample to numpy
    arrays.
    Args:
        x: webdataset sample containing ep_id, question, answer and imgs
    Returns:
        Same sample with bytes turned into np arrays.
    """
    images = []
    img_bytes: bytes
    for img_bytes in x[3:]:
        bytes_obj = BytesIO()
        bytes_obj.write(img_bytes)
        image = np.array(Image.open(bytes_obj))
        img = image.transpose(2, 0, 1)
        img = img / 255.0
        images.append(img)
    return (*x[0:3], np.array(images, dtype=np.float32))


def create_tar_archive(archive_path: str, dataset_path: str) -> None:
    """Creates tar archive of dataset and returns status code.
    Used in VQA trainer's webdataset.
    """
    logger.info("[ Creating tar archive. This will take a few minutes. ]")

    with tarfile.open(archive_path, "w:gz") as tar:
        for file in sorted(os.listdir(dataset_path)):
            tar.add(os.path.join(dataset_path, file))


def delete_folder(path: str) -> None:
    shutil.rmtree(path)
