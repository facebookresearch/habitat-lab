#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import math
import numbers
import os
import re
import shutil
import tarfile
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
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
from gym import spaces
from PIL import Image
from torch import Size, Tensor
from torch import nn as nn

from habitat import logger
from habitat.core.dataset import Episode
from habitat.core.spaces import EmptySpace
from habitat.core.utils import Singleton
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensor_dict import (
    DictTree,
    TensorDict,
    TensorOrNDArrayDict,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

if TYPE_CHECKING:
    from omegaconf import DictConfig

import cv2

if hasattr(torch, "inference_mode"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad


def cosine_decay(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)

    return (1.0 + math.cos(progress * math.pi)) / 2.0


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
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x.float(), validate_args=False)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return self.rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        return super().entropy().sum(-1, keepdim=True)


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config: "DictConfig",
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_softplus = config.use_softplus
        self.use_log_std = config.use_log_std
        use_std_param = config.use_std_param
        self.clamp_std = config.clamp_std

        if self.use_log_std:
            self.min_std = config.min_log_std
            self.max_std = config.max_log_std
            std_init = config.log_std_init
        elif self.use_softplus:
            inv_softplus = lambda x: math.log(math.exp(x) - 1)
            self.min_std = inv_softplus(config.min_std)
            self.max_std = inv_softplus(config.max_std)
            std_init = inv_softplus(1.0)
        else:
            self.min_std = config.min_std
            self.max_std = config.max_std
            std_init = 1.0  # initialize std value so that std ~ 1

        if use_std_param:
            self.std = torch.nn.parameter.Parameter(
                torch.randn(num_outputs) * 0.01 + std_init
            )
            num_linear_outputs = num_outputs
        else:
            self.std = None
            num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)

        if not use_std_param:
            nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], std_init)

    def forward(self, x: Tensor) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        if self.std is not None:
            mu = mu_maybe_std
            std = self.std
        else:
            mu, std = torch.chunk(mu_maybe_std, 2, -1)

        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        if self.clamp_std:
            std = torch.clamp(std, self.min_std, self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std, validate_args=False)


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
class _ObservationBatchingCache(metaclass=Singleton):
    r"""Helper for batching observations that maintains a cpu-side tensor
    that is the right size and is pinned to cuda memory
    """
    _pool: Dict[Any, Union[torch.Tensor, np.ndarray]] = {}

    def get(
        self,
        num_obs: int,
        sensor_name: Any,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """
        key = (
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            cache = self._pool[key]
            if cache.shape[0] >= num_obs:
                return cache[0:num_obs]
            else:
                cache = None
                del self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            cache = cache.pin_memory()

        if cache.device.type == "cpu":
            # Pytorch indexing is slow,
            # so convert to numpy
            cache = cache.numpy()

        self._pool[key] = cache
        return cache

    def batch_obs(
        self,
        observations: List[DictTree],
        device: Optional[torch.device] = None,
    ) -> TensorDict:
        observations = [
            TensorOrNDArrayDict.from_tree(o).map(
                lambda t: t.numpy()
                if isinstance(t, torch.Tensor) and t.device.type == "cpu"
                else t
            )
            for o in observations
        ]
        observation_keys, _ = observations[0].flatten()
        observation_tensors = [o.flatten()[1] for o in observations]

        # Order sensors by size, stack and move the largest first
        upload_ordering = sorted(
            range(len(observation_keys)),
            key=lambda idx: 1
            if isinstance(observation_tensors[0][idx], numbers.Number)
            else int(np.prod(observation_tensors[0][idx].shape)),  # type: ignore
            reverse=True,
        )

        batched_tensors = []
        for sensor_name, obs in zip(observation_keys, observation_tensors[0]):
            batched_tensors.append(
                self.get(
                    len(observations),
                    sensor_name,
                    torch.as_tensor(obs),
                    device,
                )
            )

        for idx in upload_ordering:
            for i, all_obs in enumerate(observation_tensors):
                obs = all_obs[idx]
                # Use isinstance(sensor, np.ndarray) here instead of
                # np.asarray as this is quickier for the more common
                # path of sensor being an np.ndarray
                # np.asarray is ~3x slower than checking
                if isinstance(obs, np.ndarray):
                    batched_tensors[idx][i] = obs  # type: ignore
                elif isinstance(obs, torch.Tensor):
                    batched_tensors[idx][i].copy_(obs, non_blocking=True)  # type: ignore
                # If the sensor wasn't a tensor, then it's some CPU side data
                # so use a numpy array
                else:
                    batched_tensors[idx][i] = np.asarray(obs)  # type: ignore

            # With the batching cache, we use pinned mem
            # so we can start the move to the GPU async
            # and continue stacking other things with it
            # If we were using a numpy array to do indexing and copying,
            # convert back to torch tensor
            # We know that batch_t[sensor_name] is either an np.ndarray
            # or a torch.Tensor, so this is faster than torch.as_tensor
            if isinstance(batched_tensors[idx], np.ndarray):
                batched_tensors[idx] = torch.from_numpy(batched_tensors[idx])

            batched_tensors[idx] = batched_tensors[idx].to(  # type: ignore
                device, non_blocking=True
            )

        return TensorDict.from_flattened(observation_keys, batched_tensors)


@inference_mode()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
    Returns:
        transposed dict of torch.Tensor of observations.
    """

    return _ObservationBatchingCache().batch_obs(observations, device)


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
        filter(
            lambda name: "latest" not in name,
            filter(os.path.isfile, glob.glob(checkpoint_folder + "/*")),
        )
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
    verbose: bool = True,
    keys_to_include_in_name: Optional[List[str]] = None,
) -> str:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "SPL".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        The saved video name.
    """
    if len(images) < 1:
        return ""

    metric_strs = []
    if (
        keys_to_include_in_name is not None
        and len(keys_to_include_in_name) > 0
    ):
        use_metrics_k = [
            k
            for k in metrics
            if any(
                to_include_k in k for to_include_k in keys_to_include_in_name
            )
        ]
    else:
        use_metrics_k = list(metrics.keys())

    for k in use_metrics_k:
        metric_strs.append(f"{k}={metrics[k]:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(
            images, video_dir, video_name, fps=fps, verbose=verbose
        )
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )
    return video_name


def tensor_to_depth_images(
    tensor: Union[torch.Tensor, List]
) -> List[np.ndarray]:
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
    img: Tensor,
    size: int,
    channels_last: bool = False,
    interpolation_mode="area",
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
        img.float(), size=(h, w), mode=interpolation_mode
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
    img: Union[spaces.Box, np.ndarray, torch.Tensor],
    channels_last: bool = False,
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


def overwrite_gym_box_shape(box: spaces.Box, shape) -> spaces.Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return spaces.Box(low=low, high=high, shape=shape, dtype=box.dtype)


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


def action_to_velocity_control(
    action: torch.Tensor,
    allow_sliding: bool = None,
) -> Dict[str, Any]:
    lin_vel, ang_vel = torch.clip(action, min=-1, max=1)
    step_action = {
        "action": {
            "action": "velocity_control",
            "action_args": {
                "linear_velocity": lin_vel.item(),
                "angular_velocity": ang_vel.item(),
                "allow_sliding": allow_sliding,
            },
        }
    }
    return step_action


def iterate_action_space_recursively(action_space):
    if isinstance(action_space, spaces.Dict):
        for v in action_space.values():
            yield from iterate_action_space_recursively(v)
    else:
        yield action_space


def is_continuous_action_space(action_space) -> bool:
    possible_discrete_spaces = (
        spaces.Discrete,
        spaces.MultiDiscrete,
        spaces.Dict,
    )
    if isinstance(action_space, spaces.Box):
        return True
    elif isinstance(action_space, possible_discrete_spaces):
        return False
    else:
        raise NotImplementedError(
            f"Unknown action space {action_space}. Is neither continuous nor discrete"
        )


def get_action_space_info(ac_space: spaces.Space) -> Tuple[Tuple[int], bool]:
    """
    :returns: The shape of the action space and if the action space is discrete. If the action space is discrete, the shape will be `(1,)`.
    """
    if is_continuous_action_space(ac_space):
        # Assume NONE of the actions are discrete
        return (
            (
                get_num_actions(
                    ac_space,
                ),
            ),
            False,
        )

    elif isinstance(ac_space, spaces.MultiDiscrete):
        return ac_space.shape, True
    elif isinstance(ac_space, spaces.Dict):
        num_actions = 0
        for _, ac_sub_space in ac_space.items():
            num_actions += get_action_space_info(ac_sub_space)[0][0]
        return (num_actions,), True

    else:
        # For discrete pointnav
        return (1,), True


def get_num_actions(action_space) -> int:
    num_actions = 0
    for v in iterate_action_space_recursively(action_space):
        if isinstance(v, spaces.Box):
            assert (
                len(v.shape) == 1
            ), f"shape was {v.shape} but was expecting a 1D action"
            num_actions += v.shape[0]
        elif isinstance(v, EmptySpace):
            num_actions += 1
        elif isinstance(v, spaces.Discrete):
            num_actions += v.n
        else:
            raise NotImplementedError(
                f"Trying to count the number of actions with an unknown action space {v}"
            )

    return num_actions


class LagrangeInequalityCoefficient(nn.Module):
    r"""Implements a learnable lagrange coefficient for a constrained
    optimization problem.


    Given the constrained optimization problem
        min f(x)
            st. x < threshold

    The lagrangian relaxation is then the dual problem
        argmax_alpha argmin_x f(x) + alpha * (x - threshold)
            st. alpha > 0

    We can optimize the dual problem via coordinate descent as
        f(x) + [[alpha]]_sg * x - alpha * ([[x]]_sg - threshold)
    To satisfy the constraint on alpha, we use projected gradient
    descent and project alpha to be > 0 after every step.

    To enforce x > threshold, we negate x and the threshold.
    This yields the coordinate descent objective
       alpha * (threshold - [[x]]_sg) - [[alpha]]_sg * x
    """

    def __init__(
        self,
        threshold: float,
        init_alpha: float = 1.0,
        alpha_min: float = 1e-4,
        alpha_max: float = 1.0,
        greater_than: bool = False,
    ):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((), math.log(init_alpha)))
        self.threshold = float(threshold)
        self.log_alpha_min = math.log(alpha_min)
        self.log_alpha_max = math.log(alpha_max)
        self._greater_than = greater_than

    def project_into_bounds(self):
        r"""Projects alpha back into bounds. To be called after each optim step"""
        with torch.no_grad():
            self.log_alpha.data.clamp_(self.log_alpha_min, self.log_alpha_max)

    def forward(self):
        r"""Compute alpha. This is done to allow forward hooks to work,
        the expected entry point is ref:`lagrangian_loss`"""
        return torch.exp(self.log_alpha)

    def lagrangian_loss(self, x):
        r"""Return the coordinate ascent lagrangian loss that keeps x
        less than or greater than the threshold.
        """
        alpha = self()

        if not self._greater_than:
            return alpha.detach() * x - alpha * (x.detach() - self.threshold)
        else:
            return alpha * (self.threshold - x.detach()) - alpha.detach() * x
