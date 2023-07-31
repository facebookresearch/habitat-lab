import abc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ColorJitter, RandomApply


class RandomShiftsAug(nn.Module):
    """
    Defines the Random Shifts Augmentation from the paper
    DrQ-v2: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning
    Paper Link: https://arxiv.org/abs/2107.09645
    Code borrowed from here: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
    """

    def __init__(self, pad):
        """
        Initialize the RandomShiftsAug object.

        :param pad: value to be used for padding
        """
        super().__init__()
        self.pad = pad

    def forward(self, x):
        """
        Apply the Random Shifts Augmentation to the input tensor.

        :param x: input tensor to augment
        :return: augmented tensor
        """
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )


class Transform(abc.ABC):
    """
    Abstract class for applying transformation to a tensor.
    Subclasses should implement the apply method
    """

    randomize_environments: bool = False

    @abc.abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to the input tensor.

        :param x: input tensor to transform
        :return: transformed tensor
        """

    def __call__(
        self, x: torch.Tensor, N: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply the transformation to the input tensor, and if N is provided,
        applies the same transformation to all the environments.

        :param x: input tensor to transform
        :param N: number of environments
        :return: transformed tensor
        """
        if not self.randomize_environments or N is None:
            return self.apply(x)

        # shapes
        TN = x.size(0)
        T = TN // N

        # apply the same augmentation when t == 1 for speed
        # typically, t == 1 during policy rollout
        if T == 1:
            return self.apply(x)

        # put environment (n) first
        _, A, B, C = x.shape
        x = torch.einsum("tnabc->ntabc", x.view(T, N, A, B, C))

        # apply the same transform within each environment
        x = torch.cat([self.apply(imgs) for imgs in x])

        # put timestep (t) first
        _, A, B, C = x.shape
        x = torch.einsum("ntabc->tnabc", x.view(N, T, A, B, C)).flatten(0, 1)

        return x


class ShiftAndJitterTransform(Transform):
    """
    Class for applying random color jitter and random shift transform to a tensor.
    """

    def __init__(self, size, jitter_value=0.3, pad_value=4):
        """
        Initialize the ShiftAndJitterTransform object.

        :param size: size of the output image
        :param jitter_value: value to be used in color jitter
        :param pad_value: value to be used in random shift
        """
        self.size = size
        self.jitter_value = jitter_value
        self.pad_value = pad_value

    def apply(self, x) -> torch.Tensor:
        """
        Apply the random color jitter and random shift transform to the input tensor.

        :param x: input tensor to transform
        :return: transformed tensor
        """
        x = RandomApply(
            [
                ColorJitter(
                    self.jitter_value,
                    self.jitter_value,
                    self.jitter_value,
                    self.jitter_value,
                )
            ],
            p=1.0,
        )(x)
        x = RandomShiftsAug(self.pad_value)(x)
        return x
