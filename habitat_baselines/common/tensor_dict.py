#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Callable, Dict, Optional, Tuple, Union, overload

import numpy as np
import torch

TensorLike = Union[
    torch.Tensor, np.ndarray, int, float, np.float32, np.float64
]
DictTree = Dict[str, Union[TensorLike, "DictTree"]]


def _to_tensor(v: TensorLike) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class TensorDict(dict):
    r"""A dictionary of tensors that can be indexed like a tensor or like a dictionary.  Also
        supports access via dot notation.

    .. code:: py
        t = TensorDict(a=torch.randn(2, 2), b=TensorDict(c=torch.randn(3, 3)))

        print(t)

        print(t[0, 0])

        print(t["a"])
        print(t.b.c)

    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @classmethod
    def from_tree(cls, tree: DictTree) -> "TensorDict":
        res = cls()
        for k, v in tree.items():
            if isinstance(v, dict):
                res[k] = cls.from_tree(v)
            else:
                res[k] = _to_tensor(v)

        return res

    def to_tree(self) -> DictTree:
        res: DictTree = dict()
        for k, v in self.items():
            if isinstance(v, TensorDict):
                res[k] = v.to_tree()
            else:
                res[k] = v

        return res

    @overload
    def __getitem__(self, index: str) -> Union["TensorDict", torch.Tensor]:
        ...

    @overload
    def __getitem__(self, index: Union[int, slice, Tuple]) -> "TensorDict":
        ...

    def __getitem__(
        self, index: Union[str, int, slice, Tuple]
    ) -> Union["TensorDict", torch.Tensor]:
        if isinstance(index, str):
            return super().__getitem__(index)
        else:
            return TensorDict({k: v[index] for k, v in self.items()})

    @overload
    def set(
        self,
        index: str,
        value: Union[TensorLike, "TensorDict", DictTree],
        strict: bool = True,
    ):
        ...

    @overload
    def set(
        self,
        index: Union[int, slice, Tuple],
        value: Union["TensorDict", DictTree],
        strict: bool = True,
    ):
        ...

    def set(
        self,
        index: Union[str, int, slice, Tuple],
        value: Union[TensorLike, "TensorDict"],
        strict: bool = True,
    ):
        if isinstance(index, str):
            super().__setitem__(index, value)
        else:
            if strict and (self.keys() != value.keys()):
                raise RuntimeError(
                    "Keys don't match: Dest={} Source={}".format(
                        self.keys(), value.keys()
                    )
                )

            for k in self.keys():
                if k not in value:
                    if strict:
                        raise RuntimeError(
                            f"Key {k} not in new value dictionary"
                        )
                    else:
                        continue

                v = value[k]

                if isinstance(v, (TensorDict, dict)):
                    self[k].set(index, v, strict=strict)
                else:
                    self[k][index].copy_(_to_tensor(v))

    def __setitem__(
        self,
        index: Union[str, int, slice, Tuple],
        value: Union[torch.Tensor, "TensorDict"],
    ):
        self.set(index, value)

    @classmethod
    def map_func(
        cls,
        func: Callable[[torch.Tensor], torch.Tensor],
        src: "TensorDict",
        dst: Optional["TensorDict"] = None,
    ) -> "TensorDict":
        if dst is None:
            dst = TensorDict()

        for k, v in src.items():
            if torch.is_tensor(v):
                dst[k] = func(v)
            else:
                dst[k] = cls.map_func(func, v, dst.get(k, None))

        return dst

    def map(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "TensorDict":
        return self.map_func(func, self)

    def map_in_place(
        self, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> "TensorDict":
        return self.map_func(func, self, self)

    def __deepcopy__(self, _memo=None) -> "TensorDict":
        return TensorDict.from_tree(copy.deepcopy(self.to_tree()))
