#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import numbers
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray, numbers.Real]
DictTree = Dict[str, Union[TensorLike, "DictTree"]]  # type: ignore
TensorIndexType = Union[
    int,
    slice,
    torch.Tensor,
    np.ndarray,
    Tuple[Union[int, slice, torch.Tensor, np.ndarray], ...],
]


class SupportsIndexing(Protocol):
    def __getitem__(self, key: Any) -> Any:
        pass

    def __setitem__(self, key: Any, value: Any) -> None:
        pass


T = TypeVar("T", bound=SupportsIndexing)
_DictTreeInst = TypeVar("_DictTreeInst", bound="_DictTreeBase")

_MapFuncType = Callable[[T], T]

_ApplyFuncType = Callable[[T], None]


class _DictTreeBase(Dict[str, Union["_DictTreeBase[T]", T]]):
    r"""Base class that represents a dictionary tree (DictTree).

    In a DictTree, all elements of the dict are either a leaf (of type T) or
    a subtree. This is setup for T's that are indexable, like a torch.Tensor
    or np.ndarray
    """

    @classmethod
    def _to_instance(cls, v: Any) -> T:
        raise NotImplementedError()

    @classmethod
    def from_tree(
        cls: Type[_DictTreeInst], tree: Dict[str, Any]
    ) -> _DictTreeInst:
        res = cls()
        for k, v in tree.items():
            if isinstance(v, dict):
                res[k] = cls.from_tree(v)
            else:
                res[k] = cls._to_instance(v)

        return res

    def to_tree(self) -> DictTree:
        res: DictTree = dict()
        for k, v in self.items():
            if isinstance(v, _DictTreeBase):
                res[k] = v.to_tree()
            else:
                res[k] = v

        return res

    @classmethod
    def _from_flattened_helper(
        cls: Type[_DictTreeInst],
        spec: List[Tuple[str, ...]],
        leaves: List[TensorLike],
    ) -> _DictTreeInst:
        res = cls()
        remaining: Tuple[List[Tuple[str, ...]], List[TensorLike]] = ([], [])
        for i, (first_key, *other_keys_l), v in zip(
            range(len(spec)), spec, leaves
        ):
            other_keys = tuple(other_keys_l)
            if len(other_keys) == 0:
                v = cls._to_instance(v)

                if first_key in res:
                    raise RuntimeError(
                        f"Key '{first_key}' already in the tree. Invalid spec."
                    )

                res[first_key] = v
            else:
                remaining[0].append(other_keys)
                remaining[1].append(v)

            if ((i == len(spec) - 1) or first_key != spec[i + 1][0]) and len(
                remaining[0]
            ) > 0:
                res[first_key] = cls._from_flattened_helper(*remaining)
                remaining = ([], [])

        return res

    @classmethod
    def from_flattened(
        cls: Type[_DictTreeInst],
        spec: List[Tuple[str, ...]],
        leaves: List[TensorLike],
    ) -> _DictTreeInst:
        r"""Construct a DictTree from the flattened representation, i.e. from :py:ref:`flatten`.

        :param spec: The key for each leaf.
        :param leaves: The leaves.
        """
        assert len(spec) == len(leaves)
        sort_ordering = sorted(range(len(spec)), key=lambda i: spec[i])
        spec = [spec[i] for i in sort_ordering]
        leaves = [leaves[i] for i in sort_ordering]

        return cls._from_flattened_helper(spec, leaves)

    def flatten(self) -> Tuple[List[Tuple[str, ...]], List[T]]:
        r"""Returns a flattened representation of the tree.

        This is useful for
        operating on the leaves and then making a new tree with the same keys
        when operation on the leaves can't be easily done with :py:ref:`map`

        A new instance can be created via :py:ref:`from_flattened`.

        :return: A tuple of lists where the first list is the key for each leaf and the second
        list is all the leaves.
        """
        spec = []
        tensors = []
        for k, v in self.items():
            if isinstance(v, _DictTreeBase):
                for subk, subv in zip(*v.flatten()):
                    spec.append((k, *subk))
                    tensors.append(subv)
            else:
                spec.append((k,))
                tensors.append(v)

        return spec, tensors

    @overload
    def __getitem__(
        self: _DictTreeInst, index: str
    ) -> Union[_DictTreeInst, T]:
        ...

    @overload
    def __getitem__(
        self: _DictTreeInst, index: TensorIndexType
    ) -> _DictTreeInst:
        ...

    def __getitem__(
        self: _DictTreeInst, index: Union[str, TensorIndexType]
    ) -> Union[_DictTreeInst, T]:
        if isinstance(index, str):
            return cast(Union[_DictTreeInst, T], super().__getitem__(index))
        else:
            return type(self)((k, v[index]) for k, v in self.items())

    @overload
    def set(
        self,
        index: str,
        value: Union[TensorLike, _DictTreeBase[T], DictTree],
        strict: bool = True,
    ) -> None:
        ...

    @overload
    def set(
        self,
        index: TensorIndexType,
        value: Union[_DictTreeBase[T], DictTree],
        strict: bool = True,
    ) -> None:
        ...

    def set(
        self,
        index: Union[str, TensorIndexType],
        value: Union[TensorLike, _DictTreeBase[T]],
        strict: bool = True,
    ) -> None:
        if isinstance(index, str):
            if not isinstance(value, _DictTreeBase):
                value = self._to_instance(value)

            super().__setitem__(index, value)
        else:
            assert isinstance(value, dict)
            if strict and (self.keys() != value.keys()):
                raise KeyError(
                    "Keys don't match: Dest={} Source={}".format(
                        self.keys(), value.keys()
                    )
                )

            for k in self.keys():
                if k not in value:
                    if strict:
                        raise KeyError(f"Key {k} not in new value dictionary")
                    else:
                        continue

                v = value[k]
                dst = self[k]

                if isinstance(v, dict):
                    assert isinstance(dst, _DictTreeBase)
                    dst.set(index, v, strict=strict)
                else:
                    assert not isinstance(dst, _DictTreeBase)
                    dst[index] = self._to_instance(v)

    def __setitem__(
        self,
        index: Union[str, TensorIndexType],
        value: Union[torch.Tensor, _DictTreeBase[T]],
    ):
        self.set(index, value)

    @classmethod
    def _map_apply_func(
        cls: Type[_DictTreeInst],
        func: Union[_MapFuncType, _ApplyFuncType],
        src: Union[_DictTreeBase[T], DictTree],
        dst_in: Optional[_DictTreeInst] = None,
        needs_return: bool = True,
        prefix: str = "",
    ) -> Union[_DictTreeInst, None]:
        if dst_in is None and needs_return:
            dst = cls()
        else:
            dst = dst_in

        for k, v in src.items():
            if isinstance(v, (cls, dict)):
                dst_k = dst.get(k, None) if needs_return else None
                if dst_k is not None:
                    assert isinstance(dst_k, _DictTreeBase)
                res = cls._map_apply_func(
                    func,
                    v,  # type: ignore
                    dst_k,  # type: ignore
                    needs_return,
                    prefix=f"{prefix}{k}.",
                )
            else:
                res = func(v)

            if needs_return:
                dst[k] = res

        return dst

    @classmethod
    def map_func(
        cls: Type[_DictTreeInst],
        func: _MapFuncType,
        src: Union[_DictTreeBase[T], DictTree],
        dst: Optional[_DictTreeInst] = None,
    ) -> _DictTreeInst:
        return cls._map_apply_func(func, src, dst, needs_return=True)

    def map(self: _DictTreeInst, func: _MapFuncType) -> _DictTreeInst:
        r"""Apply a function to all leaves that transforms that leaf into a new value.
        Returns a new instances with the transformed leafs."""
        return self.map_func(func, self)

    def map_in_place(self: _DictTreeInst, func: _MapFuncType) -> _DictTreeInst:
        r"""Same as :py:ref:`map` but modifies the current tree. The current tree is
        returned for chaining.
        """
        return self.map_func(func, self, self)

    def apply(self, func: _ApplyFuncType) -> None:
        r"""Applies a function to all leaves where the function doesn't
        return a new value
        """
        self._map_apply_func(func, self, dst_in=None, needs_return=False)

    def slice_keys(
        self: _DictTreeInst, *keys: Union[str, Iterable[str]]
    ) -> _DictTreeInst:
        r"""Returns a new instance that only has the specified keys.

        The new instance is a shallow copy and references the same underlying data.
        """

        res = type(self)()
        for _k in keys:
            for k in (_k,) if isinstance(_k, str) else _k:
                assert k in self, f"Key {k} not in self"
                res[k] = self[k]

        return res

    def __deepcopy__(self: _DictTreeInst, _memo=None) -> _DictTreeInst:
        return self.from_tree(copy.deepcopy(self.to_tree(), memo=_memo))


class TensorDict(_DictTreeBase[torch.Tensor]):
    r"""A dictionary of tensors that can be indexed like a tensor or like a dictionary.

    .. code:: py
        t = TensorDict(a=torch.randn(2, 2), b=TensorDict(c=torch.randn(3, 3)))

        print(t)

        print(t[0, 0])

        print(t["a"])

    """

    @classmethod
    def _to_instance(cls, v: Any) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v
        elif isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        else:
            return torch.as_tensor(v)

    def numpy(self) -> NDArrayDict:
        return NDArrayDict.from_tree(self.to_tree())


class NDArrayDict(_DictTreeBase[np.ndarray]):
    @classmethod
    def _to_instance(cls, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            return v
        elif isinstance(v, torch.Tensor):
            return v.numpy()
        else:
            return np.asarray(v)

    def as_tensor(self) -> TensorDict:
        return TensorDict.from_tree(self.to_tree())


class TensorOrNDArrayDict(_DictTreeBase[Union[torch.Tensor, np.ndarray]]):
    @classmethod
    def _to_instance(cls, v: Any) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(v, (np.ndarray, torch.Tensor)):
            return v
        else:
            return np.asarray(v)  # type: ignore


def iterate_dicts_recursively(
    *dicts_i: _DictTreeBase[T],
) -> Iterable[Tuple[T, ...]]:
    r"""Iterate a list of DictTrees recursively and yield a tuple of each tree's leaves."""
    dicts = tuple(dicts_i)
    for k in dicts[0].keys():
        assert all(k in d for d in dicts)

        if isinstance(dicts[0][k], _DictTreeBase):
            yield from iterate_dicts_recursively(*tuple(d[k] for d in dicts))  # type: ignore
        else:
            yield tuple(cast(T, d[k]) for d in dicts)


def transpose_list_of_dicts(*dicts_i: Dict[Any, Any]) -> Dict[Any, List[Any]]:
    r"""Transposes a list of dicts into a dict of lists."""
    res: Dict[Any, List[Any]] = {}
    dicts = tuple(dicts_i)
    for k in dicts[0].keys():
        assert all(k in d for d in dicts)
        if isinstance(dicts[0][k], dict):
            res[k] = transpose_list_of_dicts(*tuple(d[k] for d in dicts))  # type: ignore
        else:
            res[k] = [d[k] for d in dicts]

    return res
