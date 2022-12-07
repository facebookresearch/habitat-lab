#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except ImportError:
    pass


@pytest.mark.skipif(torch is None, reason="Test requires pytorch")
def test_tensor_dict_constructor():
    dict_tree = dict(
        a=torch.randn(2, 2), b=dict(c=dict(d=np.random.randn(3, 3)))
    )
    tensor_dict = TensorDict.from_tree(dict_tree)

    assert torch.is_tensor(tensor_dict["a"])
    assert isinstance(tensor_dict["b"], TensorDict)
    assert isinstance(tensor_dict["b"]["c"], TensorDict)
    assert torch.is_tensor(tensor_dict["b"]["c"]["d"])


@pytest.mark.skipif(torch is None, reason="Test requires pytorch")
def test_tensor_dict_to_tree():
    dict_tree = dict(a=torch.randn(2, 2), b=dict(c=dict(d=torch.randn(3, 3))))

    assert dict_tree == TensorDict.from_tree(dict_tree).to_tree()


@pytest.mark.skipif(torch is None, reason="Test requires pytorch")
def test_tensor_dict_str_index():
    dict_tree = dict(a=torch.randn(2, 2), b=dict(c=dict(d=torch.randn(3, 3))))
    tensor_dict = TensorDict.from_tree(dict_tree)

    x = torch.randn(5, 5)
    tensor_dict["a"] = x
    assert (tensor_dict["a"] == x).all()

    with pytest.raises(KeyError):
        _ = tensor_dict["c"]


@pytest.mark.skipif(torch is None, reason="Test requires pytorch")
def test_tensor_dict_index():
    dict_tree = dict(a=torch.randn(2, 2), b=dict(c=dict(d=torch.randn(3, 3))))
    tensor_dict = TensorDict.from_tree(dict_tree)

    with pytest.raises(KeyError):
        tensor_dict["b"][0] = dict(q=torch.randn(3))

    tmp = dict(c=dict(d=torch.randn(3)))
    tensor_dict["b"][0] = tmp
    assert torch.allclose(tensor_dict["b"]["c"]["d"][0], tmp["c"]["d"])
    assert not torch.allclose(tensor_dict["b"]["c"]["d"][1], tmp["c"]["d"])

    tensor_dict["b"]["c"]["x"] = torch.randn(5, 5)
    with pytest.raises(KeyError):
        tensor_dict["b"][1] = tmp

    tensor_dict["b"].set(1, tmp, strict=False)  # type: ignore
    assert torch.allclose(tensor_dict["b"]["c"]["d"][1], tmp["c"]["d"])

    tmp = dict(c=dict(d=torch.randn(1, 3)))
    del tensor_dict["b"]["c"]["x"]
    tensor_dict["b"][2:3] = tmp
    assert torch.allclose(tensor_dict["b"]["c"]["d"][2:3], tmp["c"]["d"])


@pytest.mark.skipif(torch is None, reason="Test requires pytorch")
def test_tensor_dict_map():
    dict_tree = dict(a=dict(b=[0]))
    tensor_dict = TensorDict.from_tree(dict_tree)

    res = tensor_dict.map(lambda x: x + 1)
    assert (res["a"]["b"] == 1).all()  # type: ignore

    tensor_dict.map_in_place(lambda x: x + 1)

    assert res == tensor_dict
