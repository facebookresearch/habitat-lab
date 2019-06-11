import numpy as np
import pytest

import habitat.utils

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False



def test_mapping_matrix():
    mapping = {0: 1, 1: 2, 2: 3, 3: 4}
    gt_matrix = np.zeros((4, 5), dtype=np.float32)
    gt_matrix[0, 1] = 1
    gt_matrix[1, 2] = 1
    gt_matrix[2, 3] = 1
    gt_matrix[3, 4] = 1

    assert (habitat.utils.build_mapping_matrix(mapping) == gt_matrix).all()


@pytest.mark.parametrize("use_torch", [(True,), (False,)])
def test_mapping_probs(use_torch):
    if use_torch and not has_torch:
        pytest.skip("Test requires torch")

    mapping = {0: 1, 1: 2, 2: 3, 3: 4}
    if use_torch:
        curr_probs = torch.rand(2, 4)
        curr_probs /= curr_probs.sum(-1, keepdim=True)
        expected_probs = torch.zeros(2, 5)

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        curr_probs = curr_probs.to(device)
        expected_probs = expected_probs.to(device)
    else:
        curr_probs = np.random.randn(2, 4)
        curr_probs /= curr_probs.sum(-1)
        expected_probs = np.zeros(2, 5)

    for k, v in mapping.items():
        expected_probs[:, v] = curr_probs[:, k]

    mapped_probs = habitat.utils.map_action_distribution(
        mapping, probs=curr_probs, use_torch=use_torch
    )

    assert (mapped_probs - expected_probs).norm() < 1e-3


@pytest.mark.parametrize("use_torch", [(True,), (False,)])
def test_mapping_logits(use_torch):
    if use_torch and not has_torch:
        pytest.skip("Test requires torch")

    mapping = {0: 1, 1: 2, 2: 3, 3: 4}
    if use_torch:
        curr_logits = torch.rand(2, 4)
        expected_logits = torch.full((2, 5), np.finfo(np.float32).min)

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        curr_logits = curr_logits.to(device)
        expected_logits = expected_logits.to(device)
    else:
        curr_logits = np.random.randn(2, 4)
        expected_logits = np.full((2, 5), np.finfo(np.float32).min)

    for k, v in mapping.items():
        expected_logits[:, v] = curr_logits[:, k]

    mapped_logits = habitat.utils.map_action_distribution(
        mapping, logits=curr_logits, use_torch=use_torch
    )

    assert (mapped_logits - expected_logits).norm() < 1e-3
