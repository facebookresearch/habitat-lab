from typing import Dict

import numpy as np


def build_mapping_matrix(mapping: Dict[int, int]) -> np.ndarray:
    r"""Creates a matrix to map probabilities from one action space to another

    Params:
        mapping (Dict[int, int]):  The mapping between action spaces.  Keys in the dictionary
            are the index in the current action space and the corresponding value is the action
            index in the new action space

    Returns:
        (np.ndarray): Matrix to map between action spaces


    Usage::
        
        # Mapping between stop on 3 and stop on 0
        mapping = {0: 1, 1: 2, 2: 3, 3: 0}
        probs = np.random.randn(8, 4)
        probs /= probs.sum(-1)

        new_probs = probs @ build_mapping_matrix(mapping)
    """

    d1 = max(list(mapping.values())) + 1
    d0 = max(list(mapping.keys())) + 1
    assert d0 == len(mapping), "All indices in the source must be mapped"
    assert (
        d1 >= d0
    ), "The destination action space must be at least the same size as the source"

    mapping_matrix = np.zeros((d0, d1), dtype=np.float32)
    for k, v in mapping.items():
        mapping_matrix[k, v] = 1.0

    return mapping_matrix


def _map_probs(mapping, probs, use_torch):
    mapping_matrix = build_mapping_matrix(mapping)
    if use_torch:
        import torch

        assert torch.is_tensor(probs)
        mapping_matrix = torch.from_numpy(mapping_matrix).to(
            device=probs.device, dtype=probs.dtype
        )

    return probs @ mapping_matrix


def _map_logits(mapping, logits, use_torch):
    mapping_matrix = build_mapping_matrix(mapping)

    if use_torch:
        import torch

        assert torch.is_tensor(logits)
        assert logits.dtype == torch.float32
        mapping_matrix = torch.from_numpy(mapping_matrix).to(
            device=logits.device, dtype=logits.dtype
        )

        mask = torch.ones_like(logits) @ mapping_matrix
        flmin = torch.full(
            (logits.size(0), mask.size(1)),
            np.finfo(np.float32).min,
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        mask = np.ones_like(logits) @ mapping_matrix
        flmin = np.full(
            (logits.shape[0], mask.shape[1]), np.finfo(np.float32).min
        )

    return logits @ mapping_matrix + (1.0 - mask) * flmin


def map_action_distribution(
    mapping: Dict[int, int], *, probs=None, logits=None, use_torch=False
):
    r"""Maps probabilities or logits from one action space to another.

    Params:
        mapping (Dict[int, int]):  The mapping between action spaces.  Keys in the dictionary
            are the index in the current action space and the corresponding value is the action
            index in the new action space
        logits: The logits to map, mutually exclusive with probs.  
            If the destinate action space is larger than the source action space, the unspecified logits
            are set to the minimal floating point value.
        probs: The probs to map, mutually exclusive with logits
        use_torch (bool): Whether or not to use PyTorch.  If true, this operation is differentiable!

    Returns:
        The mapped probs or logits


    Usage::
        
        # Mapping between stop on 3 and stop on 0
        mapping = {0: 1, 1: 2, 2: 3, 3: 0}
        probs = np.random.randn(8, 4)
        probs /= probs.sum(-1)

        new_probs = map_action_distribution(mapping, probs=probs)

        logits = np.random.randn(8, 4)
        new_logits = map_action_distribution(mapping, logits=logits)

    """
    assert probs is not None or logits is not None
    assert probs is None or logits is None

    if probs is not None:
        return _map_probs(mapping, probs, use_torch)
    else:
        return _map_logits(mapping, logits, use_torch)
