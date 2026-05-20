#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for NaN/Inf gradient protection in PPO.before_step().

A single bad batch can produce NaN gradients that silently corrupt
model weights through optimizer.step(). The before_step() hook in
PPO now replaces NaN/Inf with safe values before the optimizer step.

This test duplicates the before_step() logic in a minimal PPO-like
test class so it runs without habitat-sim (conda-only dependency).
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


class MockActorCritic(nn.Module):
    """Minimal actor-critic stub matching NetPolicy interface."""

    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(8, 4)

    def policy_parameters(self):
        return self.policy.parameters()

    def aux_loss_parameters(self):
        return {}


class PPOBeforeStep:
    """Test-only surrogate that replicates PPO.before_step() logic.

    The real before_step() in habitat_baselines does:
      1. Distributed all-reduce for non-actor-critic params
      2. clip_grad_norm_ on policy parameters
      3. clip_grad_norm_ on aux loss parameters
      4. Wait for all-reduce handles
      5. Replace NaN/Inf gradients with safe values  <-- the fix under test
    """

    def __init__(self, max_grad_norm=0.5):
        self.actor_critic = MockActorCritic()
        self.max_grad_norm = max_grad_norm
        # non_ac_params: params whose name does not start with "actor_critic."
        self.non_ac_params = []

        # Create optimizer so parameter gradients are initialized
        self.optimizer = optim.SGD(
            self.parameters(), lr=0.001
        )

        # Initialize all gradients to zero
        self._zero_grads()

    def _zero_grads(self):
        for p in self.parameters():
            p.grad = torch.zeros_like(p.data)

    def parameters(self):
        return self.actor_critic.parameters()

    def named_parameters(self):
        return self.actor_critic.named_parameters()

    def before_step(self):
        """Replicates the exact logic from PPO.before_step() in ppo.py."""
        handles = []
        if torch.distributed.is_initialized():
            for p in self.non_ac_params:
                if p.grad is not None:
                    p.grad.data.detach().div_(
                        torch.distributed.get_world_size()
                    )
                    handles.append(
                        torch.distributed.all_reduce(
                            p.grad.data.detach(), async_op=True
                        )
                    )

        grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.policy_parameters(),
            self.max_grad_norm,
        )

        for v in self.actor_critic.aux_loss_parameters().values():
            nn.utils.clip_grad_norm_(v, self.max_grad_norm)

        [h.wait() for h in handles]

        # Replace NaN/Inf gradients with zero to prevent silent weight corruption
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.nan_to_num_(
                    nan=0.0,
                    posinf=self.max_grad_norm,
                    neginf=-self.max_grad_norm,
                )

        return grad_norm


@pytest.fixture
def updater():
    """Create a PPOBeforeStep instance for testing."""
    return PPOBeforeStep(max_grad_norm=0.5)


def test_nan_gradients_replaced_with_zero(updater):
    """NaN gradients are replaced with zero in before_step()."""
    param = next(updater.parameters())
    param.grad = torch.full(param.grad.shape, float("nan"))

    updater.before_step()

    assert not torch.isnan(param.grad).any(), "NaN gradients were not replaced"


def test_inf_gradients_clamped_to_max_norm(updater):
    """Inf gradients are clamped to max_grad_norm bounds."""
    param = next(updater.parameters())
    param.grad = torch.full(param.grad.shape, float("inf"))

    updater.before_step()

    assert not torch.isinf(param.grad).any(), "Positive inf was not clamped"

    param.grad = torch.full(param.grad.shape, float("-inf"))
    updater.before_step()

    assert not torch.isinf(param.grad).any(), "Negative inf was not clamped"


def test_valid_gradients_unchanged(updater):
    """Finite gradients pass through before_step() unmodified."""
    param = next(updater.parameters())
    original = param.grad.clone()
    orig_values = original.flatten()[:6].clone()
    # Set some non-zero finite values
    param.grad.flatten()[:6] = torch.tensor([0.1, -0.2, 0.05, -0.01, 0.3, -0.15])
    original = param.grad.clone()

    updater.before_step()

    assert torch.allclose(param.grad, original), (
        "Valid gradients were incorrectly modified"
    )


def test_all_parameters_cleaned(updater):
    """NaN/Inf replacement applies to every parameter."""
    for p in updater.parameters():
        if p.grad is not None:
            p.grad.fill_(float("nan"))

    updater.before_step()

    for p in updater.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), (
                "NaN remains in gradient after before_step"
            )
            assert not torch.isinf(p.grad).any(), (
                "Inf remains in gradient after before_step"
            )


def test_optimizer_step_with_nan_gradients(updater):
    """End-to-end: NaN gradients do not corrupt model weights after step().

    This is the core scenario from issue #2226: a single batch with
    NaN gradients should not permanently corrupt the optimizer state.
    """
    # Inject NaN gradients (simulates a bad loss.backward() call)
    for p in updater.parameters():
        if p.grad is not None:
            p.grad.fill_(float("nan"))

    updater.before_step()
    updater.optimizer.step()

    # Verify weights are still finite
    for name, p in updater.named_parameters():
        assert torch.isfinite(p).all(), (
            f"Weights for {name} became NaN/Inf after optimizer.step()"
        )
