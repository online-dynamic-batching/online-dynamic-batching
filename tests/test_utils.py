# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for scale_loss and is_valid_batch utilities."""

import torch

from odb.utils import scale_loss, is_valid_batch


class TestScaleLoss:
    def test_equal_batch_sizes(self):
        """When all ranks have same batch size, scaling is identity."""
        loss = torch.tensor(1.0, requires_grad=True)
        # local_bs=4, total_bs=8 (2 ranks × 4), world_size=2
        scaled = scale_loss(loss, batch_or_local_bs=4, total_batch_size=8, world_size=2)
        assert torch.isclose(scaled, loss)

    def test_unequal_batch_sizes(self):
        """When rank has smaller batch, its loss should be scaled down."""
        loss = torch.tensor(1.0, requires_grad=True)
        # rank 0: local_bs=2, rank 1: local_bs=6, total=8, world=2
        scaled = scale_loss(loss, batch_or_local_bs=2, total_batch_size=8, world_size=2)
        expected = 1.0 * (2 / 8) * 2  # = 0.5
        assert torch.isclose(scaled, torch.tensor(expected))

    def test_larger_batch_rank(self):
        """When rank has larger batch, its loss should be scaled up."""
        loss = torch.tensor(1.0, requires_grad=True)
        # rank with local_bs=6 out of total_bs=8, world=2
        scaled = scale_loss(loss, batch_or_local_bs=6, total_batch_size=8, world_size=2)
        expected = 1.0 * (6 / 8) * 2  # = 1.5
        assert torch.isclose(scaled, torch.tensor(expected))

    def test_gradient_flows(self):
        """Scaled loss should be differentiable."""
        loss = torch.tensor(2.0, requires_grad=True)
        scaled = scale_loss(loss, batch_or_local_bs=3, total_batch_size=9, world_size=3)
        scaled.backward()
        assert loss.grad is not None

    def test_single_gpu(self):
        """Single GPU: local_bs == total_bs, world=1 → identity."""
        loss = torch.tensor(1.5, requires_grad=True)
        scaled = scale_loss(loss, batch_or_local_bs=4, total_batch_size=4, world_size=1)
        assert torch.isclose(scaled, loss)

    def test_zero_total_bs(self):
        """Edge case: zero total_batch_size should return loss unchanged."""
        loss = torch.tensor(1.0)
        scaled = scale_loss(loss, batch_or_local_bs=0, total_batch_size=0, world_size=2)
        assert torch.isclose(scaled, loss)

    def test_batch_dict_input(self):
        """scale_loss should accept a batch dict and extract bs automatically."""
        loss = torch.tensor(1.0, requires_grad=True)
        batch = {"local_batch_size": 2, "total_batch_size": 8}
        # Without DDP, world_size=1, so scaling = (2/8)*1 = 0.25
        scaled = scale_loss(loss, batch)
        expected = 1.0 * (2 / 8) * 1  # world_size=1 (no DDP in test)
        assert torch.isclose(scaled, torch.tensor(expected))


class TestIsValidBatch:
    def test_valid_batch(self):
        batch = [
            {"input_ids": torch.zeros(1, 100)},
            {"input_ids": torch.zeros(1, 200)},
        ]
        assert is_valid_batch(batch) is True

    def test_empty_batch(self):
        assert is_valid_batch([]) is False

    def test_missing_input_ids(self):
        assert is_valid_batch([{}]) is False

    def test_empty_tensor(self):
        assert is_valid_batch([{"input_ids": torch.zeros(1, 0)}]) is False

    def test_list_input_ids(self):
        assert is_valid_batch([{"input_ids": [1, 2, 3]}]) is True
        assert is_valid_batch([{"input_ids": []}]) is False
