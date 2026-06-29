# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the ODB grouping algorithm."""

import torch

from odb.grouping import (
    _apply_group_order,
    _rank_group_order_indices,
    grouping_data,
    normalize_group_order_flip,
)
from odb.constants import TOTAL_BATCH_SIZE_KEY
from odb.utils import compute_batch_size, get_input_length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(length: int) -> dict:
    """Create a minimal sample dict with input_ids of the given length."""
    return {"input_ids": torch.zeros(1, length, dtype=torch.long)}


def _make_samples(lengths: list[int]) -> list[list[dict]]:
    """Wrap samples into the data_buffer format (list of single-sample lists)."""
    return [[_make_sample(length)] for length in lengths]




# ---------------------------------------------------------------------------
# group_order_flip helpers
# ---------------------------------------------------------------------------

class TestGroupOrderFlip:
    def test_mode_aliases(self):
        assert normalize_group_order_flip("A") == "none"
        assert normalize_group_order_flip("B") == "rank_epoch_random"
        assert normalize_group_order_flip("C") == "rank_window_random"
        assert normalize_group_order_flip("D") == "rank_window_balanced"
        assert normalize_group_order_flip("rank-window-balanced") == "rank_window_balanced"

    def test_none_preserves_order(self):
        assert _rank_group_order_indices(4, "none", 123, 0, 1, 4) == [0, 1, 2, 3]

    def test_balanced_flips_half_the_ranks(self):
        orders = [
            _rank_group_order_indices(4, "rank_window_balanced", 123, 7, rank, 4)
            for rank in range(4)
        ]
        flipped = [order == [3, 2, 1, 0] for order in orders]
        assert sum(flipped) == 2

    def test_apply_group_order(self):
        assert _apply_group_order(["g0", "g1", "g2"], [2, 0, 1]) == ["g2", "g0", "g1"]

# ---------------------------------------------------------------------------
# compute_batch_size
# ---------------------------------------------------------------------------

class TestComputeBatchSize:
    def test_short_sequence(self):
        assert compute_batch_size(1024, 8192) == 8

    def test_max_length(self):
        assert compute_batch_size(8192, 8192) == 1

    def test_longer_than_max(self):
        assert compute_batch_size(16384, 8192) == 1

    def test_zero_length(self):
        assert compute_batch_size(0, 8192) == 1

    def test_various_lengths(self):
        assert compute_batch_size(2048, 8192) == 4
        assert compute_batch_size(4096, 8192) == 2
        assert compute_batch_size(100, 8192) == 81


# ---------------------------------------------------------------------------
# get_input_length
# ---------------------------------------------------------------------------

class TestGetInputLength:
    def test_2d_tensor(self):
        assert get_input_length({"input_ids": torch.zeros(1, 512)}) == 512

    def test_1d_tensor(self):
        assert get_input_length({"input_ids": torch.zeros(256)}) == 256

    def test_list(self):
        assert get_input_length({"input_ids": [0] * 100}) == 100

    def test_none(self):
        assert get_input_length({}) == 0
        assert get_input_length({"input_ids": None}) == 0

    def test_empty_tensor(self):
        assert get_input_length({"input_ids": torch.zeros(0)}) == 0
        assert get_input_length({"input_ids": torch.zeros(1, 0)}) == 0


# ---------------------------------------------------------------------------
# grouping_data (non-DDP)
# ---------------------------------------------------------------------------

class TestGroupingNonDDP:
    """Test grouping without DDP (ddp_group=None)."""

    def test_uniform_short_sequences(self):
        """All short sequences should be grouped into one large batch."""
        lengths = [100] * 20
        data_buffer = _make_samples(lengths)
        groups, overflow, is_done, skip = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=10,
        )
        assert not skip
        assert len(groups) >= 1
        total_samples = sum(len(g) for g in groups) + len(overflow)
        assert total_samples == 20  # no samples lost

    def test_mixed_lengths(self):
        """Mixed lengths should produce multiple groups."""
        lengths = [100, 200, 500, 1000, 2000, 4000, 8000]
        data_buffer = _make_samples(lengths)
        groups, overflow, is_done, skip = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=10,
        )
        assert not skip
        total_samples = sum(len(g) for g in groups) + len(overflow)
        assert total_samples == 7

    def test_no_data_loss(self):
        """All samples must be accounted for (in groups or overflow)."""
        lengths = [100, 500, 1000, 2000, 4000, 8000] * 5
        data_buffer = _make_samples(lengths)
        groups, overflow, is_done, skip = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=3,
        )
        total_samples = sum(len(g) for g in groups) + len(overflow)
        assert total_samples == 30

    def test_total_batch_size_set_on_all_samples(self):
        """total_batch_size should be set on every sample in each group."""
        lengths = [100, 200, 300, 400, 500]
        data_buffer = _make_samples(lengths)
        groups, overflow, _, _ = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=10,
        )
        for group in groups:
            tbs_values = [s[TOTAL_BATCH_SIZE_KEY] for s in group]
            # All samples in a group should have the same total_batch_size
            assert len(set(tbs_values)) == 1
            # total_batch_size should equal group size (non-DDP)
            assert tbs_values[0] == len(group)

    def test_empty_buffer(self):
        groups, overflow, is_done, skip = grouping_data(
            [], max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=10,
        )
        assert groups == []
        assert overflow == []

    def test_is_finished_flag(self):
        lengths = [100, 200]
        data_buffer = _make_samples(lengths)
        _, _, is_done, _ = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=True, idx_budget=10,
        )
        assert is_done is True

    def test_overflow_recycling(self):
        """When idx_budget < number of groups, excess goes to overflow."""
        # Create many different-length sequences to force many groups
        lengths = [8000, 7000, 6000, 5000, 4000]  # each ~1 per group
        data_buffer = _make_samples(lengths)
        groups, overflow, _, _ = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=2,
        )
        assert len(groups) <= 2
        assert len(overflow) > 0
        total = sum(len(g) for g in groups) + len(overflow)
        assert total == 5

    def test_single_sample(self):
        data_buffer = _make_samples([1000])
        groups, overflow, _, _ = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=10,
        )
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert overflow == []

    def test_empty_input_ids_filtered(self):
        """Samples with empty input_ids should be filtered out."""
        data_buffer = [
            [{"input_ids": torch.zeros(1, 0)}],  # empty
            [{"input_ids": torch.zeros(1, 100)}],  # valid
        ]
        groups, overflow, _, _ = grouping_data(
            data_buffer, max_input_length=8192, ddp_group=None,
            is_finished=False, idx_budget=10,
        )
        total = sum(len(g) for g in groups) + len(overflow)
        assert total == 1  # only the valid sample
