# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ODBDataLoader."""

import pytest
import torch
from torch.utils.data import Dataset

import odb


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"input_ids": torch.ones(1, 8, dtype=torch.long) * idx}


def test_odb_dataloader_constructs_with_handle():
    dataloader = odb.ODBDataLoader(_TinyDataset(), token_budget=128, num_workers=1)
    assert dataloader.batch_size == 1
    assert dataloader.odb_handle.token_budget == 128
    assert dataloader._odb_handle == dataloader.odb_handle


def test_odb_dataloader_defaults_to_worker_prefetching():
    dataloader = odb.ODBDataLoader(_TinyDataset(), token_budget=128)
    assert dataloader.batch_size == 1
    assert dataloader.num_workers == 4
    assert dataloader.prefetch_factor == 2


def test_odb_dataloader_default_workers_iterates():
    dataloader = odb.ODBDataLoader(_TinyDataset(), token_budget=128)
    batch = next(iter(dataloader))
    assert "input_ids" in batch


def test_odb_dataloader_accepts_config():
    config = odb.ODBConfig(token_budget=256, join=True, loss_scaling="approx")
    dataloader = odb.ODBDataLoader(_TinyDataset(), config=config, num_workers=1)
    assert dataloader.odb_handle.config == config
    assert dataloader._odb_join_mode is True


def test_odb_dataloader_rejects_non_one_batch_size():
    with pytest.raises(ValueError, match="batch_size=1"):
        odb.ODBDataLoader(_TinyDataset(), token_budget=128, batch_size=2, num_workers=1)
