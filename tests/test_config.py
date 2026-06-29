# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for clean ODB configuration and apply aliases."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import odb
from odb.config import normalize_loss_scaling, resolve_config
from odb.handle import ODBHandle


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"input_ids": torch.ones(1, 8, dtype=torch.long) * idx}


def _loader() -> DataLoader:
    return DataLoader(_TinyDataset(), batch_size=1, num_workers=1)


def test_token_budget_call_returns_handle():
    handle = odb.apply(_loader(), token_budget=128)
    assert isinstance(handle, ODBHandle)
    assert handle.token_budget == 128
    assert handle.config.join is True
    assert handle.config.loss_scaling == "none"
    assert handle.step_info_key == odb.ODB_STEP_INFO_KEY


def test_join_defaults_to_true_and_can_be_disabled():
    assert resolve_config(token_budget=128).join is True
    assert resolve_config(token_budget=128, join=False).join is False
    assert resolve_config(max_input_length=128, join_mode=False).join is False


def test_legacy_max_input_length_resolves_to_token_budget():
    legacy = resolve_config(max_input_length=128)
    clean = resolve_config(token_budget=128)
    assert legacy == clean


def test_config_call_sets_dataloader_attrs():
    dataloader = _loader()
    config = odb.ODBConfig(token_budget=256, join=True, loss_scaling="exact", buffer_size=4)
    handle = odb.apply(dataloader, config=config)
    assert handle.config == config
    assert dataloader._odb_config == config
    assert dataloader._odb_join_mode is True
    assert dataloader._odb_effective_join_mode is False
    assert dataloader._odb_loss_scaling_mode == "exact"


def test_apply_enables_worker_prefetching_for_zero_worker_loader():
    dataloader = DataLoader(_TinyDataset(), batch_size=1, num_workers=0)
    handle = odb.apply(dataloader, token_budget=128)
    assert isinstance(handle, ODBHandle)
    assert dataloader.num_workers == 4
    assert dataloader.prefetch_factor == 2


def test_conflicting_token_budget_raises():
    config = odb.ODBConfig(token_budget=128)
    with pytest.raises(ValueError, match="token_budget"):
        odb.apply(_loader(), token_budget=256, config=config)


def test_loss_scaling_normalization():
    assert normalize_loss_scaling(None) == "none"
    assert normalize_loss_scaling(False) == "none"
    assert normalize_loss_scaling(True, True) == "approx"
    assert normalize_loss_scaling(True, False) == "exact"
    assert normalize_loss_scaling("token_approx") == "approx"
    assert normalize_loss_scaling("token_exact") == "exact"
