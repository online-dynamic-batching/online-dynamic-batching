# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ODBStepInfo extraction."""

import torch

import odb


def test_pop_step_info_reads_legacy_sample_keys_and_removes_metadata():
    batch = {
        "input_ids": torch.ones(2, 8),
        "local_batch_size": torch.tensor(2),
        "total_batch_size": torch.tensor(8),
    }
    info = odb.pop_step_info(batch, loss_scaling="auto", world_size=4)
    assert info.all_samples_this_step == 8
    assert info.loss_scale == 1.0
    assert "local_batch_size" not in batch
    assert "total_batch_size" not in batch


def test_pop_step_info_token_loss_scale():
    batch = {
        "input_ids": torch.ones(2, 8),
        "local_batch_size": torch.tensor(2),
        "total_batch_size": torch.tensor(8),
        "odb_local_tokens": torch.tensor(20),
        "odb_total_tokens": torch.tensor(100),
    }
    info = odb.pop_step_info(batch, loss_scaling="approx", world_size=4)
    assert info.all_samples_this_step == 8
    assert info.loss_scale == 0.8
    assert "odb_local_tokens" not in batch
    assert "odb_total_tokens" not in batch


def test_pop_step_info_none_loss_scaling_keeps_identity_scale():
    batch = {
        "input_ids": torch.ones(2, 8),
        "local_batch_size": 2,
        "total_batch_size": 8,
        "odb_local_tokens": 20,
        "odb_total_tokens": 100,
    }
    info = odb.pop_step_info(batch, loss_scaling="none", world_size=4)
    assert info.all_samples_this_step == 8
    assert info.loss_scale == 1.0


def test_pop_step_info_prefers_reserved_step_info():
    batch = {
        "input_ids": torch.ones(2, 8),
        odb.ODB_STEP_INFO_KEY: odb.ODBStepInfo(all_samples_this_step=11, loss_scale=0.5),
        "total_batch_size": 8,
    }
    info = odb.pop_step_info(batch)
    assert info == odb.ODBStepInfo(all_samples_this_step=11, loss_scale=0.5)
    assert odb.ODB_STEP_INFO_KEY not in batch
    assert "total_batch_size" not in batch
