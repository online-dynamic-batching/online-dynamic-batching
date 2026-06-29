# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for HuggingFace integration helpers that do not require transformers."""

from types import SimpleNamespace

import pytest
import torch

import odb.integrations.hf as hf
from odb.config import ODBConfig
from odb.constants import ODB_STEP_INFO_KEY
from odb.handle import ODBHandle
from odb.integrations.hf import ODBTrainerMixin, _set_trainer_max_steps, configure_trainer, enable_odb
from odb.step_info import ODBStepInfo


class _Args:
    def __init__(self, max_steps=-1, per_device_train_batch_size=1, num_train_epochs=1, odb_token_budget=None):
        self.max_steps = max_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.odb_token_budget = odb_token_budget


def test_set_trainer_max_steps_uses_sample_budget_when_no_cap():
    args = _Args(max_steps=-1)
    _set_trainer_max_steps(args, sample_budget=100, max_optimizer_steps=None, policy="error")
    assert args.max_steps == 100


def test_set_trainer_max_steps_uses_optimizer_cap_when_provided():
    args = _Args(max_steps=-1)
    _set_trainer_max_steps(args, sample_budget=100, max_optimizer_steps=7, policy="error")
    assert args.max_steps == 7


def test_set_trainer_max_steps_conflict_errors_by_default():
    args = _Args(max_steps=12)
    with pytest.raises(ValueError, match="max_steps"):
        _set_trainer_max_steps(args, sample_budget=100, max_optimizer_steps=None, policy="error")


def test_set_trainer_max_steps_preserve_policy():
    args = _Args(max_steps=12)
    _set_trainer_max_steps(args, sample_budget=100, max_optimizer_steps=None, policy="preserve")
    assert args.max_steps == 12


def test_set_trainer_max_steps_overwrite_policy():
    args = _Args(max_steps=12)
    _set_trainer_max_steps(args, sample_budget=100, max_optimizer_steps=None, policy="overwrite")
    assert args.max_steps == 100


class _BaseTrainer:
    def __init__(self):
        self.state = SimpleNamespace(odb_step_infos=[])
        self.callbacks = []
        self.train_dataloader_calls = 0
        self._default_dataloader = object()

    def compute_loss(self, model, inputs, *args, **kwargs):
        assert ODB_STEP_INFO_KEY not in inputs
        assert "total_batch_size" not in inputs
        assert "input_ids" in inputs
        return torch.tensor(2.0)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def get_train_dataloader(self):
        self.train_dataloader_calls += 1
        return self._default_dataloader


class _NativeTrainer(ODBTrainerMixin, _BaseTrainer):
    pass


class _ReadyDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {"input_ids": torch.ones(2), "labels": torch.ones(2)}


class _RawDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {"text": f"sample {idx}"}


class _DeclaredReadyDataset(_RawDataset):
    odb_ready = True


class _FakeDataLoader:
    def __init__(self, dataset, *, batch_size=1, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


def test_odb_trainer_mixin_pops_metadata_and_scales_loss():
    trainer = _NativeTrainer()
    trainer.set_odb_loss_scaling("approx")
    batch = {
        "input_ids": torch.ones(2, 8),
        "total_batch_size": 8,
        "odb_local_tokens": 20,
        "odb_total_tokens": 80,
    }

    loss = trainer.compute_loss(None, batch)

    assert loss.item() == 0.5
    assert trainer.state.odb_step_infos[-1].all_samples_this_step == 8
    assert list(batch) == ["input_ids"]
    assert torch.equal(batch["input_ids"], torch.ones(2, 8))


def test_odb_trainer_mixin_scales_reserved_step_info():
    trainer = _NativeTrainer()
    batch = {
        "input_ids": torch.ones(2, 8),
        ODB_STEP_INFO_KEY: ODBStepInfo(all_samples_this_step=6, loss_scale=0.5),
    }

    loss = trainer.compute_loss(None, batch)

    assert loss.item() == 1.0
    assert trainer.state.odb_step_infos[-1].all_samples_this_step == 6


def test_configure_trainer_uses_native_trainer_without_wrapping_compute_loss():
    trainer = _NativeTrainer()
    trainer.args = _Args(max_steps=-1)
    handle = ODBHandle(config=ODBConfig(token_budget=8192, loss_scaling="exact"), step_info_key=ODB_STEP_INFO_KEY)

    bridge = configure_trainer(
        trainer,
        handle=handle,
        sample_budget=100,
        max_steps_policy="overwrite",
    )

    assert bridge.handle is handle
    assert trainer.odb_loss_scaling == "exact"
    assert trainer.args.max_steps == 100
    assert len(trainer.callbacks) == 1


def test_configure_trainer_replaces_train_dataloader_by_default():
    class _FakeDataLoader:
        pass

    trainer = _BaseTrainer()
    trainer.args = _Args(max_steps=-1)
    dataloader = _FakeDataLoader()

    configure_trainer(
        trainer,
        dataloader=dataloader,
        handle=ODBHandle(config=ODBConfig(token_budget=8192), step_info_key=ODB_STEP_INFO_KEY),
        sample_budget=100,
        max_steps_policy="overwrite",
    )

    assert trainer.get_train_dataloader() is dataloader
    assert trainer.train_dataloader_calls == 0


def test_enable_odb_defaults_to_exact_loss_scaling_and_join(monkeypatch):
    calls = {}

    def fake_configure_trainer(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(hf, "configure_trainer", fake_configure_trainer)

    dataset = _ReadyDataset()
    dataloader = _FakeDataLoader(dataset)
    trainer = _BaseTrainer()
    trainer.args = _Args(num_train_epochs=2, odb_token_budget=8192)
    trainer.train_dataset = dataset

    bridge = enable_odb(trainer, train_dataloader=dataloader, train_dataset=dataset)

    assert bridge == "bridge"
    assert calls["args"] == (trainer,)
    assert calls["kwargs"]["dataloader"] is dataloader
    assert calls["kwargs"]["sample_budget"] == 6
    assert calls["kwargs"]["token_budget"] == 8192
    assert calls["kwargs"]["loss_scaling"] == "exact"
    assert calls["kwargs"]["join"] is True
    assert calls["kwargs"]["max_steps_policy"] == "overwrite"


def test_enable_odb_can_use_trainer_default_dataloader(monkeypatch):
    calls = {}

    def fake_configure_trainer(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(hf, "configure_trainer", fake_configure_trainer)

    dataset = _ReadyDataset()
    dataloader = _FakeDataLoader(dataset)
    trainer = _BaseTrainer()
    trainer.args = _Args(odb_token_budget=8192)
    trainer.train_dataset = dataset
    trainer._default_dataloader = dataloader

    bridge = enable_odb(trainer)

    assert bridge == "bridge"
    assert calls["kwargs"]["dataloader"] is dataloader
    assert trainer.train_dataloader_calls == 1


def test_enable_odb_rejects_non_unit_dataloader_batch_size():
    dataset = _ReadyDataset()
    trainer = _BaseTrainer()
    trainer.args = _Args(odb_token_budget=8192)

    with pytest.raises(ValueError, match="DataLoader batch_size=1"):
        enable_odb(trainer, train_dataloader=_FakeDataLoader(dataset, batch_size=2), train_dataset=dataset)


def test_enable_odb_rejects_zero_workers():
    dataset = _ReadyDataset()
    trainer = _BaseTrainer()
    trainer.args = _Args(odb_token_budget=8192)

    with pytest.raises(ValueError, match="worker prefetching"):
        enable_odb(trainer, train_dataloader=_FakeDataLoader(dataset, num_workers=0), train_dataset=dataset)


def test_enable_odb_rejects_processor_in_collator_pipeline():
    dataset = _RawDataset()
    trainer = _BaseTrainer()
    trainer.args = _Args(odb_token_budget=8192)

    with pytest.raises(ValueError, match="must already contain input_ids"):
        enable_odb(trainer, train_dataloader=_FakeDataLoader(dataset), train_dataset=dataset)


def test_enable_odb_accepts_dataset_declared_odb_ready(monkeypatch):
    calls = {}

    def fake_configure_trainer(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(hf, "configure_trainer", fake_configure_trainer)

    dataset = _DeclaredReadyDataset()
    trainer = _BaseTrainer()
    trainer.args = _Args(odb_token_budget=8192)

    bridge = enable_odb(trainer, train_dataloader=_FakeDataLoader(dataset), train_dataset=dataset)

    assert bridge == "bridge"
    assert calls["kwargs"]["dataloader"].dataset is dataset
