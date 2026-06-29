# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the LLaMA-Factory integration shim."""

from __future__ import annotations

import pytest

import odb.integrations.llamafactory as lf
from odb.integrations.llavafactory import configure_trainer as configure_llavafactory_trainer
from odb.integrations.llavafactory import enable_odb as enable_llavafactory_odb


class _Args:
    def __init__(
        self,
        *,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        max_steps=-1,
        odb_token_budget=None,
        odb_loss_scaling=None,
        odb_join=None,
    ):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.odb_token_budget = odb_token_budget
        self.odb_loss_scaling = odb_loss_scaling
        self.odb_join = odb_join


class _Trainer:
    def __init__(self, args, dataset):
        self.args = args
        self.train_dataset = dataset


class _ReadyDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {"input_ids": [idx, idx + 1], "labels": [idx, idx + 1]}


class _RawDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {"text": f"sample {idx}"}


class _FakeDataLoader:
    def __init__(self, dataset, *, batch_size=1, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


def test_llamafactory_configure_trainer_derives_budget_and_args(monkeypatch):
    calls = {}

    def fake_hf_configure_trainer(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(lf, "configure_hf_trainer", fake_hf_configure_trainer)

    args = _Args(odb_token_budget=8192, odb_loss_scaling="exact", odb_join=True)
    trainer = _Trainer(args=args, dataset=range(10))

    bridge = lf.configure_trainer(trainer, train_dataloader="loader")

    assert bridge == "bridge"
    assert calls["args"] == (trainer,)
    assert calls["kwargs"]["dataloader"] == "loader"
    assert calls["kwargs"]["sample_budget"] == 20
    assert calls["kwargs"]["max_optimizer_steps"] is None
    assert calls["kwargs"]["token_budget"] == 8192
    assert calls["kwargs"]["loss_scaling"] == "exact"
    assert calls["kwargs"]["join"] is True
    assert calls["kwargs"]["max_steps_policy"] == "overwrite"


def test_llamafactory_configure_trainer_maps_existing_max_steps(monkeypatch):
    calls = {}

    def fake_hf_configure_trainer(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(lf, "configure_hf_trainer", fake_hf_configure_trainer)

    trainer = _Trainer(args=_Args(max_steps=7), dataset=range(10))

    lf.configure_trainer(trainer, dataloader="loader", token_budget=4096)

    assert calls["kwargs"]["sample_budget"] == 20
    assert calls["kwargs"]["max_optimizer_steps"] == 7


def test_llamafactory_configure_trainer_rejects_non_unit_batch_size():
    trainer = _Trainer(args=_Args(per_device_train_batch_size=2), dataset=range(10))

    with pytest.raises(ValueError, match="per_device_train_batch_size=1"):
        lf.configure_trainer(trainer, dataloader="loader", token_budget=4096)


def test_llamafactory_configure_trainer_requires_budget_source():
    trainer = _Trainer(args=_Args(), dataset=None)

    with pytest.raises(ValueError, match="could not infer dataset size"):
        lf.configure_trainer(trainer, dataloader="loader", token_budget=4096)


def test_llamafactory_configure_trainer_rejects_apply_args_with_handle():
    trainer = _Trainer(args=_Args(), dataset=range(10))

    with pytest.raises(ValueError, match="handle is already provided"):
        lf.configure_trainer(trainer, handle=object(), token_budget=4096)


def test_llavafactory_alias_points_to_llamafactory_adapter():
    assert configure_llavafactory_trainer is lf.configure_trainer


def test_llamafactory_enable_odb_defaults_to_exact_loss_scaling_and_join(monkeypatch):
    calls = {}

    def fake_hf_configure_trainer(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(lf, "configure_hf_trainer", fake_hf_configure_trainer)

    dataset = _ReadyDataset()
    dataloader = _FakeDataLoader(dataset)
    trainer = _Trainer(args=_Args(odb_token_budget=8192), dataset=dataset)

    bridge = lf.enable_odb(
        trainer,
        train_dataloader=dataloader,
        train_dataset=dataset,
    )

    assert bridge == "bridge"
    assert calls["args"] == (trainer,)
    assert calls["kwargs"]["dataloader"] is dataloader
    assert calls["kwargs"]["sample_budget"] == 6
    assert calls["kwargs"]["token_budget"] == 8192
    assert calls["kwargs"]["loss_scaling"] == "exact"
    assert calls["kwargs"]["join"] is True
    assert calls["kwargs"]["max_steps_policy"] == "overwrite"


def test_llamafactory_enable_odb_framework_integration_applies_loader_without_hf_callback(monkeypatch):
    calls = {}
    handle = object()

    def fake_hf_configure_trainer(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("framework integration must not call the generic HF trainer adapter")

    def fake_apply(dataloader, **kwargs):
        calls["dataloader"] = dataloader
        calls["kwargs"] = kwargs
        return handle

    monkeypatch.setattr(lf, "configure_hf_trainer", fake_hf_configure_trainer)
    monkeypatch.setattr(lf, "apply_odb", fake_apply)

    dataset = _ReadyDataset()
    dataloader = _FakeDataLoader(dataset)
    trainer = _Trainer(args=_Args(odb_token_budget=8192), dataset=dataset)

    bridge = lf.enable_odb(
        trainer,
        train_dataloader=dataloader,
        train_dataset=dataset,
        trainer_integration="framework",
    )

    assert bridge.trainer is trainer
    assert bridge.handle is handle
    assert bridge.sample_budget == 6
    assert calls["dataloader"] is dataloader
    assert calls["kwargs"]["token_budget"] == 8192
    assert calls["kwargs"]["loss_scaling"] == "exact"
    assert calls["kwargs"]["join"] is True


def test_llamafactory_enable_odb_rejects_non_unit_dataloader_batch_size():
    dataset = _ReadyDataset()
    trainer = _Trainer(args=_Args(odb_token_budget=8192), dataset=dataset)
    dataloader = _FakeDataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="DataLoader batch_size=1"):
        lf.enable_odb(trainer, train_dataloader=dataloader, train_dataset=dataset)


def test_llamafactory_enable_odb_rejects_zero_workers():
    dataset = _ReadyDataset()
    trainer = _Trainer(args=_Args(odb_token_budget=8192), dataset=dataset)
    dataloader = _FakeDataLoader(dataset, num_workers=0)

    with pytest.raises(ValueError, match="worker prefetching"):
        lf.enable_odb(trainer, train_dataloader=dataloader, train_dataset=dataset)


def test_llamafactory_enable_odb_rejects_processor_in_collator_pipeline():
    dataset = _RawDataset()
    trainer = _Trainer(args=_Args(odb_token_budget=8192), dataset=dataset)
    dataloader = _FakeDataLoader(dataset)

    with pytest.raises(ValueError, match="must already contain input_ids"):
        lf.enable_odb(trainer, train_dataloader=dataloader, train_dataset=dataset)


def test_llamafactory_enable_odb_validates_actual_dataloader_dataset(monkeypatch):
    calls = {}

    def fake_hf_configure_trainer(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "bridge"

    monkeypatch.setattr(lf, "configure_hf_trainer", fake_hf_configure_trainer)

    ready_dataset = _ReadyDataset()
    raw_train_dataset = _RawDataset()
    trainer = _Trainer(args=_Args(odb_token_budget=8192), dataset=raw_train_dataset)
    dataloader = _FakeDataLoader(ready_dataset)

    bridge = lf.enable_odb(
        trainer,
        train_dataloader=dataloader,
        train_dataset=raw_train_dataset,
        sample_budget=3,
    )

    assert bridge == "bridge"
    assert calls["kwargs"]["dataloader"] is dataloader
    assert calls["kwargs"]["sample_budget"] == 3


def test_llavafactory_enable_odb_alias_points_to_llamafactory_adapter():
    assert enable_llavafactory_odb is lf.enable_odb
