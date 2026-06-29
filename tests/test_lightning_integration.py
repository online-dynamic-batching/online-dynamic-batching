# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for Lightning integration helpers without requiring Lightning."""

import torch

import odb
from odb.integrations.lightning import ODBLightningCallback, ODBLightningMixin, configure_lightning_module


def _handle(loss_scaling: str = "exact") -> odb.ODBHandle:
    return odb.ODBHandle(
        config=odb.ODBConfig(token_budget=8, loss_scaling=loss_scaling),
        step_info_key=odb.ODB_STEP_INFO_KEY,
    )


class _FakeModule:
    def __init__(self):
        self.seen_batches = []

    def training_step(self, batch, batch_idx):
        assert odb.ODB_STEP_INFO_KEY not in batch
        assert "total_batch_size" not in batch
        self.seen_batches.append((dict(batch), batch_idx))
        return batch["input_ids"].float().mean()


class _FakeDictModule:
    def training_step(self, batch, batch_idx):
        return {"loss": batch["input_ids"].float().mean(), "batch_idx": batch_idx}


class _MixinModule(ODBLightningMixin):
    pass


def test_lightning_wrapper_pops_metadata_scales_loss_and_tracks_progress():
    module = _FakeModule()
    bridge = configure_lightning_module(module, handle=_handle(), sample_budget=6)
    batch = {
        "input_ids": torch.ones(2, 4),
        odb.ODB_STEP_INFO_KEY: odb.ODBStepInfo(all_samples_this_step=6, loss_scale=0.25),
    }

    loss = module.training_step(batch, 3)

    assert loss.item() == 0.25
    assert bridge.state.emitted_samples == 6
    assert module.odb_emitted_samples == 6
    assert module.odb_last_step_info.all_samples_this_step == 6
    assert bridge.should_stop
    assert list(batch) == ["input_ids"]


def test_lightning_wrapper_scales_dict_loss_only():
    module = _FakeDictModule()
    configure_lightning_module(module, handle=_handle())
    batch = {
        "input_ids": torch.ones(2, 4),
        odb.ODB_STEP_INFO_KEY: odb.ODBStepInfo(all_samples_this_step=2, loss_scale=0.5),
    }

    output = module.training_step(batch, 4)

    assert output["loss"].item() == 0.5
    assert output["batch_idx"] == 4


def test_lightning_mixin_explicit_contract():
    module = _MixinModule()
    module.set_odb_handle(_handle())
    batch = {
        "input_ids": torch.ones(2, 4),
        odb.ODB_STEP_INFO_KEY: odb.ODBStepInfo(all_samples_this_step=5, loss_scale=0.2),
    }

    info = module.consume_odb_batch(batch)
    loss = module.scale_odb_loss(torch.tensor(10.0), info=info)

    assert list(batch) == ["input_ids"]
    assert module.odb_emitted_samples == 5
    assert loss.item() == 2.0


def test_lightning_callback_stops_trainer_at_budget():
    class _Trainer:
        should_stop = False

    class _Module:
        odb_emitted_samples = 10

    trainer = _Trainer()
    module = _Module()

    ODBLightningCallback(sample_budget=10).on_train_batch_end(trainer, module, None, None, 0)

    assert trainer.should_stop is True
