# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end interface matrix tests for ODB and trainer integration modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import odb
from odb.integrations.hf import ODBTrainerMixin, configure_trainer


ODBMode = Literal["odb_dataloader", "apply_config", "apply_legacy"]
TrainerMode = Literal["manual_contract", "configure_existing", "native_mixin"]


class _VariableLengthDataset(Dataset):
    lengths = (1, 2, 3, 4, 5, 6)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        length = self.lengths[index]
        value = float(index + 1)
        return {"input_ids": torch.full((length,), value)}


def _pad_collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(sample["input_ids"].numel() for sample in samples)
    input_ids = torch.zeros(len(samples), max_len)
    for row, sample in enumerate(samples):
        value = sample["input_ids"]
        input_ids[row, : value.numel()] = value
    return {"input_ids": input_ids}


def _make_dataloader(mode: ODBMode) -> tuple[DataLoader, odb.ODBHandle]:
    dataset = _VariableLengthDataset()
    if mode == "odb_dataloader":
        dataloader = odb.ODBDataLoader(
            dataset,
            token_budget=8,
            loss_scaling="approx",
            batch_size=1,
            num_workers=1,
            collate_fn=_pad_collate,
        )
        return dataloader, dataloader.odb_handle

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=_pad_collate)
    if mode == "apply_config":
        handle = odb.apply(
            dataloader,
            config=odb.ODBConfig(token_budget=8, loss_scaling="approx"),
        )
    elif mode == "apply_legacy":
        handle = odb.apply(
            dataloader,
            max_input_length=8,
            loss_scaling=True,
            loss_scaling_approx=True,
        )
    else:
        raise AssertionError(f"unexpected ODB mode: {mode}")
    return dataloader, handle


@dataclass
class _Args:
    max_steps: int = -1


class _State:
    def __init__(self) -> None:
        self.odb_step_infos: list[odb.ODBStepInfo] = []
        self.unscaled_losses: list[float] = []


class _BaseTrainer:
    def __init__(self) -> None:
        self.args = _Args()
        self.state = _State()
        self.callbacks = []

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def compute_loss(self, model, inputs, *args, **kwargs):
        assert odb.ODB_STEP_INFO_KEY not in inputs
        assert "total_batch_size" not in inputs
        assert "local_batch_size" not in inputs
        assert "odb_local_tokens" not in inputs
        assert "odb_total_tokens" not in inputs
        loss = inputs["input_ids"].float().mean()
        self.state.unscaled_losses.append(float(loss.item()))
        return loss


class _NativeTrainer(ODBTrainerMixin, _BaseTrainer):
    pass


def _handle(loss_scaling: str = "approx") -> odb.ODBHandle:
    return odb.ODBHandle(
        config=odb.ODBConfig(token_budget=8, loss_scaling=loss_scaling),
        step_info_key=odb.ODB_STEP_INFO_KEY,
    )


def _run_training_window(odb_mode: ODBMode, trainer_mode: TrainerMode) -> list[dict[str, object]]:
    dataloader, handle = _make_dataloader(odb_mode)

    trainer: _BaseTrainer | _NativeTrainer | None = None
    if trainer_mode == "configure_existing":
        trainer = _BaseTrainer()
        configure_trainer(
            trainer,
            handle=handle,
            sample_budget=len(_VariableLengthDataset()),
            max_steps_policy="overwrite",
        )
    elif trainer_mode == "native_mixin":
        trainer = _NativeTrainer()
        configure_trainer(
            trainer,
            handle=handle,
            sample_budget=len(_VariableLengthDataset()),
            max_steps_policy="overwrite",
        )

    records = []
    for batch in dataloader:
        if trainer_mode == "manual_contract":
            info = odb.pop_step_info(batch, loss_scaling=handle.config.loss_scaling)
            unscaled_loss = batch["input_ids"].float().mean()
            scaled_loss = unscaled_loss * info.loss_scale
        else:
            assert trainer is not None
            scaled_loss = trainer.compute_loss(None, batch)
            info = trainer.state.odb_step_infos[-1]
            unscaled_loss = torch.tensor(trainer.state.unscaled_losses[-1])

        records.append(
            {
                "shape": tuple(batch["input_ids"].shape),
                "all_samples_this_step": int(info.all_samples_this_step),
                "loss_scale": float(info.loss_scale),
                "unscaled_loss": float(unscaled_loss.item()),
                "scaled_loss": float(scaled_loss.item()),
            }
        )

    return records


@pytest.mark.parametrize("odb_mode", ["odb_dataloader", "apply_config", "apply_legacy"])
@pytest.mark.parametrize("trainer_mode", ["manual_contract", "configure_existing", "native_mixin"])
def test_odb_and_trainer_interface_matrix_aligns_loss(odb_mode: ODBMode, trainer_mode: TrainerMode):
    reference = _run_training_window("apply_config", "manual_contract")
    actual = _run_training_window(odb_mode, trainer_mode)

    assert actual == reference


@pytest.mark.parametrize("trainer_mode", ["manual_contract", "configure_existing", "native_mixin"])
def test_trainer_modes_apply_reserved_step_info_loss_scale(trainer_mode: TrainerMode):
    batch = {
        "input_ids": torch.ones(2, 4),
        odb.ODB_STEP_INFO_KEY: odb.ODBStepInfo(all_samples_this_step=7, loss_scale=0.5),
    }

    if trainer_mode == "manual_contract":
        info = odb.pop_step_info(batch, loss_scaling="exact")
        unscaled_loss = batch["input_ids"].float().mean()
        scaled_loss = unscaled_loss * info.loss_scale
    elif trainer_mode == "configure_existing":
        trainer = _BaseTrainer()
        configure_trainer(
            trainer,
            handle=_handle("exact"),
            sample_budget=7,
            max_steps_policy="overwrite",
        )
        scaled_loss = trainer.compute_loss(None, batch)
        info = trainer.state.odb_step_infos[-1]
    elif trainer_mode == "native_mixin":
        trainer = _NativeTrainer()
        configure_trainer(
            trainer,
            handle=_handle("exact"),
            sample_budget=7,
            max_steps_policy="overwrite",
        )
        scaled_loss = trainer.compute_loss(None, batch)
        info = trainer.state.odb_step_infos[-1]
    else:
        raise AssertionError(f"unexpected trainer mode: {trainer_mode}")

    assert info.all_samples_this_step == 7
    assert scaled_loss.item() == pytest.approx(0.5)
    assert list(batch) == ["input_ids"]
