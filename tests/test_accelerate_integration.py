# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for Accelerate integration helpers without requiring Accelerate."""

from contextlib import contextmanager

import torch

import odb
from odb.integrations.accelerate import ODBAccelerateBridge, configure_accelerator


class _FakeAccelerator:
    def __init__(self):
        self.backward_losses = []
        self.prepared = []
        self.joined = False

    def backward(self, loss, **kwargs):
        self.backward_losses.append((loss, kwargs))

    def prepare(self, dataloader):
        self.prepared.append(dataloader)
        return ("prepared", dataloader)

    @contextmanager
    def join_uneven_inputs(self, joinables, **kwargs):
        self.joined = (tuple(joinables), kwargs)
        yield "joined"


def _handle(loss_scaling: str = "exact") -> odb.ODBHandle:
    return odb.ODBHandle(
        config=odb.ODBConfig(token_budget=8, loss_scaling=loss_scaling),
        step_info_key=odb.ODB_STEP_INFO_KEY,
    )


def test_accelerate_bridge_consumes_metadata_scales_loss_and_tracks_budget():
    accelerator = _FakeAccelerator()
    bridge = ODBAccelerateBridge(accelerator, dataloader=None, handle=_handle(), sample_budget=7)
    batch = {
        "input_ids": torch.ones(2, 4),
        odb.ODB_STEP_INFO_KEY: odb.ODBStepInfo(all_samples_this_step=7, loss_scale=0.5),
    }

    info = bridge.consume_batch(batch)
    scaled = bridge.backward(torch.tensor(2.0), info=info, marker=True)

    assert list(batch) == ["input_ids"]
    assert info.all_samples_this_step == 7
    assert scaled.item() == 1.0
    assert accelerator.backward_losses[0][0].item() == 1.0
    assert accelerator.backward_losses[0][1] == {"marker": True}
    assert bridge.should_stop


def test_configure_accelerator_can_prepare_dataloader_with_existing_handle():
    accelerator = _FakeAccelerator()
    dataloader = object()

    bridge = configure_accelerator(
        accelerator,
        dataloader,
        handle=_handle("none"),
        sample_budget=10,
        prepare_dataloader=True,
    )

    assert accelerator.prepared == [dataloader]
    assert bridge.dataloader == ("prepared", dataloader)
    assert bridge.sample_budget == 10


def test_accelerate_join_uneven_inputs_delegates_when_available():
    accelerator = _FakeAccelerator()
    bridge = ODBAccelerateBridge(accelerator, dataloader=None, handle=_handle())

    with bridge.join_uneven_inputs(["model"], even_batches=False) as marker:
        assert marker == "joined"

    assert accelerator.joined == (("model",), {"even_batches": False})
