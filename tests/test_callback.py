# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the ODBDynamicBatchCallback."""

import pytest

from odb.callbacks import ODBDynamicBatchCallback, _HF_AVAILABLE


pytestmark = pytest.mark.skipif(
    not _HF_AVAILABLE,
    reason="ODBDynamicBatchCallback tests require the optional transformers dependency",
)


class _FakeArgs:
    """Minimal args mock."""

    def __init__(self, num_train_epochs=1, max_steps=100):
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps


class _FakeState:
    """Minimal state mock."""

    def __init__(self):
        self.total_data_step = 0
        self.total_batch_sizes = []
        self.accumulation_batch_size = 0
        self.epoch = 0.0


class _FakeControl:
    """Minimal control mock."""

    def __init__(self):
        self.should_training_stop = False


class _FakeLRScheduler:
    def __init__(self):
        self.last_epoch = 0
        self._step_count = 0

    def step(self):
        self._step_count += 1
        self.last_epoch += 1


class TestODBDynamicBatchCallback:
    def test_on_train_begin_initializes_state(self):
        cb = ODBDynamicBatchCallback(odb_total_samples=1000)
        state = _FakeState()
        cb.on_train_begin(_FakeArgs(), state, _FakeControl())
        assert state.total_data_step == 0
        assert state.total_batch_sizes == []
        assert state.accumulation_batch_size == 0

    def test_epoch_tracking(self):
        """Epoch should be calculated from samples processed, not steps."""
        cb = ODBDynamicBatchCallback(odb_total_samples=100)
        args = _FakeArgs(num_train_epochs=1)
        state = _FakeState()
        control = _FakeControl()
        cb.on_train_begin(args, state, control)

        # Simulate: substep with batch_size=10
        state.total_batch_sizes.append(10)
        cb.on_substep_end(args, state, control)
        assert state.total_data_step == 10
        assert abs(state.epoch - 0.1) < 1e-6

        # Another substep with batch_size=20
        state.total_batch_sizes.append(20)
        cb.on_substep_end(args, state, control)
        assert state.total_data_step == 30
        assert abs(state.epoch - 0.3) < 1e-6

    def test_should_stop_at_total_samples(self):
        """Training should stop when total samples are processed."""
        cb = ODBDynamicBatchCallback(odb_total_samples=50)
        args = _FakeArgs(num_train_epochs=1)
        state = _FakeState()
        control = _FakeControl()
        cb.on_train_begin(args, state, control)

        # Process 30 samples
        state.total_batch_sizes.append(30)
        cb.on_substep_end(args, state, control)
        assert not control.should_training_stop

        # Process 25 more (total=55 >= 50)
        state.total_batch_sizes.append(25)
        cb.on_substep_end(args, state, control)
        assert control.should_training_stop

    def test_lr_scheduler_compensation(self):
        """LR scheduler should get extra steps to match sample progress."""
        cb = ODBDynamicBatchCallback(odb_total_samples=1000)
        args = _FakeArgs(num_train_epochs=1)
        state = _FakeState()
        control = _FakeControl()
        lr_sched = _FakeLRScheduler()
        cb.on_train_begin(args, state, control)

        # Substep 1: batch_size=5
        state.total_batch_sizes.append(5)
        cb.on_substep_end(args, state, control)

        # Substep 2: batch_size=3
        state.total_batch_sizes.append(3)
        cb.on_substep_end(args, state, control)

        # Step end: accumulation_batch_size = 5+3=8 → lr_scheduler.step() called 7 extra times
        cb.on_step_end(args, state, control, lr_scheduler=lr_sched)
        assert lr_sched._step_count == 7  # accumulation_batch_size - 1
        assert state.accumulation_batch_size == 0  # reset

    def test_multi_epoch(self):
        """Epoch calculation should work with num_train_epochs > 1."""
        cb = ODBDynamicBatchCallback(odb_total_samples=200)
        args = _FakeArgs(num_train_epochs=2)
        state = _FakeState()
        control = _FakeControl()
        cb.on_train_begin(args, state, control)

        # Process 100 samples → epoch should be 1.0
        state.total_batch_sizes.append(100)
        cb.on_substep_end(args, state, control)
        assert abs(state.epoch - 1.0) < 1e-6
        assert not control.should_training_stop

        # Process 100 more → epoch should be 2.0, training stops
        state.total_batch_sizes.append(100)
        cb.on_substep_end(args, state, control)
        assert abs(state.epoch - 2.0) < 1e-6
        assert control.should_training_stop

    def test_empty_batch_sizes_noop(self):
        """No-op when total_batch_sizes is empty."""
        cb = ODBDynamicBatchCallback(odb_total_samples=100)
        args = _FakeArgs()
        state = _FakeState()
        control = _FakeControl()
        cb.on_train_begin(args, state, control)

        # Call with empty queue — should not crash
        cb._update_step(args, state, control)
        assert state.total_data_step == 0

    def test_odb_total_samples_should_equal_dataset_size_times_epochs(self):
        """odb_total_samples = N * epochs, NOT N/(bs*ws) * epochs * ws.

        This test documents the correct semantics: odb_total_samples represents
        the total number of individual samples to process. When N=200 and
        epochs=2, processing all 400 samples should bring epoch to exactly 2.0.
        Variable batch sizes (simulating ODB dynamic batching with different bs
        per step) should not affect the final epoch calculation.
        """
        N = 200
        epochs = 2
        odb_total_samples = N * epochs  # = 400, correct

        cb = ODBDynamicBatchCallback(odb_total_samples=odb_total_samples)
        args = _FakeArgs(num_train_epochs=epochs)
        state = _FakeState()
        control = _FakeControl()
        cb.on_train_begin(args, state, control)

        # Simulate variable batch sizes (as in real ODB: bs differs per step)
        batch_sizes = [8, 12, 5, 15, 10, 20, 8, 12, 5, 5]  # sum=100 = N/2
        for bs in batch_sizes:
            state.total_batch_sizes.append(bs)
            cb.on_substep_end(args, state, control)

        assert state.total_data_step == 100
        assert abs(state.epoch - 0.5) < 1e-6  # half an epoch
        assert not control.should_training_stop

        # Process remaining 300 samples to complete 2 epochs
        state.total_batch_sizes.append(300)
        cb.on_substep_end(args, state, control)
        assert state.total_data_step == 400
        assert abs(state.epoch - 2.0) < 1e-6
        assert control.should_training_stop
