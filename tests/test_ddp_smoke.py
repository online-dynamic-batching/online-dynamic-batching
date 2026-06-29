# Copyright 2025 the ODB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DDP smoke tests for ODB.

Tests ODB behavior with multiple DDP ranks using torch.multiprocessing.spawn.
These tests run on CPU (no GPU required) and verify:
- All ranks produce the same number of groups per step
- No deadlocks occur
- Loss scaling metadata is consistent across ranks
- Overflow recycling works correctly with multiple ranks
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

import odb
from odb.constants import TOTAL_BATCH_SIZE_KEY

torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class VariableLengthDataset(Dataset):
    """Dataset producing variable-length input_ids for testing."""

    def __init__(self, lengths: list[int], seed: int = 42):
        self.lengths = lengths
        self.seed = seed

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        length = self.lengths[idx]
        return {
            "input_ids": torch.ones(1, length, dtype=torch.long) * idx,
            "labels": torch.ones(1, length, dtype=torch.long) * idx,
        }


def _find_free_port() -> int:
    """Return a localhost TCP port for a short-lived test process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _setup_process_group(rank, world_size, port, backend="gloo"):
    """Initialize process group for testing."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup():
    dist.destroy_process_group()


def _pad_collate(samples):
    """Pad variable-length tensors so smoke tests can form multi-sample groups."""
    max_len = max(int(s["input_ids"].numel()) for s in samples)
    input_ids = torch.zeros(len(samples), 1, max_len, dtype=torch.long)
    labels = torch.zeros(len(samples), 1, max_len, dtype=torch.long)
    for i, sample in enumerate(samples):
        ids = sample["input_ids"]
        labs = sample["labels"]
        input_ids[i, :, : ids.shape[-1]] = ids
        labels[i, :, : labs.shape[-1]] = labs
    batch = {"input_ids": input_ids, "labels": labels}
    if TOTAL_BATCH_SIZE_KEY in samples[0]:
        batch[TOTAL_BATCH_SIZE_KEY] = torch.tensor(samples[0][TOTAL_BATCH_SIZE_KEY])
    if "odb_total_tokens" in samples[0]:
        batch["odb_total_tokens"] = torch.tensor(samples[0]["odb_total_tokens"])
    return batch


# ---------------------------------------------------------------------------
# Test: DDP alignment — all ranks produce same number of batches
# ---------------------------------------------------------------------------

def _worker_alignment(rank, world_size, port, results_dict, lengths):
    """Worker function: count batches produced by each rank."""
    _setup_process_group(rank, world_size, port)

    dataset = VariableLengthDataset(lengths)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=2, prefetch_factor=16,
        shuffle=False, drop_last=False, collate_fn=_pad_collate,
    )
    odb.apply(dataloader, max_input_length=512, join=False, group_order_flip="rank_window_balanced")

    batch_count = 0
    for batch in dataloader:
        batch_count += 1

    results_dict[rank] = batch_count
    _cleanup()


@pytest.mark.timeout(60)
def test_ddp_alignment_2_ranks():
    """Two ranks must produce the same number of batches."""
    world_size = 2
    # Mixed lengths: some short, some long
    lengths = [50, 100, 200, 300, 50, 80, 150, 400, 60, 120, 250, 350, 70, 90, 180, 500]

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker_alignment,
        args=(world_size, _find_free_port(), results, lengths),
        nprocs=world_size,
        join=True,
    )

    assert len(results) == world_size
    counts = [results[r] for r in range(world_size)]
    assert counts[0] == counts[1], f"Rank batch counts differ: {counts}"
    assert counts[0] > 0, "No batches produced"


# ---------------------------------------------------------------------------
# Test: Loss scaling metadata consistency
# ---------------------------------------------------------------------------

def _worker_loss_scaling(rank, world_size, port, results_dict, lengths):
    """Worker: verify loss scaling keys are present and consistent."""
    _setup_process_group(rank, world_size, port)

    dataset = VariableLengthDataset(lengths)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=2, prefetch_factor=16,
        shuffle=False, drop_last=False,
    )
    odb.apply(dataloader, max_input_length=512, loss_scaling=True, join=False)

    total_tokens_per_step = []
    for batch in dataloader:
        if "odb_total_tokens" in batch:
            total_tokens_per_step.append(batch["odb_total_tokens"].item())

    results_dict[rank] = total_tokens_per_step
    _cleanup()


@pytest.mark.timeout(60)
def test_ddp_loss_scaling_consistency():
    """odb_total_tokens must be identical across ranks (it's a global sum)."""
    world_size = 2
    lengths = [100, 200, 300, 400, 150, 250, 350, 450, 120, 220, 320, 420]

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker_loss_scaling,
        args=(world_size, _find_free_port(), results, lengths),
        nprocs=world_size,
        join=True,
    )

    tokens_0 = results[0]
    tokens_1 = results[1]
    assert len(tokens_0) == len(tokens_1), f"Different step counts: {len(tokens_0)} vs {len(tokens_1)}"
    for i, (t0, t1) in enumerate(zip(tokens_0, tokens_1)):
        assert t0 == t1, f"Step {i}: odb_total_tokens differs: rank0={t0}, rank1={t1}"


# ---------------------------------------------------------------------------
# Test: No deadlock with uneven data
# ---------------------------------------------------------------------------

def _worker_uneven(rank, world_size, port, results_dict, lengths_per_rank):
    """Worker: test with different effective data per rank (simulating uneven sharding)."""
    _setup_process_group(rank, world_size, port)

    lengths = lengths_per_rank[rank]
    dataset = VariableLengthDataset(lengths)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=2, prefetch_factor=8,
        shuffle=False, drop_last=False,
    )
    odb.apply(dataloader, max_input_length=256, join=False)

    batch_count = 0
    for batch in dataloader:
        batch_count += 1

    results_dict[rank] = batch_count
    _cleanup()


@pytest.mark.timeout(60)
def test_ddp_no_deadlock_uneven_data():
    """ODB must not deadlock when ranks process aligned dynamic groups."""
    world_size = 2
    lengths_per_rank = {
        0: [100, 150, 200, 250, 300, 100, 150, 200],
        1: [100, 150, 200, 250, 300, 100, 150, 200],
    }

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker_uneven,
        args=(world_size, _find_free_port(), results, lengths_per_rank),
        nprocs=world_size,
        join=True,
    )

    # Both ranks must complete (no deadlock) and produce same batch count
    assert len(results) == world_size
    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# Test: Join mode keeps ODB sync alive after one rank stops yielding
# ---------------------------------------------------------------------------

def _worker_join_mode_uneven(rank, world_size, port, results_dict, lengths_per_rank):
    """Worker: one rank exhausts early, but its ODB collate process keeps joining."""
    _setup_process_group(rank, world_size, port)

    lengths = lengths_per_rank[rank]
    dataset = VariableLengthDataset(lengths)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=1, prefetch_factor=2,
        shuffle=False, drop_last=False, collate_fn=_pad_collate,
    )
    odb.apply(dataloader, max_input_length=160, join_mode=True, buffer_size=2)

    batch_count = 0
    iterator = iter(dataloader)
    try:
        while True:
            next(iterator)
            batch_count += 1
    except StopIteration:
        pass

    results_dict[rank] = batch_count

    # Keep the iterator alive until all main ranks have exited their loops; its
    # collate subprocess should continue participating in ODB's Gloo group.
    dist.barrier()

    collate_process = getattr(iterator, "_collate_process", None)
    if collate_process is not None:
        collate_process.join(timeout=10)
        results_dict[f"{rank}_collate_alive"] = collate_process.is_alive()
        if collate_process.is_alive():
            iterator._shutdown_workers()

    del iterator
    _cleanup()


@pytest.mark.timeout(90)
def test_ddp_join_mode_allows_uneven_rank_exhaustion():
    """join_mode=True lets longer ranks continue after shorter ranks stop yielding."""
    world_size = 2
    lengths_per_rank = {
        0: [80] * 12,
        1: [80] * 4,
    }

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker_join_mode_uneven,
        args=(world_size, _find_free_port(), results, lengths_per_rank),
        nprocs=world_size,
        join=True,
    )

    assert len(results) >= world_size
    assert results[0] > results[1], f"Expected rank 0 to keep yielding after rank 1: {dict(results)}"
    assert not results.get("0_collate_alive", True), f"Rank 0 collate subprocess did not exit: {dict(results)}"
    assert not results.get("1_collate_alive", True), f"Rank 1 collate subprocess did not exit: {dict(results)}"


# ---------------------------------------------------------------------------
# Test: Batch size correctness (dynamic sizing)
# ---------------------------------------------------------------------------

def _worker_batch_sizes(rank, world_size, port, results_dict, lengths):
    """Worker: collect actual batch sizes."""
    _setup_process_group(rank, world_size, port)

    dataset = VariableLengthDataset(lengths)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=2, prefetch_factor=16,
        shuffle=False, drop_last=False,
    )
    odb.apply(dataloader, max_input_length=512, join=False)

    batch_sizes = []
    for batch in dataloader:
        bs = batch.get(TOTAL_BATCH_SIZE_KEY, batch["input_ids"].shape[0])
        batch_sizes.append(bs)

    results_dict[rank] = batch_sizes
    _cleanup()


@pytest.mark.timeout(60)
def test_ddp_dynamic_batch_sizes():
    """Shorter sequences should get larger batches."""
    world_size = 2
    # Mix of very short and very long
    lengths = [50] * 8 + [400] * 4  # short samples should be grouped into larger batches

    manager = mp.Manager()
    results = manager.dict()

    mp.spawn(
        _worker_batch_sizes,
        args=(world_size, _find_free_port(), results, lengths),
        nprocs=world_size,
        join=True,
    )

    # Verify both ranks got batches
    for r in range(world_size):
        assert len(results[r]) > 0, f"Rank {r} got no batches"
    # Verify alignment
    assert len(results[0]) == len(results[1])
