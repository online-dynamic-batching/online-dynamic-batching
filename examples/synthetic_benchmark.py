"""Synthetic ODB benchmark that runs without private data.

The script compares fixed-size DataLoader batching with ODB on a deterministic
long-tail sequence distribution. It is intentionally small: the goal is to
verify the integration path and illustrate padding reduction, not to reproduce
paper-scale numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

import odb


class SyntheticSequenceDataset(Dataset):
    """Variable-length token dataset with a deterministic long tail."""

    def __init__(self, num_samples: int, *, vocab_size: int = 256, seed: int = 7) -> None:
        self.num_samples = int(num_samples)
        self.vocab_size = int(vocab_size)
        generator = torch.Generator().manual_seed(seed)
        short = torch.randint(64, 384, (int(num_samples * 0.72),), generator=generator)
        medium = torch.randint(384, 1400, (int(num_samples * 0.22),), generator=generator)
        long = torch.randint(1400, 4096, (num_samples - short.numel() - medium.numel(),), generator=generator)
        self.lengths = torch.cat([short, medium, long]).tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        length = int(self.lengths[idx])
        values = (torch.arange(length, dtype=torch.long) + idx) % self.vocab_size
        return {"input_ids": values, "labels": values.clone()}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = True
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.proj(self.embedding(input_ids))
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction="none")
        loss = loss.view_as(labels)
        return (loss * attention_mask).sum() / attention_mask.sum().clamp_min(1)


@dataclass
class RunResult:
    name: str
    seconds: float
    samples: int
    real_tokens: int
    padded_tokens: int
    steps: int
    final_loss: float

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds

    @property
    def real_tokens_per_second(self) -> float:
        return self.real_tokens / self.seconds

    @property
    def padding_ratio(self) -> float:
        return 1.0 - (self.real_tokens / max(self.padded_tokens, 1))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(name: str, dataloader, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> RunResult:
    model.train()
    start = time.perf_counter()
    samples = 0
    real_tokens = 0
    padded_tokens = 0
    steps = 0
    final_loss = math.nan

    for batch in dataloader:
        info = odb.pop_step_info(batch, loss_scaling="exact")
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model(**batch) * info.loss_scale
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        local_samples = int(batch["input_ids"].shape[0])
        samples += int(info.all_samples_this_step or local_samples)
        real_tokens += int(batch["attention_mask"].sum().item())
        padded_tokens += int(batch["input_ids"].numel())
        steps += 1
        final_loss = float(loss.detach().cpu().item())

    seconds = time.perf_counter() - start
    return RunResult(name, seconds, samples, real_tokens, padded_tokens, steps, final_loss)


def print_result(result: RunResult) -> None:
    print(
        f"{result.name:>9} | "
        f"{result.samples_per_second:8.2f} samples/s | "
        f"{result.real_tokens_per_second:10.0f} real tok/s | "
        f"padding {result.padding_ratio * 100:5.1f}% | "
        f"steps {result.steps:4d} | "
        f"loss {result.final_loss:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--fixed-batch-size", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")

    device = resolve_device(args.device)
    dataset = SyntheticSequenceDataset(args.num_samples)

    fixed_loader = DataLoader(
        dataset,
        batch_size=args.fixed_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate,
    )
    odb_loader = odb.ODBDataLoader(
        dataset,
        token_budget=args.token_budget,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        collate_fn=collate,
        loss_scaling="exact",
    )

    fixed_model = TinyLM().to(device)
    odb_model = TinyLM().to(device)
    odb_model.load_state_dict(fixed_model.state_dict())

    fixed_result = run_epoch(
        "fixed",
        fixed_loader,
        fixed_model,
        torch.optim.AdamW(fixed_model.parameters(), lr=1e-3),
        device,
    )
    odb_result = run_epoch(
        "odb",
        odb_loader,
        odb_model,
        torch.optim.AdamW(odb_model.parameters(), lr=1e-3),
        device,
    )

    print(f"device={device} samples={args.num_samples} token_budget={args.token_budget}")
    print_result(fixed_result)
    print_result(odb_result)
    print(f"speedup: {odb_result.samples_per_second / fixed_result.samples_per_second:.2f}x samples/s")


if __name__ == "__main__":
    main()
