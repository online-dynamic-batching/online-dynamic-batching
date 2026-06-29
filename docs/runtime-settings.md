# Runtime Settings

These settings are shared across ODB's PyTorch, Hugging Face Trainer,
LLaMA-Factory, Accelerate, and Lightning paths. Framework guides describe where
to attach ODB; this page describes runtime behavior that is common after ODB is
attached.

## Core Knobs

| Setting | Default | Purpose |
| --- | --- | --- |
| `token_budget` | required | Maximum total observed input length per dynamic group. |
| `loss_scaling` | `"none"` for raw DataLoader APIs; `"exact"` in high-level trainer hooks | Scales the local loss when ranks emit different local samples/tokens. |
| `join` | `True` | Keeps the ODB DataLoader/collate protocol alive while uneven ranks drain. |
| `buffer_size` | `num_workers * prefetch_factor` | Number of single samples available to the online grouping window. |
| `num_workers` / `prefetch_factor` | DataLoader-dependent | Worker prefetching that feeds the online grouping window. |
| `group_order_flip` | `"none"` | Advanced rank-wise post-alignment group emission order. |
| `no_warmup` | `False` | Controls ODB's buffer-fill warm-up ramp, not optimizer/LR warmup. |

## Token Budget

`token_budget` is ODB's primary batching knob. It limits the total observed
input length in each dynamic group; it is not the model's truncation
`max_length`.

Too small a budget leaves GPU memory underused. Too large a budget can increase
step time or trigger OOM. Start from the framework example closest to your
workload, then tune with real post-processor lengths and memory headroom.

Legacy name: `max_input_length`.

## Buffer And Prefetch

ODB groups samples after they have passed through the Dataset or single-sample
processor path. It therefore needs a small online window of already processed
samples.

Use `batch_size=1`, `num_workers > 0`, and a positive `prefetch_factor`.
`buffer_size` defaults to `num_workers * prefetch_factor`. Increasing it gives
ODB more length diversity but can increase startup latency and distributed
metadata traffic.

High-level integrations validate this contract where possible. If you call the
raw DataLoader APIs directly, make sure the DataLoader emits one fully processed
sample at a time.

## Loss Scaling

Dynamic batching can make each rank emit different local sample or token counts
at the same optimizer step. `loss_scaling="exact"` injects ODB metadata into the
batch and lets the trainer multiply the local loss by `info.loss_scale`.

Use `odb.pop_step_info(...)` before model forward so ODB transport metadata does
not reach `model(**batch)`.

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss * info.loss_scale
```

For DDP training and the public examples, prefer `loss_scaling="exact"`.
`"approx"` is kept for compatibility and experiments; `"none"` is appropriate
only when all local dynamic batches should contribute equally.

## Sample Progress

`info.all_samples_this_step` is the all-rank emitted sample count for the
current step. Use it for sample-budget stopping and sample-based scheduler
progress.

```python
emitted_samples += info.all_samples_this_step
if emitted_samples >= sample_budget:
    break
```

If your scheduler should follow optimizer steps instead of emitted samples, keep
that policy explicit in the framework adapter or training loop.

## Join

`join=True` is the default ODB DataLoader/collate runtime setting. It lets ranks
that finish their local input early keep participating in ODB's DataLoader-side
drain protocol while other ranks continue.

This is separate from model forward/backward collectives. In distributed runs,
pair ODB join with the framework's uneven-input guard when ranks can exhaust at
different optimizer steps:

- PyTorch DDP: DDP Join or an equivalent guard.
- Accelerate: `accelerator.join_uneven_inputs(...)` when available.
- Lightning: a DDP strategy or plugin that handles uneven inputs.

Most users should leave `join=True`.

## Group Order Flip

After local grouping and DDP alignment, ODB normally emits aligned groups in
their deterministic order: `group_order_flip="none"`.

`group_order_flip` is an advanced distributed setting that can reverse the
post-alignment group order on selected ranks. It is useful for experiments or
audits where rank-wise group order correlations matter. It does not change the
set of samples in the grouping window; it changes the order in which aligned
groups are emitted to the trainer.

Supported modes:

| Mode | Behavior |
| --- | --- |
| `"none"` | Preserve the aligned group order. |
| `"rank_epoch_random"` | Deterministically decide once per iterator whether each rank flips. |
| `"rank_window_random"` | Deterministically decide per grouping window and rank. |
| `"rank_window_balanced"` | Flip roughly half the ranks per grouping window. |

Leave this at `"none"` unless you are deliberately testing group-order effects.
Legacy aliases `A`, `B`, `C`, and `D` map to the four modes above.

## ODB Buffer-Fill Warm-Up

`no_warmup` controls ODB's internal buffer-fill ramp in the collate process. It
is not optimizer warmup, LR warmup, or a model-quality feature.

With the default `no_warmup=False`, ODB can begin emitting before the full
buffer threshold is reached and then ramps toward the full grouping window. With
`no_warmup=True`, ODB skips that ramp and waits for the full buffer threshold
immediately.

Most users should leave the default unchanged unless a framework example or
maintainer guide sets this for a specific launch path.

## PyTorch Multiprocessing Sharing Strategy

ODB uses DataLoader workers plus an additional collate process. On some systems,
many workers and large tensor batches can exhaust file descriptors and produce
errors such as `Too many open files`.

For long-running multi-worker training, examples use:

```python
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
```

Set this near the beginning of the training entrypoint, before DataLoader
workers are created. This is a PyTorch multiprocessing runtime setting, not an
ODB batching algorithm parameter.

## Framework Ownership

- PyTorch loop users usually set these directly in `ODBDataLoader(...)` or
  `odb.apply(...)`.
- Hugging Face Trainer and LLaMA-Factory users usually pass them through the
  high-level `enable_odb(...)` hook or the example project's config surface.
- Accelerate and Lightning users usually set DataLoader-side knobs on the
  ODB-enabled DataLoader and let the adapter consume metadata.

Framework-specific guides should not redefine these semantics; they should only
show where the settings are passed.
