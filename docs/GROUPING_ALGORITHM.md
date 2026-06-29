# ODB Grouping Algorithm

ODB forms dynamic batches after the input pipeline has produced fully
processed single samples. At that point the true runtime length is observable,
including chat templates, truncation, augmentation, and multimodal token
expansion.

The implementation lives in `src/odb/grouping.py`.

## What The Algorithm Optimizes

ODB targets padded-token work per model step. For a sample with observed length
`L`, the local target group size is roughly:

```text
group_size = max(floor(token_budget / L), 1)
```

Short samples therefore receive larger groups, and long samples receive smaller
groups. Because samples are grouped by similar observed length, this keeps
`max_length_in_group * group_size` near the configured `token_budget`. The
budget is a batching reference, not the model truncation length.

## Inputs

- A local buffer of fully processed sample dictionaries from DataLoader workers.
- `token_budget`: the target padded-token work per dynamic group.
- Optional DDP process group metadata and output-slot budget from the
  DataLoader iterator.
- Optional loss-scaling mode: `none`, `approx`, or `exact`.
- Join/finish flags used to keep ranks aligned near the end of an epoch.
- Optional `group_order_flip` mode for rank-wise group-order audits.

Each sample should expose an `input_ids`-like length before grouping. Framework
adapters validate this contract where possible.

## Rank-Local Grouping

Each rank groups only its own local samples:

1. Flatten the local worker buffer into a single candidate window.
2. Compute observed runtime length for each sample.
3. Sort candidates by observed length.
4. Traverse from longer to shorter candidates and compute the target group size
   from the current observed length and `token_budget`.
5. Form groups from adjacent length-sorted samples.
6. Keep overflow samples for the next grouping round.

This local step is intentionally simple. It does not require a precomputed
length cache, and it does not exchange sample payloads across ranks.

## Cross-Rank Alignment

DDP requires every rank to execute the same number of model steps. Because each
rank sees different local samples, the number of local groups can differ. ODB
therefore exchanges scheduling metadata:

```text
rank0_group_count = 3
rank1_group_count = 2
rank2_group_count = 1

all_gather -> [3, 2, 1]
aligned_steps = max([3, 2, 1]) = 3
```

In practice, ODB exchanges more than a single count. Each rank gathers:

- available output slots in the DataLoader iterator;
- local group count, or `-1` if that rank has finished;
- local group sizes;
- optional per-group token counts for loss scaling.

ODB then computes a shared target group count. The target is bounded by the
maximum active group count, the smallest positive active output-slot budget, and
the smallest positive active sample count. This keeps all ranks within their
available iterator capacity while still emitting as much work as the current
window permits.

After the target is chosen, every rank applies the same deterministic rule:

- if a rank has too few groups, split larger groups when possible;
- if a rank has too many groups, keep the largest target groups and recycle the
  rest into the next grouping round;
- if a rank still has fewer emitted groups than the aligned step count, the
  DataLoader side emits IDLE sentinels for the missing positions.

The trainer skips IDLE batches, while distributed collectives remain aligned.

## Join Mode

`join=True` is the default DataLoader-side drain mode. When one rank exhausts
its local input early, that rank keeps participating in ODB's lightweight
metadata protocol until all ranks finish. Active ranks can therefore continue
emitting aligned groups instead of ending the epoch at the shortest rank.

This is separate from model forward/backward collectives. Distributed training
should pair ODB join with the framework's uneven-input guard when model
collectives can see uneven rank progress, such as PyTorch DDP Join or
Accelerate's `join_uneven_inputs(...)`.

`join=False` remains available for deployments that deliberately use the
shortest-rank closure behavior.

## Loss-Scaling Metadata

ODB attaches trainer-facing metadata to emitted batches when requested:

- all-rank emitted sample counts for the current step;
- local sample counts for the current rank;
- optional local/global token counts;
- a loss multiplier for DDP gradient averaging when ranks process different
  local sample or token counts.

Training code should call `odb.pop_step_info(batch, ...)` before model forward.
That removes ODB transport fields from the batch and returns:

```python
ODBStepInfo(
    all_samples_this_step=...,
    loss_scale=...,
)
```

Use `info.all_samples_this_step` for sample-budget progress and multiply the
local loss by `info.loss_scale` when exact or approximate loss scaling is
enabled.

## Group Order

By default, aligned groups are emitted in deterministic order. The optional
`group_order_flip` runtime setting can reverse the post-alignment group order on
selected ranks for audits or experiments. It does not change which samples are
selected from the grouping window; it only changes the order in which aligned
groups reach the trainer.

## Communication Cost

The distributed path uses a small CPU metadata exchange for group-count,
group-size, output-budget, and finish-state alignment. Exact token loss scaling
may add one more metadata exchange for post-alignment token counts. ODB does
not exchange model tensors or sample payloads during grouping.
