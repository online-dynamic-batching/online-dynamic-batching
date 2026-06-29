# PyTorch Loop Integration

Use this path when you own the PyTorch training loop and can decide where ODB
is attached.

## DataLoader

Choose one DataLoader path.

If you own DataLoader construction, use `ODBDataLoader(...)`:

```python
import odb

train_loader = odb.ODBDataLoader(
    train_dataset,
    token_budget=8192,
    batch_size=1,          # ODB forms the real batch dynamically
    shuffle=True,
    num_workers=4,         # worker prefetching feeds the online buffer
    prefetch_factor=64,
    collate_fn=data_collator,
    loss_scaling="exact",
    join=True,
)
```

If a regular DataLoader already exists, patch that loader instead:

```python
handle = odb.apply(
    train_loader,
    token_budget=8192,
    loss_scaling="exact",
    join=True,
)
```

## Training Step

Call `odb.pop_step_info(...)` before `model(**batch)` so ODB transport metadata
does not reach model forward.

```python
emitted_samples = 0

for batch in train_loader:
    info = odb.pop_step_info(batch, loss_scaling="exact")

    loss = model(**batch).loss
    loss = loss * info.loss_scale
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    emitted_samples += info.all_samples_this_step
```

If you used `odb.apply(...)`, you can read the resolved mode from the handle:

```python
info = odb.pop_step_info(batch, loss_scaling=handle.config.loss_scaling)
```

## ODB-Ready Batch Contract

- The DataLoader emits one fully processed sample at a time: `batch_size=1`.
- Worker prefetching is enabled: `num_workers > 0`.
- Each single sample exposes `input_ids`, or the project provides a compatible
  length hook before grouping.
- The original `collate_fn` can still batch the dynamic group that ODB emits.
- The training loop removes ODB metadata with `odb.pop_step_info(...)` before
  calling the model.

## Stopping And Scheduling

Use emitted samples for sample-budget stopping:

```python
if emitted_samples >= sample_budget:
    break
```

If the LR schedule should follow samples rather than optimizer steps, advance or
recompute the scheduler from emitted-sample progress.

## Runtime Settings

For shared runtime behavior such as `join`, exact loss scaling, `token_budget`,
buffer/prefetch, `group_order_flip`, ODB buffer-fill warm-up, and PyTorch
multiprocessing sharing strategy, see
[Runtime Settings](../runtime-settings.md).

## Verify First

If you only want to confirm that ODB is installed and can run a toy training
loop, start with the [Quickstart](../quickstart.md). This guide is for wiring
ODB into your own PyTorch loop.
