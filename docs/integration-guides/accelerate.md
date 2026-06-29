# Accelerate Integration

Install with the optional dependency group:

```bash
pip install "online-dynamic-batching[accelerate]"
```

## Run A Full Example

For a complete public multimodal workflow with data preparation, ODB training,
fixed-batch baseline training, validation loss, and MMMU-MC evaluation, use
[`odb-example-accelerate`](https://github.com/online-dynamic-batching/odb-example-accelerate).

This guide explains the package API and the required Accelerate data-pipeline
contract. The example repository shows the same contract in a runnable
Qwen3-VL MM-Mix-style project.

## Data Pipeline Contract

Accelerate keeps the training loop in user code. ODB keeps the same boundary:
your Dataset or single-sample processor path should already emit model-ready
tensor samples before ODB grouping.

For raw multimodal records, run tokenizer/processor, template expansion,
truncation, visual-token expansion, and label masking before ODB grouping. The
collator should only pad/stack the dynamic group that ODB emits.

Accelerate integration then has three separate decisions:

1. Where to enable the ODB DataLoader.
2. Where to consume ODB metadata and scale the loss.
3. Whether the ODB DataLoader is passed through `accelerator.prepare(...)`.

## DataLoader

Choose one DataLoader path before the training loop. The recommended path is
to construct an ODB loader, keep its `odb_handle`, and pass the same loader to
the custom loop:

```python
import odb

train_loader = odb.ODBDataLoader(
    train_dataset,
    token_budget=8192,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    prefetch_factor=256,
    collate_fn=collate_fn,
    loss_scaling="exact",
)
handle = train_loader.odb_handle
```

If a plain PyTorch `DataLoader` is constructed elsewhere, call `odb.apply(...)`
on that loader before training and keep the returned handle:

```python
handle = odb.apply(train_loader, token_budget=8192, loss_scaling="exact", join=True)
```

For the ODB path, prepare the model and optimizer with Accelerate and keep the
ODB loader object available to the loop:

```python
model, optimizer = accelerator.prepare(model, optimizer)
```

If your code prepares the DataLoader as well, store the ODB handle before
preparation and iterate over the prepared loader returned by your code or by
`configure_accelerator(..., prepare_dataloader=True)`.

## Recommended Loop: `configure_accelerator(...)`

Use `configure_accelerator(...)` to consume ODB transport metadata, apply exact
loss scaling, and track emitted-sample progress:

```python
from accelerate import Accelerator
from accelerate.utils import send_to_device
from odb.integrations.accelerate import configure_accelerator

accelerator = Accelerator()

bridge = configure_accelerator(
    accelerator,
    train_loader,
    handle=handle,
    sample_budget=len(train_dataset) * num_epochs,
    loss_scaling="exact",
)

with bridge.join_uneven_inputs([model]):
    for batch in train_loader:
        batch = send_to_device(batch, accelerator.device)
        info = bridge.consume_batch(batch)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        bridge.backward(outputs.loss, info=info)
        optimizer.step()

        bridge.mark_optimizer_step()
        if bridge.should_stop:
            break
```

`bridge.consume_batch(batch)` removes ODB transport metadata before
`model(**batch)`. `bridge.backward(...)` applies `info.loss_scale` before
delegating to `accelerator.backward(...)`.

## Runtime Settings

Runtime behavior such as `join`, exact loss scaling, sample progress,
`token_budget`, buffer/prefetch, `group_order_flip`, ODB buffer-fill warm-up,
and PyTorch multiprocessing sharing strategy is shared across frameworks. See
[Runtime Settings](../runtime-settings.md).

For distributed runs, wrap uneven-rank model collectives with Accelerate's
uneven-input support when available:

```python
with bridge.join_uneven_inputs([model]):
    train_loop()
```

ODB join mode handles DataLoader/collate synchronization. Accelerate/DDP must
still protect model collectives.

## Integration Contract

An Accelerate integration should:

- keep model-specific preprocessing before ODB grouping;
- keep the ODB loader and its `odb_handle` paired together;
- move ODB batches to `accelerator.device` when the loader is not prepared by
  Accelerate;
- stop when emitted samples reach `sample_budget`;
- avoid passing ODB transport metadata into the model;
- keep LR scheduling semantics explicit for custom loops.
