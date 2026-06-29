# PyTorch Lightning Integration

Install with the optional dependency group:

```bash
pip install "online-dynamic-batching[lightning]"
```

## Run A Full Example

For a complete public multimodal workflow with data preparation, ODB training,
fixed-batch baseline training, validation loss, and MMMU-MC evaluation, use
[`odb-example-lightning`](https://github.com/online-dynamic-batching/odb-example-lightning).

This guide explains the package API and the required Lightning data-pipeline
contract. The example repository shows the same contract in a runnable
Qwen3-VL MM-Mix-style project.

## Data Pipeline Contract

Keep your LightningDataModule, tokenizer/processor, template, and collator
semantics unchanged. ODB starts after your Dataset or single-sample processor
path has produced model-ready tensor samples.

For raw multimodal records, the model-specific processor should run before ODB
grouping. The collator should only pad/stack the dynamic group that ODB emits.

Lightning integration then has two separate decisions:

1. Where to enable the ODB DataLoader.
2. Which hook consumes ODB metadata and scales the loss.

## DataLoader

Choose one DataLoader path. The most direct path is to construct the training
loader explicitly and pass that same object to Lightning:

```python
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
```

If your `LightningDataModule` constructs train dataloaders, have
`train_dataloader()` return an ODB-enabled loader and keep a reference to the
same loader or its `odb_handle` for the hook below. If a plain PyTorch
`DataLoader` is constructed elsewhere, call `odb.apply(...)` on that loader
before training.

## Recommended Hook: `configure_lightning_module(...)`

Use the module wrapper when your `training_step` returns a tensor loss or a
dictionary containing `{"loss": ...}`:

```python
from odb.integrations.lightning import ODBLightningCallback, configure_lightning_module

module = MyLightningModule(...)
bridge = configure_lightning_module(
    module,
    handle=train_loader.odb_handle,
    sample_budget=len(train_dataset) * max_epochs,
)
trainer = L.Trainer(callbacks=[ODBLightningCallback(bridge.sample_budget)])
trainer.fit(module, train_dataloaders=train_loader)
```

The wrapper removes ODB transport metadata before the module sees the batch and
scales tensor or `{"loss": ...}` returns.

Here, `train_loader` is the `odb.ODBDataLoader` instance constructed in the
DataLoader step above. If you use a `LightningDataModule`, use the exact loader
returned by `train_dataloader()` or the `odb_handle` stored from that loader.

## Alternative Hook: Explicit `training_step`

For more complex modules, use `ODBLightningMixin` and consume step info before
calling the model forward:

```python
from odb.integrations.lightning import ODBLightningMixin

class MyModule(ODBLightningMixin, L.LightningModule):
    ...

def training_step(self, batch, batch_idx):
    info = self.consume_odb_batch(batch)
    output = self.model(**batch)
    return self.scale_odb_loss(output.loss, info=info)
```

## Runtime Settings

Runtime behavior such as `join`, exact loss scaling, sample progress,
`token_budget`, buffer/prefetch, `group_order_flip`, ODB buffer-fill warm-up,
and PyTorch multiprocessing sharing strategy is shared across frameworks. See
[Runtime Settings](../runtime-settings.md).

For distributed runs, configure DDP Join or an equivalent Lightning strategy
guard when ranks may exhaust at different optimizer steps.

## Integration Contract

A Lightning integration should:

- stop when emitted samples reach `sample_budget`;
- keep LR scheduling semantics explicit for nonstandard schedulers;
- avoid passing ODB transport metadata into the model;
- keep the runtime join setting separate from the chosen loss hook.
