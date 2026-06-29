# PyTorch Lightning

Use `ODBDataLoader` from `LightningDataModule.train_dataloader()` when possible
and keep the returned loader or its `odb_handle`:

```python
def train_dataloader(self):
    return odb.ODBDataLoader(
        self.train_dataset,
        token_budget=8192,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=self.collate_fn,
        loss_scaling="exact",
        join=True,
    )
```

Prefer the package helper when the module's `training_step` returns a tensor
loss or `{"loss": ...}`:

```python
from odb.integrations.lightning import ODBLightningCallback, configure_lightning_module

bridge = configure_lightning_module(
    module,
    handle=train_loader.odb_handle,
    sample_budget=len(train_dataset) * max_epochs,
)
trainer = L.Trainer(callbacks=[ODBLightningCallback(bridge.sample_budget)])
trainer.fit(module, train_dataloaders=train_loader)
```

If the project needs a custom `training_step`, use the mixin or consume metadata
explicitly before forward:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
outputs = self.model(**batch)
loss = outputs.loss * info.loss_scale
self.odb_emitted_samples += info.all_samples_this_step
return loss
```

With default `join=True`, configure DDP Join or an equivalent Lightning strategy
guard when distributed ranks may exhaust unevenly.
