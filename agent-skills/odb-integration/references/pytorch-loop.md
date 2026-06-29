# PyTorch Loop

Use `odb.ODBDataLoader` when the project owns DataLoader construction:

```python
train_loader = odb.ODBDataLoader(
    train_dataset,
    token_budget=8192,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    collate_fn=data_collator,
    loss_scaling="exact",
    join=True,
)
```

Patch the training step:

```python
for batch in train_loader:
    info = odb.pop_step_info(batch, loss_scaling="exact")
    loss = model(**batch).loss * info.loss_scale
    loss.backward()
    emitted_samples += info.all_samples_this_step
```

If the model does not return `.loss`, multiply the tensor loss computed by the
project. Stop by `sample_budget` when required.

Start with conservative worker/prefetch settings, verify correctness, then tune
`token_budget`, `buffer_size`, `num_workers`, and `prefetch_factor` for the
target machine.
