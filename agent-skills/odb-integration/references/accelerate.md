# Accelerate

Use `ODBDataLoader` before the training loop and keep its `odb_handle`:

```python
import odb
from odb.integrations.accelerate import configure_accelerator

train_loader = odb.ODBDataLoader(
    train_dataset,
    token_budget=8192,
    batch_size=1,
    num_workers=4,
    prefetch_factor=2,
    collate_fn=data_collator,
    loss_scaling="exact",
    join=True,
)
handle = train_loader.odb_handle
```

Prepare model and optimizer, then let the ODB bridge consume metadata, scale
loss, and track emitted-sample progress:

```python
model, optimizer = accelerator.prepare(model, optimizer)
bridge = configure_accelerator(
    accelerator,
    train_loader,
    handle=handle,
    sample_budget=len(train_dataset) * num_epochs,
    loss_scaling="exact",
)

with bridge.join_uneven_inputs([model]):
    for batch in train_loader:
        info = bridge.consume_batch(batch)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            bridge.backward(outputs.loss, info=info)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        bridge.mark_optimizer_step()
        if bridge.should_stop:
            break
```

If the project must use a fully manual loop, consume metadata before forward:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss * info.loss_scale
accelerator.backward(loss)
emitted_samples += info.all_samples_this_step
```

With default `join=True`, wrap training with
`accelerator.join_uneven_inputs(...)` when available; the bridge exposes this
as `bridge.join_uneven_inputs(...)`.
