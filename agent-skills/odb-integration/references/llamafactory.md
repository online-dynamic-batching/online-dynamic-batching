# LLaMA-Factory

Find the code that builds the training DataLoader and Trainer. Keep
LLaMA-Factory's dataset, template, tokenizer, multimodal plugin, and processor
semantics unchanged. ODB must start after that stack can emit model-ready
single-sample tensor dicts containing `input_ids`.

Use these paths in order:

1. Official hook: call `enable_odb(...)` if the fork already exposes a hook
   after DataLoader and Trainer construction.
2. Agent-assisted patch: add the hook at that construction point and keep the
   existing lazy tensor-sample path.
3. Manual integration: fall back to `odb.apply(...)` plus
   `configure_trainer(...)` only when writing framework glue yourself.

Expose config fields:

```yaml
use_odb: true
odb_token_budget: 8192
odb_loss_scaling: exact
odb_join: true  # default
per_device_train_batch_size: 1
dataloader_num_workers: 4
dataloader_prefetch_factor: 2
```

After DataLoader and Trainer construction:

```python
from odb.integrations.llamafactory import enable_odb

enable_odb(
    trainer,
    train_dataloader=train_dataloader,
    training_args=training_args,
    train_dataset=train_dataset,
    token_budget=args.odb_token_budget,
    loss_scaling="exact",
    join=getattr(args, "odb_join", True),
    scheduler_progress="samples",
    max_steps_policy="overwrite",
)
```

The adapter infers `sample_budget`, maps an active `training_args.max_steps` to
`max_optimizer_steps`, validates `per_device_train_batch_size=1`, validates
DataLoader `batch_size=1` and `num_workers>0`, checks that a sampled dataset
item contains `input_ids`, then delegates to the HuggingFace Trainer adapter.

If the sampled item does not contain `input_ids`, do not make ODB group raw
records by an estimated length field. Move only single-sample tensorization
before grouping so ODB observes the true post-pipeline length. Do not duplicate
or replace LLaMA-Factory templates or multimodal plugins.

With default `odb_join=true`, verify the training stack also uses DDP Join or
Accelerate uneven-input handling.

Start with conservative DataLoader worker/prefetch settings in a new
environment. Tune prefetch and ODB buffer settings only after the ODB-ready
single-sample path, exact loss scaling, sample progress, and join behavior are
verified.
