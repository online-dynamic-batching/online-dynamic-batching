# API Design And Adapter Notes

This page is for package maintainers and framework integrators. User-facing
setup lives in [Quickstart](quickstart.md) and
[Integration Guides](integration-guides/README.md). Runtime behavior shared by
all integrations lives in [Runtime Settings](runtime-settings.md).

## Compatibility Boundary

ODB keeps historical aliases where they are already public:

```python
odb.apply(dataloader, max_input_length=16384, join_mode=True)
odb.apply(dataloader, token_budget=16384, join=True)
```

Public docs should prefer the newer names:

- `token_budget` instead of `max_input_length`;
- `join` instead of `join_mode`;
- `loss_scaling="exact" | "approx" | "none"` instead of boolean-only modes.

Aliases should remain explicit compatibility layers. Do not silently change
their semantics.

## Core API Shape

```python
from odb import ODBConfig, ODBDataLoader, apply

loader = ODBDataLoader(
    dataset,
    token_budget=16384,
    batch_size=1,
    num_workers=4,
    prefetch_factor=2,
    collate_fn=collate_fn,
    loss_scaling="exact",
    join=True,
)

handle = apply(existing_loader, token_budget=16384, loss_scaling="exact", join=True)
```

Useful handle fields:

- `handle.config`
- `handle.step_info_key`
- `handle.token_budget`
- `handle.join`

Keep `ODBHandle` small. Framework-specific state should live in that
framework's adapter bridge rather than in the core DataLoader handle.

## Trainer Responsibilities

ODB is not only a sampler. A trainer or loop integration must define:

- how ODB metadata is removed before model forward;
- how `info.loss_scale` reaches the backward path;
- how sample-budget stopping uses `info.all_samples_this_step`;
- how optimizer-step and scheduler progress are defined;
- how DataLoader-side join is paired with framework-level uneven-input handling
  in distributed training.

These responsibilities should be visible in adapter APIs. A convenient
`enable_odb(...)` entry point should still make clear which DataLoader and
trainer hooks it configures.

## Upward Runtime Interface

For normal training, prefer the structured `ODBStepInfo` object:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss * info.loss_scale
emitted_samples += info.all_samples_this_step
```

`all_samples_this_step` is the all-rank sum of samples emitted for the current
yielded batch or micro-step. `loss_scale` is the current-rank multiplier to
apply to the model loss before backward.

Flat transport keys such as `total_batch_size`, `local_batch_size`,
`odb_local_tokens`, and `odb_total_tokens` are implementation details consumed
by `odb.pop_step_info(batch)`. New framework integrations should not depend on
those keys directly.

## Integration Levels

ODB exposes two levels of framework API:

- high-level `enable_odb(...)` helpers for users who already have an
  ODB-ready single-sample tensor pipeline;
- lower-level `configure_*` helpers for framework maintainers or downstream
  integrations that already control DataLoader and trainer construction.

Both levels should preserve the same runtime contract: apply ODB after
model-specific preprocessing, remove transport metadata before model forward,
scale the loss when configured, and account progress with
`info.all_samples_this_step`.

## Adapter Naming

High-level adapters should read as "enable ODB for this framework":

```python
odb.integrations.hf.enable_odb(...)
odb.integrations.llamafactory.enable_odb(...)
```

Lower-level adapters should be named after the framework object they configure:

```python
odb.integrations.hf.configure_trainer(...)
odb.integrations.accelerate.configure_accelerator(...)
odb.integrations.lightning.configure_lightning_module(...)
```

Avoid adding a high-level adapter that only changes trainer accounting while
leaving raw multimodal processing inside a collator after ODB grouping. In that
case, expose the lower-level hook and document the missing data-pipeline
contract instead.

## Join Policy

ODB defaults to `join=True` at the DataLoader/collate layer. That keeps the ODB
grouping protocol alive while uneven ranks drain.

Distributed model collectives are a separate layer. Framework integrations
should pair ODB join with the appropriate framework mechanism when ranks can
exhaust at different optimizer steps:

- PyTorch DDP: DDP Join or an equivalent guard.
- Accelerate: `accelerator.join_uneven_inputs(...)` when available.
- Lightning: a strategy or plugin that supports uneven inputs.

See [Runtime Settings](runtime-settings.md) for user-facing runtime guidance.
