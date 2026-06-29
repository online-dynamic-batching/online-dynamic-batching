# Adapter Principles

Every framework adapter should preserve the same ODB runtime contract. This
page is for adapter authors and framework integrators. If you only want to run
an example, start from the framework-specific guide or example repository in
[Integration Guides](README.md).

ODB is not a tokenizer, image processor, chat-template engine, or multimodal
collator. Framework adapters should attach after the project has produced one
model-ready tensor sample at a time.

```text
raw record
  -> project tokenizer / processor / template / multimodal adapter
  -> ODB-ready single-sample tensor dict
  -> ODB dynamic grouping
  -> trainer or training loop
```

Do not make Accelerate, Lightning, or Hugging Face Trainer adapters depend on
LLaMA-Factory to obtain this tensor sample. LLaMA-Factory support lives in its
own adapter and example project.

The collator that receives ODB groups should pad or stack already processed
tensor samples. It should not be the first place where raw multimodal records
are tokenized or expanded into model tokens; otherwise ODB cannot observe the
real per-sample length before grouping.

## Trainer To ODB

Resolve framework configuration into an `ODBConfig` or equivalent `odb.apply`
arguments:

```python
odb.ODBConfig(
    token_budget=...,
    loss_scaling="exact",
    join=True,
    buffer_size=...,
)
```

Use `loss_scaling="exact"` for distributed trainer integrations unless a guide
has a specific reason to use another mode. `join=True` is the default
DataLoader-side drain setting.

Then choose one DataLoader path:

- use `ODBDataLoader(...)` when the framework lets you construct the loader;
- use `odb.apply(existing_loader, ...)` when the framework already constructed
  the loader.

In both cases, the DataLoader should emit one fully processed sample at a time
with `batch_size=1` and enough worker prefetching for an online grouping window.

## ODB To Trainer

Consume:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
```

Use only:

- `info.all_samples_this_step`
- `info.loss_scale`

Framework code should not depend on legacy flat transport keys.

## Training Semantics

Adapters must define:

- where ODB metadata is removed before model forward;
- where loss is multiplied by `info.loss_scale`;
- how sample progress accumulates across gradient accumulation;
- how sample-budget stopping interacts with optimizer-step caps;
- how scheduler progress is computed;
- how the runtime join setting is paired with framework-level DDP Join /
  uneven-input support.

These semantics are part of the adapter contract, not optional logging. A path
that applies ODB to a DataLoader but never consumes `ODBStepInfo` is incomplete
for distributed exact-loss training.

## Choose One Trainer Hook

Trainer-facing adapters should expose whichever of these hook points the
framework supports. Users normally choose one:

- **High-level enable hook**: a helper such as `enable_odb(...)` applies ODB to
  an ODB-ready DataLoader and wires the trainer accounting.
- **Native Trainer hook**: a subclass or mixin such as `ODBTrainerMixin` or
  `ODBTrainer` consumes ODB metadata in the Trainer method itself.
- **Existing Trainer hook**: a helper such as `configure_trainer(...)`,
  `configure_accelerator(...)`, or `configure_lightning_module(...)` configures
  an already-created framework object.
- **Manual contract**: framework code directly calls
  `odb.pop_step_info(...)`, applies `info.loss_scale`, and accounts
  `info.all_samples_this_step`.

Join is not a trainer hook. It is an ODB runtime setting for DataLoader/collate
termination and must be paired with the framework's distributed uneven-input
guard when model collectives can run unevenly.

For API naming and compatibility rules, see
[API Design And Adapter Notes](../API_DESIGN_NOTES.md). For shared runtime
settings such as `token_budget`, `join`, `loss_scaling`, `buffer_size`,
`group_order_flip`, buffer-fill warm-up, and multiprocessing sharing strategy,
see [Runtime Settings](../runtime-settings.md).
