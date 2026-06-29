# Integration Model

This page explains the integration boundary. To choose a concrete framework
path, start with [Integration Guides](integration-guides/README.md).

ODB has one core runtime contract:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss * info.loss_scale
emitted_samples += info.all_samples_this_step
```

The contract begins after your framework has converted raw records into
model-ready single-sample tensors. ODB does not attempt to make different
multimodal encoding pipelines equivalent.

Think of the integration in three layers:

```text
raw records
  -> tokenizer / processor / template / multimodal adapter
  -> ODB-ready single-sample tensor dict
  -> ODB DataLoader/grouping
  -> trainer or training loop
```

The processor side is model-specific: it owns templates, tokenization,
multimodal processors, truncation, visual-token expansion, and label masking.
ODB starts after that layer emits an ODB-ready single-sample tensor dict. The
trainer or loop adapter then consumes ODB metadata for loss scaling and
sample-progress accounting.

Framework-specific guides:

- [PyTorch Loop](integration-guides/pytorch-loop.md)
- [Hugging Face Trainer](integration-guides/hf-trainer.md)
- [LLaMA-Factory](integration-guides/llamafactory.md)
- [Accelerate](integration-guides/accelerate.md)
- [PyTorch Lightning](integration-guides/lightning.md)

Code-level adapters are available under `odb.integrations.hf`,
`odb.integrations.llamafactory`, `odb.integrations.accelerate`, and
`odb.integrations.lightning`.

`odb.integrations.hf.enable_odb(...)` is the high-level hook for ODB-ready
Hugging Face Trainer pipelines. `odb.integrations.llamafactory.enable_odb(...)`
adds LLaMA-Factory-specific validation and argument resolution on top of the HF
adapter. Accelerate and Lightning use loop/trainer helpers such as
`configure_accelerator(...)` and `configure_lightning_module(...)`. None of
these adapters claim raw multimodal pipeline equivalence.

For runtime knobs shared across these integrations, including `join`,
`loss_scaling`, `token_budget`, buffer/prefetch, `group_order_flip`, ODB
buffer-fill warm-up, and PyTorch multiprocessing sharing strategy, see
[Runtime Settings](runtime-settings.md).
