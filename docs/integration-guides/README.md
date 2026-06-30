# Integration Guides

Choose the path that matches your training stack. These paths are
alternatives; you do not need to implement every adapter.

ODB starts after your framework or model-specific processor has produced
model-ready single-sample tensors. It does not replace tokenizer, processor,
multimodal template, truncation, label masking, or collator semantics across
frameworks.

```text
raw records
  -> tokenizer / processor / template / multimodal adapter
  -> ODB-ready single-sample tensor dict
  -> ODB dynamic grouping
  -> trainer or training loop
```

## Pick One Path

| Training stack | Guide | Runnable example | Install extra | Recommended entry |
| --- | --- | --- | --- | --- |
| Plain PyTorch loop | [PyTorch Loop](pytorch-loop.md) | [Synthetic benchmark](../../examples/synthetic_benchmark.py) | none | `odb.ODBDataLoader(...)` or `odb.apply(...)` |
| Hugging Face Trainer | [HF Trainer](hf-trainer.md) | [odb-example-hf-trainer](https://github.com/online-dynamic-batching/odb-example-hf-trainer) | `online-dynamic-batching[hf]` | `enable_odb(...)` |
| LLaMA-Factory | [LLaMA-Factory](llamafactory.md) | [odb-example-llamafactory](https://github.com/online-dynamic-batching/odb-example-llamafactory) | `online-dynamic-batching[hf]` | `enable_odb(...)` for ODB-ready trainer hooks |
| Accelerate | [Accelerate](accelerate.md) | [odb-example-accelerate](https://github.com/online-dynamic-batching/odb-example-accelerate) | `online-dynamic-batching[accelerate]` | `configure_accelerator(...)` |
| Lightning | [Lightning](lightning.md) | [odb-example-lightning](https://github.com/online-dynamic-batching/odb-example-lightning) | `online-dynamic-batching[lightning]` | `configure_lightning_module(...)` |

The shared MM-Mix dataset builder used by the framework examples lives in
[build-mm-mix-dataset](https://github.com/online-dynamic-batching/build-mm-mix-dataset).

## Core Runtime Contract

Every integration consumes the same ODB metadata before model forward:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss
loss = loss * info.loss_scale
emitted_samples += info.all_samples_this_step
```

High-level adapters hide some of this boilerplate, but the responsibilities are
the same:

- emit one fully processed sample at a time into ODB;
- use worker prefetching so ODB has an online grouping window;
- remove ODB transport metadata before `model(**batch)`;
- apply `info.loss_scale` when exact loss scaling is enabled;
- stop or schedule by `info.all_samples_this_step` when using sample progress.

## Shared Settings

Runtime knobs such as `join`, `loss_scaling`, `token_budget`, buffer/prefetch,
`group_order_flip`, ODB buffer-fill warm-up, and PyTorch multiprocessing
sharing strategy are shared across frameworks. See
[Runtime Settings](../runtime-settings.md).

For adapter design rules and trainer-facing responsibilities, see
[Adapter Principles](adapter-principles.md).

For custom framework forks or large training codebases, see
[Agent-Assisted Integration](../agent-assisted-integration.md).

For first-time setup and a small synthetic demo, start with the
[Quickstart](../quickstart.md).
