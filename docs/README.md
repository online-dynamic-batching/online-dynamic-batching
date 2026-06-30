# Online Dynamic Batching Documentation

This directory is organized by reader task. Start with the quick demo, then
choose the framework guide that matches your training stack.

## Start Here

| Need | Read |
| --- | --- |
| Install ODB and run a tiny public demo | [Quickstart](quickstart.md) |
| Choose a framework integration path | [Integration Guides](integration-guides/README.md) |
| Understand shared knobs such as `join`, `loss_scaling`, and `token_budget` | [Runtime Settings](runtime-settings.md) |
| Compare benchmark reporting rules | [Benchmarks](benchmarks.md) |
| See what has been validated before release | [Validation](validation.md) |

## Integration Paths

ODB starts after your model-specific input pipeline has produced model-ready
single-sample tensors. Pick one path; these are alternatives, not a checklist.

| Training stack | Guide | Runnable example |
| --- | --- | --- |
| Plain PyTorch loop | [PyTorch Loop](integration-guides/pytorch-loop.md) | [Synthetic benchmark](../examples/synthetic_benchmark.py) |
| Hugging Face Trainer | [HF Trainer](integration-guides/hf-trainer.md) | [odb-example-hf-trainer](https://github.com/online-dynamic-batching/odb-example-hf-trainer) |
| LLaMA-Factory | [LLaMA-Factory](integration-guides/llamafactory.md) | [odb-example-llamafactory](https://github.com/online-dynamic-batching/odb-example-llamafactory) |
| Accelerate | [Accelerate](integration-guides/accelerate.md) | [odb-example-accelerate](https://github.com/online-dynamic-batching/odb-example-accelerate) |
| Lightning | [Lightning](integration-guides/lightning.md) | [odb-example-lightning](https://github.com/online-dynamic-batching/odb-example-lightning) |

The shared MM-Mix dataset builder used by the framework examples lives in
[build-mm-mix-dataset](https://github.com/online-dynamic-batching/build-mm-mix-dataset).

## Concepts

- [Integrations](integrations.md): the processor / ODB / trainer boundary.
- [Grouping Algorithm](GROUPING_ALGORITHM.md): online grouping and DDP
  alignment at a high level.
- [Runtime Settings](runtime-settings.md): shared runtime knobs and environment
  settings.
- [Adapter Principles](integration-guides/adapter-principles.md): design rules
  for framework adapters.

## Examples

- [Synthetic benchmark](../examples/synthetic_benchmark.py): small CPU/GPU
  functional demo with no private data.
- [Single-GPU notebook](../examples/notebooks/odb_single_gpu_demo.ipynb):
  preview on GitHub, then run locally in Jupyter.
- [Framework example projects](integration-guides/README.md): public MM-Mix
  style examples for HF Trainer, LLaMA-Factory, Accelerate, and Lightning.

## Maintainers

- [API Design And Adapter Notes](API_DESIGN_NOTES.md): public API compatibility
  and adapter conventions.
- [Agent-Assisted Integration](agent-assisted-integration.md): optional
  coding-agent workflow for patching a training stack.

For the package overview, see the repository [README](../README.md).
