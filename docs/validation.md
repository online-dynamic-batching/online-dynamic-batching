# Validation

This page describes the validation layers used for ODB releases and public
example projects. Validation confirms that an integration path can train and
evaluate correctly; it is separate from benchmark reporting.

## Validation Layers

| Layer | Purpose | Typical command or evidence |
| --- | --- | --- |
| Package tests | Core grouping, config parsing, metadata, loss scaling, and adapter helpers | `pytest` |
| Build tests | Wheel/source distribution can build and import | `python -m build`, `twine check` |
| Adapter contract tests | Framework adapters consume ODB metadata and preserve sample-budget behavior | unit tests under `tests/` |
| Toy workflow tests | The package can run a small public synthetic demo | `python examples/synthetic_benchmark.py` |
| Example workflow tests | Public example repositories can train and evaluate from their README workflow | `./run.sh all-odb` or framework-specific train/eval steps |
| Benchmark reporting checks | Throughput numbers use literal emitted samples over wall-clock time and report quality alongside speed | see [Benchmarks](benchmarks.md) |

## What Adapter Validation Checks

Every framework path should verify the same runtime contract:

- the DataLoader emits one fully processed sample at a time into ODB;
- ODB transport metadata is removed before `model(**batch)`;
- exact loss scaling applies `info.loss_scale` in the framework's backward path;
- emitted-sample progress uses `info.all_samples_this_step`;
- sample-budget stopping reaches the intended budget;
- distributed runs pair ODB's DataLoader-side `join=True` with the
  framework's uneven-input guard when needed.

## Framework Scope

Hugging Face Trainer, Accelerate, and Lightning integrations are
framework-native. Users bring the Dataset, tokenizer/processor, template,
collator, and model-specific multimodal encoding. These adapters do not claim
that different framework-native multimodal pipelines produce identical
`input_ids`, labels, or vision tensors.

The LLaMA-Factory example is the closest public path to the paper's
implementation style: it uses LLaMA-Factory's
dataset/template/mm-plugin/collator/trainer boundary and the released ODB
package hook. Public data, hardware, and launch environments can differ from
the paper's experimental setup, so example results should be read as workflow
validation rather than replacements for the paper's controlled benchmark
tables.

## Public Example Projects

The public example repositories answer a practical question: can a reader
follow the README, train a model, and run validation loss plus benchmark
evaluation?

| Example | Purpose |
| --- | --- |
| [online-dynamic-batching](https://github.com/online-dynamic-batching/online-dynamic-batching) | Package tests, synthetic benchmark, and framework adapter unit tests. |
| [build-mm-mix-dataset](https://github.com/online-dynamic-batching/build-mm-mix-dataset) | Builds the local public MM-Mix-style dataset used by the framework examples. |
| [odb-example-hf-trainer](https://github.com/online-dynamic-batching/odb-example-hf-trainer) | HF Trainer path with a model-specific lazy tensor pipeline. |
| [odb-example-llamafactory](https://github.com/online-dynamic-batching/odb-example-llamafactory) | LLaMA-Factory path using the ODB hook and LLaMA-Factory-compatible processing. |
| [odb-example-accelerate](https://github.com/online-dynamic-batching/odb-example-accelerate) | Native Accelerate loop path. |
| [odb-example-lightning](https://github.com/online-dynamic-batching/odb-example-lightning) | Native Lightning Trainer path. |

Result JSON files in those repositories are example run records. They help
catch integration regressions, but they should not be presented as replacements
for the paper's controlled benchmark tables.
