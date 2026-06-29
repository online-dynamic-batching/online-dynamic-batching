# Engineering Roadmap

This roadmap tracks planned ODB runtime, integration, and validation
capabilities. Packaging, release, and repository administration tasks are
tracked separately.

## Supported Today

ODB currently provides:

- DataLoader-level integration through `ODBDataLoader(...)` and
  `odb.apply(dataloader, ...)`.
- Default join-mode DataLoader-side drain for distributed training.
- Trainer-facing step metadata through `odb.pop_step_info(batch)`.
- Framework examples and adapters for PyTorch loops, Hugging Face Trainer,
  LLaMA-Factory-style trainers, Accelerate, and Lightning.
- CPU smoke tests, DDP smoke tests, synthetic benchmarks, and release
  validation notes.

## Distributed Training Semantics

- Profile larger world sizes for the metadata synchronization path and document
  when flat `all_gather` is sufficient versus when hierarchical alignment may
  be useful.
- Expand validation for common large-model distributed settings, including
  ZeRO-3 and FSDP.
- Extend isolated gradient-accumulation validation into a documented guide and
  CI-friendly smoke path.
- Improve diagnostics for distributed startup failures such as Gloo/TCPStore
  initialization errors and misconfigured network interfaces.

## Trainer Interface

- Keep the trainer-facing contract narrow: framework code should consume
  `info.all_samples_this_step` and `info.loss_scale`, while debug-only metadata
  remains optional.
- Make sample-budget stopping, optimizer-step caps, epoch boundaries, and
  scheduler progress explicit in each adapter guide.
- Add regression tests that compare loss, emitted-sample accounting, and
  stopping behavior across the supported DataLoader and trainer-adapter modes.
- Add more examples for framework-owned DataLoader construction, where users
  cannot directly replace the DataLoader class.

## Batching Policies

- Add pluggable grouping policies beyond the current sorted-window token-budget
  policy, while keeping the default policy simple and predictable.
- Add optional token-budget auto-tuning from observed memory headroom, with
  conservative behavior after OOM or near-OOM events.
- Support streaming and `IterableDataset` workloads where global dataset length
  and full-epoch identity coverage are not always available.
- Expose policy-level debug summaries that explain why samples were grouped or
  deferred without exposing sample contents.

## Observability And Benchmarking

- Emit structured runtime metrics for samples/s, real tokens/s, padded tokens/s,
  padding ratio, local group counts, IDLE steps, and per-rank imbalance.
- Add JSON/CSV output to the synthetic benchmark and validation scripts so
  results can be compared in CI and user environments.
- Provide reproducible public benchmark configs for package validation and
  performance regression tracking.
- Add built-in summaries for DataLoader worker behavior, file descriptor
  pressure, multiprocessing sharing strategy, and distributed network setup.
