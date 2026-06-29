# Changelog

## Unreleased

- No unreleased changes.

## 0.1.2 (2026-06-24)

### Added
- Added high-level `enable_odb(...)` entry points for HuggingFace Trainer and
  LLaMA-Factory integrations.
- Added pipeline-readiness checks for one-sample tensorized datasets,
  DataLoader `batch_size=1`, and worker prefetching before ODB grouping.
- Updated framework integration guides and agent-assisted integration audits.

## 0.1.1 (2026-06-19)

### Fixed
- Stabilized tensor sharing in the ODB collate worker for multiprocessing and
  DDP smoke-test environments.
- Updated CI smoke tests to use PyTorch's filesystem sharing strategy on
  runners where file-descriptor tensor sharing is unreliable.

## 0.1.0 (2026-06-18)

### Added
- Added `ODBConfig`, `ODBHandle`, and clean `token_budget` / `join` API names while keeping legacy aliases.
- Added `ODBDataLoader` as the preferred DataLoader replacement API for new PyTorch code.
- Added `ODBStepInfo` and `odb.pop_step_info(batch)` so trainer callbacks consume `all_samples_this_step` and `loss_scale` instead of multiple flat metadata keys.
- Added `odb.integrations.hf.configure_trainer(...)` with explicit `sample_budget`, `max_optimizer_steps`, `scheduler_progress`, and `max_steps_policy`.
- Added LLaMA-Factory and LLaVA-Factory adapter entry points that delegate to the HuggingFace Trainer bridge.
- Added public docs, integration guides, synthetic benchmark examples, and CPU/DDP smoke tests.

### Changed
- Updated `odb.apply(...)` to return an `ODBHandle`.
- Made join-mode termination the default. Pass `join=False` only for runtimes that cannot support drain-before-finish semantics.
- Kept `setup_odb_training(...)` as a compatibility wrapper around the new HuggingFace adapter.
- Updated examples and README to prefer the clean API.
