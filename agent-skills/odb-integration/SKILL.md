---
name: odb-integration
description: Integrate the online-dynamic-batching Python package into PyTorch, HuggingFace Trainer, LLaMA-Factory, Accelerate, or Lightning training code. Use when a user wants an AI coding agent to add ODB dynamic batching, wire loss scaling, sample-budget accounting, max_steps semantics, or audit whether ODB metadata is consumed correctly.
---

# ODB Integration Skill

Use this skill to modify a training codebase to use the
`online-dynamic-batching` package.

## Workflow

1. Detect the training framework and read only the matching reference:
   - PyTorch loop: `references/pytorch-loop.md`
   - HuggingFace Trainer: `references/hf-trainer.md`
   - LLaMA-Factory: `references/llamafactory.md`
   - Accelerate: `references/accelerate.md`
   - Lightning: `references/lightning.md`
2. Locate DataLoader construction and confirm whether the dataset emits
   model-ready single-sample tensors before grouping.
3. Choose one ODB DataLoader path:
   - use `odb.ODBDataLoader(...)` when the project owns DataLoader construction;
   - use `odb.apply(existing_loader, ...)` when the framework constructs the
     DataLoader for you.
4. Prefer high-level framework hooks when available. For example, after
   LLaMA-Factory constructs its Trainer and DataLoader:

```python
from odb.integrations.llamafactory import enable_odb

enable_odb(
    trainer=trainer,
    train_dataloader=train_dataloader,
    training_args=training_args,
    train_dataset=train_dataset,
    token_budget=args.odb_token_budget,
    loss_scaling="exact",
    join=True,
)
```

   If a high-level hook reports missing `input_ids`, do not group raw records
   by an estimated length field. Keep model-specific preprocessing semantics
   and move only the single-sample tensorization boundary before ODB grouping.
5. Locate loss computation and ensure ODB metadata is consumed before model
   forward:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss
loss = loss * info.loss_scale
```

6. Wire sample-progress accounting from `info.all_samples_this_step`.
7. Make `sample_budget`, `max_optimizer_steps`, `scheduler_progress`, and
   framework `max_steps` semantics explicit.
8. ODB defaults to `join=True`; ensure the framework also protects model
   collectives with DDP Join or equivalent uneven-input handling.
9. Run lint/tests or a short smoke train.

## Core Contract

ODB owns dynamic batch formation and computes trainer-facing step info:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
```

Use only:

- `info.all_samples_this_step`
- `info.loss_scale`

Do not make trainer code depend on legacy flat transport keys such as
`total_batch_size`, `local_batch_size`, `odb_local_tokens`, or
`odb_total_tokens`.

ODB must see post-pipeline lengths before grouping. Do not integrate a raw
multimodal dataset whose tokenizer or processor only runs inside the collator;
add or preserve a lazy tensor-sample path first.

## Helpers

Run these scripts from the target training repository when useful:

```bash
python /path/to/agent-skills/odb-integration/scripts/detect_training_framework.py .
python /path/to/agent-skills/odb-integration/scripts/audit_odb_integration.py .
```

The scripts are heuristics; inspect the code before editing.
