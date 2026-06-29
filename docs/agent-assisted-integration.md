# Agent-Assisted Integration

ODB includes a small coding-agent skill for projects where a training framework
needs explicit adapter edits. The skill is intentionally separate from the
runtime package: it helps an agent inspect a codebase, choose the right adapter
pattern, and verify that ODB metadata is consumed correctly.

## Skill Location

```text
agent-skills/odb-integration/
```

The skill contains:

- `SKILL.md`: core workflow and trigger description.
- `references/`: framework-specific integration notes.
- `scripts/detect_training_framework.py`: heuristic framework detector.
- `scripts/audit_odb_integration.py`: heuristic ODB integration audit.

For LLaMA-Factory-style forks, use the skill in this order: prefer an official
post-DataLoader hook that calls `enable_odb(...)`; if the fork does not have
that hook, let the agent add the thin hook after DataLoader and Trainer
construction; fall back to manual `odb.apply(...)` plus `configure_trainer(...)`
only when writing framework glue yourself.

## Codex

For local Codex use, copy or symlink the skill directory into the Codex skills
directory:

```bash
mkdir -p ~/.codex/skills
ln -s /path/to/online-dynamic-batching/agent-skills/odb-integration ~/.codex/skills/odb-integration
```

Then ask Codex to use `$odb-integration` in the target training repository.

## Claude Code

The same `SKILL.md`, `references/`, and `scripts/` layout is written to be
portable to Claude Code project or user skills. Install it in the equivalent
Claude Code skill location for the target environment, then ask the agent to
apply the ODB integration skill.

## Audit Helpers

From the target training repository, run:

```bash
python /path/to/online-dynamic-batching/agent-skills/odb-integration/scripts/detect_training_framework.py .
python /path/to/online-dynamic-batching/agent-skills/odb-integration/scripts/audit_odb_integration.py .
```

These scripts are heuristics. They are useful for finding likely files and
missing integration points, but an agent or engineer should still inspect the
actual training loop before editing.

The audit also flags likely raw-data pipeline mistakes, such as processor work
inside a collator before ODB can observe `input_ids`, non-unit batch sizes, and
missing Trainer metadata consumption.

## Expected Adapter Contract

The agent should converge on the same runtime contract used by the package:

```python
info = odb.pop_step_info(batch, loss_scaling="exact")
loss = model(**batch).loss
loss = loss * info.loss_scale
progress += info.all_samples_this_step
```

The trainer should not depend on legacy flat transport keys. For join mode,
ODB's DataLoader-side join protocol must be paired with framework-level uneven
input handling such as DDP Join or `Accelerator.join_uneven_inputs(...)`.
