# Contributing

Thanks for improving Online Dynamic Batching.

## Development Setup

```bash
python -m pip install -e ".[dev,hf]"
python -m pytest
python -m ruff check .
```

## What To Include In A PR

- A concise description of the training stack or bug being addressed.
- A focused test when behavior changes.
- Throughput numbers only when the sample-count definition is clear.
- Quality numbers for benchmark claims; speed without quality is not enough.

## Design Principles

- Keep ODB a DataLoader-level integration.
- Avoid model-specific assumptions in core code.
- Prefer explicit batch metadata over implicit trainer counters.
- Preserve DDP step alignment.

## Reporting Benchmarks

Use literal emitted-sample throughput:

```text
train-split emitted samples / wall-clock training time
```

Do not use trainer counters that assume fixed batch size under dynamic batching.
