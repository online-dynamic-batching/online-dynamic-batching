# Quickstart

This page is for first contact: install ODB, run a public toy training demo,
and then choose the integration guide that matches your stack.

## Install

```bash
pip install online-dynamic-batching
```

Optional framework extras:

```bash
pip install "online-dynamic-batching[hf]"          # Hugging Face Trainer / LLaMA-Factory-style adapters
pip install "online-dynamic-batching[accelerate]"  # Accelerate loops
pip install "online-dynamic-batching[lightning]"   # Lightning Trainer
```

## Run A Public Toy Benchmark

From a cloned repository:

```bash
git clone https://github.com/online-dynamic-batching/online-dynamic-batching.git
cd online-dynamic-batching
python -m pip install -e .
python examples/synthetic_benchmark.py --device auto --num-samples 128
```

The script runs a tiny training loop on a synthetic long-tail sequence dataset.
It prints emitted samples/s, real token/s, padding ratio, step count, and a toy
loss for fixed batching and ODB. It is a functional demo, not a paper benchmark.

## What The Demo Shows

- ODB receives one fully processed sample at a time.
- ODB forms dynamic groups under a token budget.
- The training loop removes ODB metadata with `odb.pop_step_info(...)`.
- The loss is multiplied by `info.loss_scale`.
- Padding can drop on variable-length data because short and long samples no
  longer share the same fixed-size batch shape.

## Next Step

Choose the guide that matches your training stack:

| Stack | Guide |
| --- | --- |
| PyTorch loop | [`pytorch-loop.md`](integration-guides/pytorch-loop.md) |
| Hugging Face Trainer | [`hf-trainer.md`](integration-guides/hf-trainer.md) |
| LLaMA-Factory | [`llamafactory.md`](integration-guides/llamafactory.md) |
| Accelerate | [`accelerate.md`](integration-guides/accelerate.md) |
| Lightning | [`lightning.md`](integration-guides/lightning.md) |
