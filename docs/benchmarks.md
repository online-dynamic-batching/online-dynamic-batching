# Benchmarks

ODB benchmark numbers should report literal emitted-sample throughput:

```text
train-split emitted samples / wall-clock training time
```

Do not use trainer counters that multiply fixed batch size by update count under
dynamic batching.

## Paper Highlights

The table reports emitted-sample throughput speedup versus the fixed-batch
Standard baseline. Quality notes use the paper's validation metrics.

| Workload | Setting | ODB result | Quality note |
| --- | --- | ---: | --- |
| MM-Mix | 2-node 16xH20, Qwen3-VL-2B Full FT | 4.43x vs Standard | MMMU-MC 46.31 +/- 0.44% vs 43.33 +/- 2.24% Standard |
| LLaVA-150K | 8xH20, Qwen3-VL-8B Full FT | 1.73x vs Standard | MMMU-MC 54.08% vs 55.88% Standard |
| ShareGPT4o | 8xH20, Qwen3-VL-8B Full FT | 2.46x vs Standard | MMMU-MC 53.88% vs 52.43% Standard |
| UltraChat | 2-node 16xH20, Qwen3-VL-2B Full FT | 2.86x vs Standard | MMLU 59.09% vs 58.82% Standard |

## Public Synthetic Benchmark

The included synthetic benchmark lets users test ODB without external training
data:

```bash
python examples/synthetic_benchmark.py --device auto --num-samples 128
```

The synthetic benchmark uses a deterministic long-tail sequence distribution and
reports:

- literal emitted samples/s;
- real token/s;
- padding ratio;
- optimizer step count;
- a toy language-modeling loss.

The script is designed for integration sanity checks. It is not a replacement
for full LLM/VLM training benchmarks.

For the framework validation matrix, see [validation.md](validation.md).

## Benchmark Policy

For public claims:

- include Standard and ODB quality metrics;
- disclose the dataset source and whether the dataset is publicly
  redistributable;
- label text-only packing as a strong comparator with different integration
  requirements where applicable;
- prefer a quality-safe ODB config over the fastest config when they differ.

## Why MM-Mix Is The Stress Case

MM-Mix mixes OCR, VQA, and captioning samples. Its bimodal length distribution
contains many short examples and a long tail. That is the regime where ODB can
aggregate short multimodal samples online without relying on a stale length
cache.
