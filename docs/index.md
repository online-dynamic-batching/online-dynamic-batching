# Online Dynamic Batching Docs

ODB is a PyTorch DataLoader-side online batcher for LLM/VLM fine-tuning. It
observes real sequence length after preprocessing, tokenization, truncation,
augmentation, and visual-token expansion, then forms DDP-aligned dynamic
batches.

## Main Reading Paths

- [Quickstart](quickstart.md): install and run a small public demo.
- [Integration Guides](integration-guides/README.md): choose one framework path.
- [Grouping Algorithm](GROUPING_ALGORITHM.md): online grouping and DDP step
  alignment at a high level.
- [API Design And Adapter Notes](API_DESIGN_NOTES.md): compatibility boundary
  and adapter contracts for maintainers.
- [Runtime Settings](runtime-settings.md): shared knobs such as `join`,
  `loss_scaling`, `token_budget`, buffer/prefetch, and multiprocessing settings.
- [Benchmarks](benchmarks.md): benchmark reporting policy and paper highlights.
- [Validation](validation.md): what release validation checks.

## Integration Boundary

```text
raw records
  -> tokenizer / processor / template / multimodal adapter
  -> ODB-ready single-sample tensor dict
  -> ODB dynamic grouping
  -> trainer or training loop
```

ODB does not replace model-specific preprocessing. It accelerates the tensor
stream your chosen training stack already defines.

For the full documentation map, see [README](README.md).
