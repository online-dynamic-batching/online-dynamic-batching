# LLaMA-Factory Integration

LLaMA-Factory usually owns DataLoader construction and is built on Hugging Face
Trainer. Use this adapter path when your training stack is LLaMA-Factory or a
similar HF-Trainer-based integration.

Install with the optional dependency group:

```bash
pip install "online-dynamic-batching[hf]"
```

## Choose Your Path

LLaMA-Factory integration has two audiences:

| If you are... | Use this path |
| --- | --- |
| A user who wants to run ODB with LLaMA-Factory | Start from the runnable example project. |
| A LLaMA-Factory maintainer, contributor, or downstream integrator | Add the ODB hook at the internal Trainer/DataLoader construction point. |

## Path 1: Run The Example Project

For a complete public multimodal workflow with LLaMA-Factory setup, data
preparation, ODB training, fixed-batch baseline training, validation loss, and
MMMU-MC evaluation, use
[`odb-example-llamafactory`](https://github.com/online-dynamic-batching/odb-example-llamafactory).

This is the recommended path for most users. The example prepares a
compatible LLaMA-Factory checkout, builds the public data, launches training,
and runs evaluation. You do not need to call `configure_trainer(...)` directly.

## Path 2: Add ODB To The LLaMA-Factory Training Pipeline

Use this path when you maintain LLaMA-Factory itself, contribute to its
training stack, or maintain a compatible downstream branch. The goal is to keep
LLaMA-Factory's model and data semantics intact, then call `enable_odb(...)`
after LLaMA-Factory has constructed the Trainer and training DataLoader.

## Processor Boundary

Model-specific work belongs before ODB grouping. Keep LLaMA-Factory's original
dataset, template, tokenizer, processor, and multimodal plugin semantics. For
Qwen-VL, LLaVA, InternVL, or future model families, the LLaMA-Factory-side lazy
dataset or equivalent processor adapter should perform:

- chat-template rendering;
- tokenizer calls;
- image or video processor calls;
- visual-token expansion and truncation;
- label masking.

The output item must be an ODB-ready single-sample tensor dict. At minimum, it
should contain `input_ids`, `attention_mask`, and `labels`; multimodal models
should also include the tensor keys expected by their model forward, such as
`pixel_values`, `image_grid_thw`, `pixel_values_videos`, or `video_grid_thw`.

`enable_odb(...)` does not choose or replace the model processor. It validates
that the current DataLoader is already ODB-ready, applies ODB grouping, and
configures the Trainer to consume ODB metadata.

## YAML Surface For The Training Pipeline

Expose ODB-specific options in the LLaMA-Factory training config:

```yaml
use_odb: true
odb_token_budget: 8192
odb_loss_scaling: exact
odb_join: true  # default runtime setting

per_device_train_batch_size: 1
dataloader_num_workers: 4
dataloader_prefetch_factor: 256
```

## Training-Pipeline Hook: `enable_odb(...)`

After LLaMA-Factory constructs its training DataLoader and Trainer, call the
high-level adapter once:

```python
from odb.integrations.llamafactory import enable_odb

enable_odb(
    trainer,
    train_dataloader=train_dataloader,
    training_args=training_args,
    train_dataset=train_dataset,
    token_budget=args.odb_token_budget,
    loss_scaling="exact",
    join=args.odb_join,
    scheduler_progress="samples",
)
```

`enable_odb(...)` validates the LLaMA-Factory surface, checks that the data
pipeline is ODB-ready, then delegates to the lower-level Trainer adapter. It
infers `sample_budget` from `train_dataset` and `num_train_epochs`, validates
`per_device_train_batch_size=1`, validates worker prefetching, and registers
the ODB callbacks and exact loss scaling needed by the Trainer.

## Integration Contract

- `per_device_train_batch_size` must be `1`.
- the DataLoader must use `batch_size=1` and `num_workers>0`.
- each dataset item sampled before grouping must already contain `input_ids`.
- ODB metadata must be consumed before model forward.
- `info.loss_scale` must multiply the loss before backward.
- `info.all_samples_this_step` must drive sample-progress accounting.

If `enable_odb(...)` reports that `input_ids` is missing, the current
LLaMA-Factory pipeline is probably still running tokenizer/processor work
inside the collator after ODB would group samples. Move only the single-sample
tensorization boundary before ODB grouping; do not reimplement LLaMA-Factory
templates or multimodal plugins inside ODB.

## Advanced Helper: `configure_trainer(...)`

`configure_trainer(...)` is a lower-level helper for maintainers and downstream
integrators who already control the LLaMA-Factory Trainer/DataLoader hook
point. Call it only when ODB has already been applied to the training
DataLoader and you only need to wire Trainer accounting, callbacks, and loss
scaling by hand.

Most LLaMA-Factory integrations should call `enable_odb(...)` instead.

## Runtime Settings

Runtime behavior such as `odb_join`, exact loss scaling, sample progress,
`odb_token_budget`, buffer/prefetch, `group_order_flip`, ODB buffer-fill
warm-up, and PyTorch multiprocessing sharing strategy is shared across
frameworks. See [Runtime Settings](../runtime-settings.md).
