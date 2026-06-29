# Hugging Face Trainer Integration

Install with the optional dependency group:

```bash
pip install "online-dynamic-batching[hf]"
```

## Run A Full Example

For a complete public multimodal workflow with data preparation, ODB training,
fixed-batch baseline training, validation loss, and MMMU-MC evaluation, use
[`odb-example-hf-trainer`](https://github.com/online-dynamic-batching/odb-example-hf-trainer).

This guide explains the package API and the required HF Trainer data-pipeline
contract. The example repository shows the same contract in a runnable
Qwen3-VL MM-Mix-style project.

## Data Pipeline Contract

Hugging Face Trainer supports multimodal model training once the batch already
contains tensors accepted by `model.forward`. ODB additionally needs to observe
the real post-processing length before grouping. That means tokenizer,
processor, chat-template, truncation, visual-token expansion, and label masking
should happen per sample in the Dataset or an equivalent processor adapter.
The collator should only pad/stack ODB groups.

This adapter does not implement model-specific multimodal preprocessing. For
raw text/image records, first provide a Dataset or processor adapter that emits
single-sample tensor dicts with the semantics your model expects. Then use
`enable_odb(...)` to wire dynamic grouping and Trainer accounting.

`enable_odb(...)` validates this contract by checking an explicit Dataset
declaration (`odb_ready=True`) or by sampling the training dataset and checking
that items already contain `input_ids`. If raw text/images are still processed
inside the collator, ODB cannot know true sample lengths before grouping.

## Recommended Hook: `enable_odb(...)`

Use this when you can provide an ODB-ready DataLoader and a regular
`transformers.Trainer`:

```python
from odb.integrations.hf import enable_odb
from transformers import Trainer

trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=data_collator)
train_loader = trainer.get_train_dataloader()

enable_odb(
    trainer,
    train_dataloader=train_loader,
    train_dataset=train_dataset,
    token_budget=8192,
    loss_scaling="exact",
    join=True,
)
```

`enable_odb(...)` applies ODB to the DataLoader, registers Trainer accounting,
sets up metadata/loss-scaling consumption, and keeps `trainer.train()` on the
ODB-enabled DataLoader.

The recommended path works with a regular `transformers.Trainer`. Use
`ODBTrainer` or `ODBTrainerMixin` when you prefer a native Trainer subclass
instead of method wrapping.

## Alternative Hooks

Pick one lower-level hook point only when it better matches your codebase.
These paths are alternatives, not a checklist.

| If you can... | Use this path |
| --- | --- |
| Choose or subclass the Trainer class | Native Trainer hook: `ODBTrainer` or `ODBTrainerMixin` |
| Apply ODB to the DataLoader yourself | Existing DataLoader hook: `odb.apply(...)` plus `configure_trainer(...)` |
| Edit the training step directly | Manual contract: `pop_step_info`, scale loss, and account emitted samples |

### Native Trainer Hook

Use `ODBTrainer` when you can choose the Trainer class. For custom Trainer
classes, put the mixin before the concrete Trainer:

```python
from odb.integrations.hf import ODBTrainerMixin

class ProjectTrainer(ODBTrainerMixin, CustomTrainer):
    pass
```

`ODBTrainerMixin` consumes ODB metadata inside `compute_loss`, removes transport
metadata before model forward, and applies `info.loss_scale`.

### Existing DataLoader Hook

Use `configure_trainer(...)` directly when you have already called
`odb.apply(...)` yourself:

```python
import odb
from odb.integrations.hf import configure_trainer
from transformers import Trainer

trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=data_collator)
train_loader = trainer.get_train_dataloader()

handle = odb.apply(train_loader, token_budget=8192, loss_scaling="exact", join=True)

configure_trainer(
    trainer,
    dataloader=train_loader,
    handle=handle,
    sample_budget=len(train_dataset) * args.num_train_epochs,
    scheduler_progress="samples",
    max_steps_policy="overwrite",
)
```

`configure_trainer(...)` wraps `compute_loss`, registers the sample-budget
callback, keeps `trainer.train()` on the ODB-enabled DataLoader passed as
`dataloader=`, and keeps Trainer progress aligned with emitted samples. If the
trainer is an `ODBTrainerMixin` subclass, it sets the ODB loss-scaling mode and
does not wrap `compute_loss` again.

### Manual Contract

If you cannot use the native class or existing-trainer helper, the Trainer must
do the same work:

```python
info = odb.pop_step_info(inputs, loss_scaling="exact")
loss = original_compute_loss(model, inputs)
loss = loss * info.loss_scale
state.emitted_samples += info.all_samples_this_step
```

Do not pass ODB transport metadata into `model(**inputs)`.

## Runtime Settings

Runtime behavior such as `join`, exact loss scaling, sample progress,
`token_budget`, buffer/prefetch, `group_order_flip`, ODB buffer-fill warm-up,
and PyTorch multiprocessing sharing strategy is shared across frameworks. See
[Runtime Settings](../runtime-settings.md).
