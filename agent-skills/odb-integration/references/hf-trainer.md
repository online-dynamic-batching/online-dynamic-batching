# HuggingFace Trainer

Choose one Trainer mode:

- Recommended: use `enable_odb(...)` when the project already has an ODB-ready
  DataLoader whose Dataset emits tensor samples containing `input_ids`.
- Native class: use `ODBTrainer` for normal HuggingFace Trainer code.
- Native mixin: use `class ProjectTrainer(ODBTrainerMixin, CustomTrainer)` for framework forks.
- Existing instance: use `configure_trainer(...)` when the Trainer is already created.

ODB-ready means tokenizer/processor/chat-template/vision expansion runs before
ODB grouping, usually in `Dataset.__getitem__` or a processor adapter. Do not
claim a raw multimodal Dataset is ready if `processor(...)`, `tokenizer(...)`,
or `apply_chat_template(...)` only runs inside the collator.

If a Dataset has already been audited to emit ODB-ready tensor samples, it may
declare `odb_ready = True`. Otherwise `enable_odb(...)` samples the Dataset and
checks that items contain `input_ids`.

Recommended hook:

```python
from odb.integrations.hf import ODBTrainer, enable_odb

trainer = ODBTrainer(model=model, args=args, train_dataset=train_dataset, data_collator=data_collator)
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

Native mixin:

```python
from odb.integrations.hf import ODBTrainerMixin

class ProjectTrainer(ODBTrainerMixin, CustomTrainer):
    pass
```

Existing instance adapter:

```python
from odb.integrations.hf import enable_odb

train_loader = trainer.get_train_dataloader()
enable_odb(
    trainer,
    train_dataloader=train_loader,
    train_dataset=train_dataset,
    token_budget=8192,
)
```

Lower-level adapter, only when `odb.apply(...)` is already called manually:

```python
import odb
from odb.integrations.hf import configure_trainer

handle = odb.apply(train_loader, token_budget=8192, loss_scaling="exact", join=True)
configure_trainer(trainer, dataloader=train_loader, handle=handle, sample_budget=len(train_dataset))
```

Do not hand-roll `compute_loss` unless the project has a custom Trainer that
cannot use the mixin or adapter. If hand-rolling, call
`odb.pop_step_info(inputs)` before forward and multiply loss by
`info.loss_scale`.
