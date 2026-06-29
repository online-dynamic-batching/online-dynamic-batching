#!/usr/bin/env python3
"""Audit a repository for likely ODB integration gaps."""

from __future__ import annotations

import pathlib
import re
import sys


CHECKS = {
    "imports_odb": re.compile(r"\bimport odb\b|from odb\b"),
    "uses_odb_dataloader": re.compile(r"ODBDataLoader"),
    "uses_apply": re.compile(r"odb\.apply\("),
    "uses_pop_step_info": re.compile(r"pop_step_info\("),
    "uses_loss_scale": re.compile(r"loss_scale"),
    "uses_all_samples": re.compile(r"all_samples_this_step"),
    "uses_trainer_adapter": re.compile(r"configure_trainer\(|enable_odb\("),
    "uses_llamafactory_enable": re.compile(r"enable_odb\("),
    "legacy_flat_keys": re.compile(r"total_batch_size|local_batch_size|odb_local_tokens|odb_total_tokens"),
    "non_unit_batch_size": re.compile(
        r"batch_size\s*[=:]\s*(?:[2-9]|\d{2,})|per_device_train_batch_size\s*[=:]\s*(?:[2-9]|\d{2,})"
    ),
    "processor_in_collator": re.compile(
        r"(collat|Collator)[\s\S]{0,1200}(processor\(|tokenizer\(|apply_chat_template|AutoProcessor)"
    ),
}


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    hits = {name: [] for name in CHECKS}

    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for name, pattern in CHECKS.items():
            if pattern.search(text):
                hits[name].append(str(path.relative_to(root)))

    for name, files in hits.items():
        status = "OK" if files else "MISSING"
        print(f"{status}: {name}")
        for file_name in files[:10]:
            print(f"  - {file_name}")

    if hits["imports_odb"] and not hits["uses_pop_step_info"] and not hits["uses_trainer_adapter"]:
        print("\nWARNING: ODB is imported but no pop_step_info or HF adapter usage was found.")
    if hits["legacy_flat_keys"]:
        print("\nNOTE: legacy flat ODB metadata keys found; prefer pop_step_info in new trainer code.")
    if hits["non_unit_batch_size"]:
        print("\nWARNING: non-unit batch size patterns found. ODB DataLoaders must emit batch_size=1 samples.")
    if hits["processor_in_collator"]:
        print(
            "\nWARNING: processor/tokenizer work appears near collator code. "
            "Strict ODB requires model-ready single-sample tensors before grouping."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
