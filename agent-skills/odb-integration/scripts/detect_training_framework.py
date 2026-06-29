#!/usr/bin/env python3
"""Heuristically detect training frameworks in a repository."""

from __future__ import annotations

import pathlib
import re
import sys


PATTERNS = {
    "hf_trainer": re.compile(r"\bTrainer\b|transformers\.Trainer|TrainingArguments"),
    "llamafactory": re.compile(r"llamafactory|LLaMA-Factory|llama_factory", re.I),
    "accelerate": re.compile(r"\bAccelerator\b|accelerate\."),
    "lightning": re.compile(r"lightning\.pytorch|pytorch_lightning|LightningModule|LightningDataModule"),
    "pytorch_loop": re.compile(r"DataLoader|loss\.backward\(|optimizer\.step\("),
}


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    counts = {name: 0 for name in PATTERNS}
    files_seen = {name: [] for name in PATTERNS}

    for path in root.rglob("*"):
        if path.is_dir() or path.suffix not in {".py", ".yaml", ".yml", ".toml", ".md"}:
            continue
        if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                counts[name] += 1
                if len(files_seen[name]) < 8:
                    files_seen[name].append(str(path.relative_to(root)))

    for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if count:
            print(f"{name}: {count}")
            for file_name in files_seen[name]:
                print(f"  - {file_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
