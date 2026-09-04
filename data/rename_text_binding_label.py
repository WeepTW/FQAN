#!/usr/bin/env python3
"""Rename legacy retriever binding prompt headers to Retriever + Data Binding.

This scans text-like files under data/, including prompt CSV, JSON previews,
and local data-generation Python code. Binary files are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# Keep these split so this migration script does not match itself.
OLD_LABELS = (
    "# Retriever + " + "Text" + "-Binding",
    "# Retriever + " + "Text" + " Binding",
)
NEW_LABEL = "# Retriever + Data Binding"

SKIP_DIRS = {"__pycache__"}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def iter_candidate_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def replace_file(path: Path, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    count = 0
    updated = text
    for old_label in OLD_LABELS:
        occurrences = updated.count(old_label)
        if occurrences:
            count += occurrences
            updated = updated.replace(old_label, NEW_LABEL)
    if count and not dry_run:
        path.write_text(updated, encoding="utf-8", newline="")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0
    changed_files = 0
    for path in iter_candidate_files(args.root):
        count = replace_file(path, args.dry_run)
        if count:
            changed_files += 1
            total += count
            print(f"{path}: {count}")

    mode = "dry_run" if args.dry_run else "updated"
    print(f"{mode}: files={changed_files} replacements={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
