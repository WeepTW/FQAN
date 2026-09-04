#!/usr/bin/env python3
"""Validate the public or formal FQAN data layout using relative paths only."""

from __future__ import annotations

import argparse
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
PUBLIC_FILES = (
    "data/src/FinQA/dev.json",
    "data/src/FinQA/private_test.json",
    "data/src/narratives/narrative2.xlsx",
    "data/finqa_original/finqa_train_rel_fact_instruction.csv",
    "data/finqa_original/finqa_dev_rel_fact_instruction.csv",
    "data/finqa_original/finqa_test_rel_fact_instruction.csv",
)
DERIVED_MODES = ("finqa_zero_shot", "finqa_many_shot", "finqa_dynamic_shot")
FORMAL_FILES = tuple(
    f"data/{mode}/finqa_{split}_rel_fact_instruction.csv"
    for mode in DERIVED_MODES
    for split in ("train", "dev", "test")
) + tuple(
    f"data/src/FINDER/finqa_{split}_rel_fact_instruction.csv"
    for split in ("train", "dev", "test")
)


def lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            first = handle.readline(128)
    except OSError:
        return False
    return first == b"version https://git-lfs.github.com/spec/v1\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("public", "formal"), default="public")
    args = parser.parse_args()
    required = PUBLIC_FILES + (FORMAL_FILES if args.profile == "formal" else ())
    missing = []
    pointers = []
    for relative in required:
        path = WORKSPACE / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
        elif lfs_pointer(path):
            pointers.append(relative)
    if missing or pointers:
        for relative in missing:
            print(f"missing: {relative}")
        for relative in pointers:
            print(f"git-lfs-content-not-loaded: {relative}")
        print("Data check blocked. See data/README.md for licensed download and placement steps.")
        return 2
    print(f"data profile ready: {args.profile} ({len(required)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
