#!/usr/bin/env python3
"""Recover only the longest identity-valid Experiment 7 output prefix.

The source JSONL is never modified. Duplicate, missing, malformed, and
out-of-order rows are reported; only a fresh target path is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from new_full_finqa_run import (
    resume_identity_mismatches,
    validate_resume_output_prefix,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append((line_number, value))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_jsonl.exists():
        raise SystemExit(f"refusing to overwrite target: {args.output_jsonl}")
    examples = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise SystemExit("input JSON must be a list")
    source_rows = load_source_rows(args.source_jsonl)
    selected: list[dict[str, Any]] = []
    selected_lines: list[int] = []
    duplicate_candidates: dict[str, list[int]] = {}
    used_lines: set[int] = set()

    for input_index, expected in enumerate(examples):
        matches = [
            (line_number, row)
            for line_number, row in source_rows
            if line_number not in used_lines
            and not resume_identity_mismatches(expected, row)
        ]
        if not matches:
            stopped_at = input_index
            break
        line_number, row = matches[0]
        selected.append(row)
        selected_lines.append(line_number)
        used_lines.add(line_number)
        if len(matches) > 1:
            duplicate_candidates[str(input_index)] = [
                candidate_line for candidate_line, _ in matches
            ]
    else:
        stopped_at = len(examples)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    validated_rows = validate_resume_output_prefix(args.output_jsonl, examples)
    if validated_rows != len(selected):
        raise SystemExit(
            f"internal prefix validation mismatch: {validated_rows} != {len(selected)}"
        )
    report = {
        "schemaVersion": 1,
        "protocol": "experiment7-output-prefix-repair-v1",
        "status": "completed",
        "inputJson": str(args.input_json),
        "inputSha256": sha256_file(args.input_json),
        "sourceJsonl": str(args.source_jsonl),
        "sourceSha256": sha256_file(args.source_jsonl),
        "sourceObjectRows": len(source_rows),
        "outputJsonl": str(args.output_jsonl),
        "outputSha256": sha256_file(args.output_jsonl),
        "recoveredPrefixRows": validated_rows,
        "stoppedAtInputIndex": stopped_at,
        "selectedSourceLines": selected_lines,
        "duplicateCandidateLinesByInputIndex": duplicate_candidates,
        "sourceModified": False,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
