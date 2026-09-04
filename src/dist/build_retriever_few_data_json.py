#!/usr/bin/env python3
"""Build an Experiment-local FinQA-style JSON file from a few-shot retriever CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def row_text(row: dict[str, str]) -> str:
    for key in ("Sentence", "text", "Text"):
        value = row.get(key)
        if value and value.strip():
            return value
    pieces = [row.get("Pre_Text", ""), row.get("Post_Text", "")]
    return " ".join(piece for piece in pieces if piece)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    records: list[dict[str, str]] = []
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                {
                    "question": row.get("Question") or row.get("question") or "",
                    "text": row_text(row),
                    "table_text": row.get("Table_Text") or row.get("table_text") or "",
                }
            )
    if not records:
        raise ValueError(f"{args.input_csv} has no rows")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "output_json": str(args.output_json),
                "rows": len(records),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
