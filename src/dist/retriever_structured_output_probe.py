#!/usr/bin/env python3
"""Audit non-original retriever predictions without post-hoc JSON wrapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from result_organization import prediction_records, prediction_schema_failure, prediction_text  # noqa: E402
from retriever_json_schema import (  # noqa: E402
    format_backend_prediction,
    normalize_prompt_mode,
    parse_retfact_schema,
    schema_required,
)


def true_prefix(record: str) -> str:
    marker = "Pred:"
    index = record.find(marker)
    if index < 0:
        return "True:  Pred:"
    return record[: index + len(marker)]


def normalize_record(record: str, prompt_mode: str) -> str:
    predicted = prediction_text(record)
    structured_prediction = format_backend_prediction(predicted, prompt_mode, "assembler")
    return f"{true_prefix(record)} {structured_prediction}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-txt", required=True, type=Path)
    parser.add_argument("--prompt-mode", required=True)
    parser.add_argument("--output-txt", type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    prompt_mode = normalize_prompt_mode(args.prompt_mode)
    records = prediction_records(args.input_txt.read_text(encoding="utf-8"))
    if args.limit > 0:
        records = records[: args.limit]

    normalized = [normalize_record(record, prompt_mode) for record in records]
    schema_failures = sum(
        prediction_schema_failure(record, prompt_mode) is not None
        for record in normalized
    )
    valid_nonempty = 0
    if schema_required(prompt_mode):
        for record in normalized:
            result = parse_retfact_schema(prediction_text(record))
            if result.valid and result.ret_fact.strip():
                valid_nonempty += 1

    if args.output_txt is not None:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text("\n".join(normalized) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "input_txt": str(args.input_txt),
                "output_txt": str(args.output_txt) if args.output_txt else None,
                "prompt_mode": prompt_mode,
                "records": len(normalized),
                "schema_required": schema_required(prompt_mode),
                "schema_failures_after_structured_parse": schema_failures,
                "valid_nonempty_retfact_after_structured_parse": valid_nonempty,
                "note": "Formal non-original routes may use any format_backend, but the final output must validate against the new_prompt RetFact/Binding/Reason JSON schema. This probe applies the universal schema assembler.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if schema_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
