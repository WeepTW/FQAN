#!/usr/bin/env python3
"""Validate retriever prompt-mode contracts before training.

This is a lightweight guard: it reads CSV prompts and existing schema helpers
only.  It does not load models, mutate data, or create experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from result_organization import prediction_fragments, prediction_retfact_text, prediction_schema_failure
from retriever_json_schema import (
    CANONICAL_BINDING_KEYS,
    CANONICAL_POSITION_KEYS,
    CANONICAL_TOP_LEVEL_KEYS,
    format_backend_prediction,
    label_for_prompt_mode,
    normalize_prompt_mode,
    parse_retfact_schema,
    retfact_label_for_training,
    schema_required,
)


ORIGINAL_FORBIDDEN_MARKERS = (
    "Retriever + Data Binding",
    "Retriever + Text-Binding",
    "Return the relevant fact and binding result and reason",
    '"RetFact"',
    '"Binding"',
    '"Reason"',
)
SCHEMA_PROMPT_MARKERS = (
    "Retriever + Data Binding",
    "binding result and reason",
)


def read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"input", "Rel_Fact"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def validate_prompt_text(prompt_mode: str, path: Path, rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    mode_requires_schema = schema_required(prompt_mode)
    for row_index, row in enumerate(rows):
        input_text = row.get("input", "")
        if not input_text.strip():
            errors.append(f"{path}: row {row_index} input is empty")
            continue
        if mode_requires_schema:
            if not any(marker.lower() in input_text.lower() for marker in SCHEMA_PROMPT_MARKERS):
                errors.append(
                    f"{path}: row {row_index} non-original prompt lacks Data Binding/schema instruction marker"
                )
        else:
            for marker in ORIGINAL_FORBIDDEN_MARKERS:
                if marker.lower() in input_text.lower():
                    errors.append(
                        f"{path}: row {row_index} original prompt contains schema marker {marker!r}"
                    )
    return errors


def validate_label_transform(prompt_mode: str, rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    mode_requires_schema = schema_required(prompt_mode)
    for row_index, row in enumerate(rows):
        rel_fact = row.get("Rel_Fact", "")
        label = retfact_label_for_training(rel_fact, prompt_mode)
        if mode_requires_schema:
            if label != label_for_prompt_mode(rel_fact, prompt_mode):
                errors.append(f"row {row_index} non-original training label is not the JSON schema target")
            result = parse_retfact_schema(label)
            if not result.valid or " ".join(result.ret_fact.split()) != " ".join(rel_fact.split()):
                errors.append(f"row {row_index} non-original JSON target does not preserve Rel_Fact")
        else:
            if label != rel_fact:
                errors.append(f"row {row_index} plain RetFact label changed from Rel_Fact")
            try:
                parsed = json.loads(label)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and {"RetFact", "Binding", "Reason"}.intersection(parsed):
                errors.append(f"row {row_index} plain RetFact label unexpectedly looks like schema JSON")
    return errors


def validate_matching_contract() -> list[str]:
    errors: list[str] = []
    plain = "Pred: fact one; fact two"
    for plain_mode in ("original", "raw"):
        if prediction_retfact_text(plain, plain_mode) != "fact one; fact two":
            errors.append(f"{plain_mode} matching no longer keeps the full plain RetFact prediction")
        if prediction_fragments(plain, plain_mode) != ["fact one", "fact two"]:
            errors.append(f"{plain_mode} matching no longer uses plain FINDER-style prediction fragments")

    schema_prediction = (
        'Pred: {"RetFact":"fact one; fact two",'
        '"Binding":[{"ObjectName":[],"DataName":"","Position":[{"Begin":[],"End":[]}],'
        '"Trend":"None","Num":[],"Text":"binding text should not be matched"}],'
        '"Reason":"reason text should not be matched"}'
    )
    fragments = prediction_fragments(schema_prediction, "zero-shot")
    if prediction_retfact_text(schema_prediction, "zero-shot") != "fact one; fact two":
        errors.append("non-original matching does not extract full RetFact text for context matching diagnostics")
    if fragments != ["fact one", "fact two"]:
        errors.append("non-original matching does not extract only RetFact fragments")
    if any("binding" in item.lower() or "reason" in item.lower() for item in fragments):
        errors.append("non-original matching leaked Binding/Reason into retrieved facts")

    wrapped_plain = "Pred: " + format_backend_prediction("fact one; fact two", "zero-shot", "assembler")
    assembled_payload = json.loads(wrapped_plain.split("Pred:", 1)[1].strip())
    if tuple(assembled_payload.keys()) != CANONICAL_TOP_LEVEL_KEYS:
        errors.append("assembler top-level JSON key order is not canonical")
    if tuple(assembled_payload["Binding"][0].keys()) != CANONICAL_BINDING_KEYS:
        errors.append("assembler Binding object key order is not canonical")
    if tuple(assembled_payload["Binding"][0]["Position"][0].keys()) != CANONICAL_POSITION_KEYS:
        errors.append("assembler Position object key order is not canonical")
    if prediction_schema_failure(wrapped_plain, "zero-shot") is not None:
        errors.append("RetFact-only model output assembler is not valid non-original schema")
    if prediction_retfact_text(wrapped_plain, "zero-shot") != "fact one; fact two":
        errors.append("assembled RetFact-only output does not expose full RetFact text")
    if prediction_fragments(wrapped_plain, "zero-shot") != ["fact one", "fact two"]:
        errors.append("assembled RetFact-only output does not match through RetFact")

    malformed = 'Pred: {"Binding":[{"Text":"binding only"}],"Reason":"reason only"}'
    if prediction_retfact_text(malformed, "zero-shot"):
        errors.append("malformed non-original schema produced full RetFact text")
    if prediction_fragments(malformed, "zero-shot"):
        errors.append("malformed non-original schema produced retrieved fragments")
    if prediction_schema_failure(malformed, "zero-shot") is None:
        errors.append("malformed non-original schema was not reported as schema failure")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-mode", required=True)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--sample-rows", type=int, default=5)
    args = parser.parse_args()

    prompt_mode = normalize_prompt_mode(args.prompt_mode)
    errors: list[str] = []

    csv_paths = [args.train_csv, args.eval_csv]
    if args.test_csv is not None:
        csv_paths.append(args.test_csv)

    for csv_path in csv_paths:
        rows = read_rows(csv_path, args.sample_rows)
        errors.extend(validate_prompt_text(prompt_mode, csv_path, rows))
        errors.extend(validate_label_transform(prompt_mode, rows))

    errors.extend(validate_matching_contract())

    if errors:
        print("prompt_mode_contract_failed")
        print(f"prompt_mode={prompt_mode} schema_required={schema_required(prompt_mode)}")
        for error in errors:
            print(f"- {error}")
        return 2

    print(
        "prompt_mode_contract_ok "
        f"prompt_mode={prompt_mode} schema_required={int(schema_required(prompt_mode))} "
        f"train_csv={args.train_csv} eval_csv={args.eval_csv} test_csv={args.test_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
