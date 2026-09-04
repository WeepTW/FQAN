#!/usr/bin/env python3
"""Backfill missing Experiment 7 Mistral dev matched-json artifacts.

This script does not run retriever inference.  It verifies that existing
formal finqa_dev matched-json artifacts are row-aligned with the original
prompt-mode CSV, then copies the same prompt-mode rows into a derived
Mistral-dev artifact with explicit provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from experiment6_paths import PATHS


CASES = {
    "finqa_mistral_o": ("original", "finqa_flan_o", "finqa_t5gemma2_o"),
    "finqa_mistral_z": ("zero_shot", "finqa_flan_z", "finqa_t5gemma2_z"),
    "finqa_mistral_m": ("many_shot", "finqa_flan_m", "finqa_t5gemma2_m"),
    "finqa_mistral_d": ("dynamic_shot", "finqa_flan_d", "finqa_t5gemma2_d"),
}


def normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def load_csv_questions(path: Path) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return [normalize_question(row.get("Question", "")) for row in csv.DictReader(handle)]


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"matched-json must be a list: {path}")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"matched-json rows must be objects: {path}")
    return payload


def source_path(repo_root: Path, source_retriever: str) -> Path:
    return (
        repo_root
        / "Experiment"
        / "experiment_7_target_selection_gpt55_all_cases_20260612T012548Z"
        / "retriever_sources"
        / f"{source_retriever}_finqa_dev"
        / "best_matched_with_retrieved_facts_and_questions.json"
    )


def choose_source(repo_root: Path, prompt_questions: list[str], candidates: tuple[str, str]) -> tuple[str, Path, list[dict[str, Any]]]:
    errors: list[str] = []
    for retriever in candidates:
        path = source_path(repo_root, retriever)
        if not path.is_file():
            errors.append(f"{retriever}: missing {path}")
            continue
        rows = load_json_rows(path)
        row_questions = [normalize_question(str(row.get("question", ""))) for row in rows]
        if row_questions != prompt_questions:
            errors.append(f"{retriever}: question order mismatch rows={len(rows)} csv={len(prompt_questions)}")
            continue
        return retriever, path, rows
    raise RuntimeError("; ".join(errors))


def build_backfilled_rows(rows: list[dict[str, Any]], *, target_retriever: str, source_retriever: str, source_json: Path, csv_path: Path) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = deepcopy(row)
        item["experiment7_backfill_provenance"] = {
            "status": "backfilled_from_existing_finqa_dev_matched_json",
            "reason": "mistral dev retriever artifact absent; user requested question-aligned reuse without rerunning retriever",
            "target_retriever_id": target_retriever,
            "source_retriever_id": source_retriever,
            "source_matched_json": str(source_json),
            "target_input_csv": str(csv_path),
            "source_row_index": index,
            "created_at_utc": now,
        }
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace-root", type=Path, default=PATHS.workspace)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    workspace_root = args.workspace_root.resolve()
    output_root = args.output_root.resolve()
    manifest_items: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for target_retriever, (prompt_dir, primary_source, secondary_source) in CASES.items():
        csv_path = workspace_root / "data" / f"finqa_{prompt_dir}" / "finqa_dev_rel_fact_instruction.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        questions = load_csv_questions(csv_path)
        source_retriever, source_json, source_rows = choose_source(repo_root, questions, (primary_source, secondary_source))
        target_dir = output_root / f"{target_retriever}_finqa_dev"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_json = target_dir / "best_matched_with_retrieved_facts_and_questions.json"
        target_rows = build_backfilled_rows(
            source_rows,
            target_retriever=target_retriever,
            source_retriever=source_retriever,
            source_json=source_json,
            csv_path=csv_path,
        )
        target_json.write_text(json.dumps(target_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_items.append(
            {
                "target_retriever_id": target_retriever,
                "dataset": "finqa_dev",
                "prompt_dir": prompt_dir,
                "rows": len(target_rows),
                "source_retriever_id": source_retriever,
                "source_matched_json": str(source_json),
                "target_matched_json": str(target_json),
                "question_order_verified": True,
            }
        )

    manifest = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "experiment7_mistral_dev_matched_json_backfill",
        "policy": "reuse existing question-aligned finqa_dev matched-json; no retriever, RetFact, or in-context selection rerun",
        "items": manifest_items,
        "status": "completed",
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
