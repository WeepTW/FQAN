#!/usr/bin/env python3
"""Build a stress-oriented first-25 sample for Experiment 7 local QA.

The sampler does not change FINDER semantics. It only reorders/selects already
materialized generator-input rows so QA can cover long prompts, long expected
PoT programs, and RetFact differences across dataset/prompt-type cases.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from new_full_finqa_run import build_prompt, format_example_program_for_prompt



def normalize_lookup_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def load_gold_lookup(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            qa = item.get("qa") if isinstance(item.get("qa"), dict) else {}
            program = qa.get("program") or qa.get("program_re") or item.get("program")
            answer = qa.get("answer") or qa.get("exe_ans") or item.get("answer")
            question = qa.get("question") or item.get("question")
            gold = {
                "id": item.get("id"),
                "question": question,
                "program": program,
                "answer": answer,
                "gold_source_json": str(path),
            }
            for key in [item.get("id"), question]:
                normalized = normalize_lookup_key(key)
                if normalized and program not in (None, ""):
                    lookup[normalized] = gold
    return lookup


def backfill_gold(row: dict[str, Any], gold_lookup: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    if row.get("program") not in (None, "") and row.get("answer") not in (None, ""):
        return row, None
    for key in [row.get("id"), row.get("selection_key"), row.get("question")]:
        gold = gold_lookup.get(normalize_lookup_key(key))
        if not gold:
            continue
        item = dict(row)
        if item.get("id") in (None, "") and gold.get("id"):
            item["id"] = gold.get("id")
        if item.get("program") in (None, ""):
            item["program"] = gold.get("program")
        if item.get("answer") in (None, ""):
            item["answer"] = gold.get("answer")
        return item, str(gold.get("gold_source_json") or "")
    return row, None


PROMPT_TYPES = {
    "o": "original",
    "z": "zero-shot",
    "m": "many-shot",
    "d": "dynamic-shot",
}


def infer_case(path: Path) -> tuple[str | None, str | None, str | None]:
    case_name = path.parent.name
    match = re.match(r"(.+)_((?:finqa|apollo).*)$", case_name)
    if not match:
        return None, None, None
    retriever_id, dataset = match.group(1), match.group(2)
    suffix = retriever_id.rsplit("_", 1)[-1]
    return retriever_id, dataset, PROMPT_TYPES.get(suffix)


def load_rows(paths: list[Path], gold_lookup: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    gold_lookup = gold_lookup or {}
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        retriever_id, dataset, prompt_type = infer_case(path)
        for index, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            item, gold_source = backfill_gold(dict(row), gold_lookup)
            expected_python = format_example_program_for_prompt(item.get("program"), item.get("answer"))
            prompt_text = build_prompt(item)
            item["stress_sample"] = {
                "source_input_json": str(path),
                "source_index": index,
                "retriever_id": retriever_id,
                "dataset": dataset,
                "prompt_type": prompt_type,
                "prompt_chars": len(prompt_text),
                "expected_python_chars": len(expected_python),
                "expected_python_preview": expected_python[:1000],
                "gold_backfill_source": gold_source,
                "selection_reasons": [],
            }
            rows.append(item)
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    meta = row.get("stress_sample") or {}
    return (
        str(meta.get("source_input_json") or ""),
        str(row.get("selection_key") or row.get("id") or row.get("question") or ""),
        str(meta.get("retriever_id") or ""),
        int(meta.get("source_index") or 0),
    )


def add_selected(
    selected: list[dict[str, Any]],
    seen: set[tuple[str, str, str, int]],
    row: dict[str, Any],
    reason: str,
    limit: int,
) -> None:
    if len(selected) >= limit:
        return
    key = row_key(row)
    if key in seen:
        for selected_row in selected:
            if row_key(selected_row) == key:
                meta = selected_row.setdefault("stress_sample", {})
                reasons = meta.setdefault("selection_reasons", [])
                if reason not in reasons:
                    reasons.append(reason)
                break
        return
    item = dict(row)
    item["stress_sample"] = dict(row.get("stress_sample") or {})
    item["stress_sample"]["selection_reasons"] = [reason]
    selected.append(item)
    seen.add(key)


def metric(row: dict[str, Any], name: str) -> int:
    return int((row.get("stress_sample") or {}).get(name) or 0)


def select_stress_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive for stress sample selection")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()

    baseline = next((row for row in rows if row.get("id") == "ETR/2016/page_23.pdf-2"), None)
    if baseline is not None:
        add_selected(selected, seen, baseline, "baseline_etr_expected_94", limit)

    by_case: dict[tuple[str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("stress_sample") or {}
        by_case[(meta.get("dataset"), meta.get("prompt_type"))].append(row)

    for case, case_rows in sorted(by_case.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        if not case_rows:
            continue
        add_selected(selected, seen, max(case_rows, key=lambda row: metric(row, "prompt_chars")), f"case_longest_prompt:{case[0]}:{case[1]}", limit)
        add_selected(selected, seen, max(case_rows, key=lambda row: metric(row, "expected_python_chars")), f"case_longest_expected_python:{case[0]}:{case[1]}", limit)

    retfact_groups: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("stress_sample") or {}
        stable_id = str(row.get("selection_key") or row.get("id") or row.get("question") or "")
        retfact_groups[(meta.get("dataset"), stable_id)].append(row)
    ranked_groups = []
    for key, group_rows in retfact_groups.items():
        prompt_types = {((row.get("stress_sample") or {}).get("prompt_type")) for row in group_rows}
        if len(prompt_types) < 2:
            continue
        lengths = [metric(row, "prompt_chars") for row in group_rows]
        ranked_groups.append((max(lengths) - min(lengths), key, group_rows))
    for _, key, group_rows in sorted(ranked_groups, key=lambda item: item[0], reverse=True):
        for row in sorted(group_rows, key=lambda item: (str((item.get("stress_sample") or {}).get("prompt_type")), -metric(item, "prompt_chars"))):
            add_selected(selected, seen, row, f"retfact_prompt_type_comparison:{key[0]}:{key[1]}", limit)
            if len(selected) >= limit:
                return selected
        if len(selected) >= limit:
            return selected

    for row in sorted(rows, key=lambda item: (metric(item, "prompt_chars") + 4 * metric(item, "expected_python_chars")), reverse=True):
        add_selected(selected, seen, row, "global_long_prompt_or_expected_python", limit)
        if len(selected) >= limit:
            break
    return selected


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("stress_sample") or {}
    return {
        "id": row.get("id"),
        "selection_key": row.get("selection_key"),
        "dataset": meta.get("dataset"),
        "retriever_id": meta.get("retriever_id"),
        "prompt_type": meta.get("prompt_type"),
        "source_index": meta.get("source_index"),
        "prompt_chars": meta.get("prompt_chars"),
        "expected_python_chars": meta.get("expected_python_chars"),
        "selection_reasons": meta.get("selection_reasons"),
        "source_input_json": meta.get("source_input_json"),
        "gold_backfill_source": meta.get("gold_backfill_source"),
    }


def write_outputs(rows: list[dict[str, Any]], selected: list[dict[str, Any]], output_json: Path, report_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    coverage = defaultdict(int)
    for row in selected:
        meta = row.get("stress_sample") or {}
        coverage[f"{meta.get('dataset')}:{meta.get('prompt_type')}"] += 1
    report = {
        "strategy": "stress_first25_long_prompt_expected_python_retfact_comparison",
        "pool_rows": len(rows),
        "selected_rows": len(selected),
        "coverage": dict(sorted(coverage.items())),
        "selected": [compact_row(row) for row in selected],
        "top_prompt_chars": [compact_row(row) for row in sorted(rows, key=lambda item: metric(item, "prompt_chars"), reverse=True)[:10]],
        "top_expected_python_chars": [compact_row(row) for row in sorted(rows, key=lambda item: metric(item, "expected_python_chars"), reverse=True)[:10]],
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expand_inputs(raw_paths: list[str], raw_globs: list[str]) -> list[Path]:
    paths = [Path(item) for item in raw_paths]
    for pattern in raw_globs:
        paths.extend(Path(item) for item in glob.glob(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        unique.append(path)
        seen.add(resolved)
    if not unique:
        raise ValueError("No input JSON files provided")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Experiment 7 stress first-25 rows from generator inputs.")
    parser.add_argument("--input-json", action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--gold-json", action="append", default=[], help="Optional FinQA gold JSON files used to backfill missing program/answer fields by id or question.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    paths = expand_inputs(args.input_json, args.input_glob)
    gold_lookup = load_gold_lookup([Path(item) for item in args.gold_json]) if args.gold_json else {}
    rows = load_rows(paths, gold_lookup=gold_lookup)
    selected = select_stress_rows(rows, args.limit)
    write_outputs(rows, selected, args.output_json, args.report_json)
    print(json.dumps({"status": "ok", "pool_rows": len(rows), "selected_rows": len(selected), "report_json": str(args.report_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
