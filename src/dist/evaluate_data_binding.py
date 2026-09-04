#!/usr/bin/env python3
"""Evaluate Experiment 6 data binding with FinFlier vocabulary micro-F1.

The unit of counting is a vocabulary item, not a row and not a complete JSON
binding object. For each narrative row, extract three vocabulary sets from
Binding items, then micro-count TP/FP/FN across all rows:

- subject: ObjectName
- trend: Trend, excluding None/null
- numerical: Num, excluding None/null
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from binding_extraction import BindingItem, extract_result_items, item_dict

FIELD_NAMES = {
    "subject": "ObjectName",
    "trend": "Trend",
    "numerical": "Num",
}
TYPE_NAMES = tuple(FIELD_NAMES)
SCORE_POLICY = {
    "empty_set_score": 1.0,
    "runtime_error_in_official_average": "excluded",
    "runtime_error_in_penalized_average": 0.0,
    "case_alignment_required": True,
    "duplicate_case_ids": "invalid_binding_eval_input",
}


@dataclass(frozen=True)
class ExtractedRow:
    case_id: str
    sets: dict[str, set[str]]
    items: list[BindingItem]
    records: list[dict[str, Any]]
    report: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} invalid JSONL: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} row must be a JSON object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def case_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("case_id", "CaseId", "Source", "id", "uid", "question_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def items_to_sets(items: Iterable[BindingItem]) -> dict[str, set[str]]:
    sets = {kind: set() for kind in TYPE_NAMES}
    for item in items:
        if item.kind in sets and item.text:
            sets[item.kind].add(item.text)
    return sets


def extract_rows(rows: list[dict[str, Any]], strict: bool) -> list[ExtractedRow]:
    extracted: list[ExtractedRow] = []
    for row_number, row in enumerate(rows, start=1):
        cid = case_id(row, f"row_{row_number}")
        items, records, report = extract_result_items(row, fallback_case_id=cid, row_number=row_number, strict=strict)
        extracted.append(
            ExtractedRow(
                case_id=cid,
                sets=items_to_sets(items),
                items=items,
                records=records,
                report=report,
            )
        )
    return extracted


def empty_counts() -> dict[str, dict[str, int]]:
    return {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in TYPE_NAMES}


def score_from_counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def duplicate_case_ids(rows: list[ExtractedRow]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row.case_id in seen and row.case_id not in duplicates:
            duplicates.append(row.case_id)
        seen.add(row.case_id)
    return duplicates


def validate_case_alignment(gold: list[ExtractedRow], pred: list[ExtractedRow], require_data: bool) -> list[str]:
    blockers: list[str] = []
    if require_data and not gold:
        blockers.append("gold JSONL is empty")
    if require_data and not pred:
        blockers.append("prediction JSONL is empty")

    gold_duplicates = duplicate_case_ids(gold)
    pred_duplicates = duplicate_case_ids(pred)
    if gold_duplicates:
        blockers.append(f"gold JSONL has duplicate case_id values: {gold_duplicates[:5]}")
    if pred_duplicates:
        blockers.append(f"prediction JSONL has duplicate case_id values: {pred_duplicates[:5]}")

    if gold and pred and len(gold) != len(pred):
        blockers.append(f"gold/pred row count mismatch: gold={len(gold)} pred={len(pred)}")

    if gold and pred and [row.case_id for row in gold] != [row.case_id for row in pred]:
        mismatch = next(
            (
                index
                for index, (gold_row, pred_row) in enumerate(zip(gold, pred), start=1)
                if gold_row.case_id != pred_row.case_id
            ),
            None,
        )
        if mismatch is None:
            mismatch = min(len(gold), len(pred)) + 1
        blockers.append(f"gold/pred case_id order mismatch at row {mismatch}")
    return blockers


def metrics_from_extracted(gold: list[ExtractedRow], pred: list[ExtractedRow], vocabulary_types: list[str]) -> dict[str, Any]:
    pred_by_case = {row.case_id: row for row in pred}
    counts = empty_counts()
    for gold_row in gold:
        pred_row = pred_by_case.get(gold_row.case_id)
        for kind in vocabulary_types:
            gold_set = gold_row.sets.get(kind, set())
            pred_set = pred_row.sets.get(kind, set()) if pred_row else set()
            counts[kind]["tp"] += len(gold_set & pred_set)
            counts[kind]["fp"] += len(pred_set - gold_set)
            counts[kind]["fn"] += len(gold_set - pred_set)

    by_type = {kind: score_from_counts(**counts[kind]) for kind in vocabulary_types}
    by_field = {FIELD_NAMES[kind]: dict(by_type[kind]) for kind in vocabulary_types}
    total_tp = sum(counts[kind]["tp"] for kind in vocabulary_types)
    total_fp = sum(counts[kind]["fp"] for kind in vocabulary_types)
    total_fn = sum(counts[kind]["fn"] for kind in vocabulary_types)
    overall = score_from_counts(total_tp, total_fp, total_fn)
    return {
        **overall,
        "by_type": by_type,
        "by_field": by_field,
        "field_contract": [FIELD_NAMES[kind] for kind in vocabulary_types],
        "counting_unit": "vocabulary_item_micro_across_rows",
        "score_policy": SCORE_POLICY,
    }


def extracted_jsonl_rows(rows: list[ExtractedRow]) -> list[dict[str, Any]]:
    return [{"case_id": row.case_id, "items": [item_dict(item) for item in row.items]} for row in rows]


def extraction_report(gold: list[ExtractedRow], pred: list[ExtractedRow], metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "gold": [row.report for row in gold],
        "pred": [row.report for row in pred],
        "gold_records": [record for row in gold for record in row.records],
        "pred_records": [record for row in pred for record in row.records],
        "prediction_metadata": metadata or {},
    }


def load_metadata(pred_jsonl: Path) -> dict[str, Any] | None:
    path = pred_jsonl.with_suffix(pred_jsonl.suffix + ".metadata.json")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"metadata_parse_error": str(path)}


def aligned(gold: list[ExtractedRow], pred: list[ExtractedRow]) -> bool:
    return [row.case_id for row in gold] == [row.case_id for row in pred]


def average(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def run_aggregate_metrics(
    metadata: dict[str, Any] | None,
    gold: list[ExtractedRow],
    vocabulary_types: list[str],
) -> dict[str, Any] | None:
    if not metadata:
        return None
    run_paths = [Path(path) for path in metadata.get("run_prediction_jsonls", [])]
    if not run_paths:
        return None

    runs: list[dict[str, Any]] = []
    for index, path in enumerate(run_paths, start=1):
        try:
            pred = extract_rows(read_jsonl(path), strict=False)
            blockers = validate_case_alignment(gold, pred, require_data=True)
            if blockers:
                raise ValueError("; ".join(blockers))
            metrics = metrics_from_extracted(gold, pred, vocabulary_types)
            runs.append({"run": index, "pred_jsonl": str(path), "status": "completed", "metrics": metrics})
        except Exception as exc:
            runs.append(
                {
                    "run": index,
                    "pred_jsonl": str(path),
                    "status": "run_error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "metrics": None,
                }
            )

    completed_runs = [run for run in runs if run.get("status") == "completed"]
    run_errors = [run for run in runs if run.get("status") == "run_error"]

    def metric_value(run: dict[str, Any], metric: str, kind: str | None = None) -> float:
        if run.get("status") != "completed" or not run.get("metrics"):
            return 0.0
        metrics = run["metrics"]
        if kind is not None:
            return float(metrics["by_type"][kind][metric])
        return float(metrics[metric])

    def average_metrics(runs_to_average: list[dict[str, Any]], penalize_errors: bool = False) -> dict[str, Any]:
        denominator_runs = runs if penalize_errors else runs_to_average
        by_type_average: dict[str, dict[str, float | None]] = {}
        by_field_average: dict[str, dict[str, float | None]] = {}
        for kind in vocabulary_types:
            field = FIELD_NAMES[kind]
            by_type_average[kind] = {
                metric: average([metric_value(run, metric, kind) for run in denominator_runs])
                for metric in ("precision", "recall", "f1")
            }
            by_field_average[field] = dict(by_type_average[kind])
        return {
            "precision": average([metric_value(run, "precision") for run in denominator_runs]),
            "recall": average([metric_value(run, "recall") for run in denominator_runs]),
            "f1": average([metric_value(run, "f1") for run in denominator_runs]),
            "by_type": by_type_average,
            "by_field": by_field_average,
            "denominator_runs": len(denominator_runs),
        }

    def sort_key(run: dict[str, Any]) -> tuple[float, float, float, int]:
        metrics = run["metrics"]
        return (float(metrics["f1"]), float(metrics["precision"]), float(metrics["recall"]), -int(run["run"]))

    top_k = int(metadata.get("top_k") or 3)
    top_runs = sorted(completed_runs, key=sort_key, reverse=True)[: max(1, min(top_k, len(completed_runs)))] if completed_runs else []
    official_average = average_metrics(completed_runs)
    penalized_average = average_metrics(runs, penalize_errors=True)
    return {
        "num_runs": len(runs),
        "completed_runs": len(completed_runs),
        "run_errors": len(run_errors),
        "top_k": metadata.get("top_k"),
        "top_k_prediction_jsonl": metadata.get("top_k_prediction_jsonl"),
        "average": official_average,
        "official_average": official_average,
        "penalized_average": penalized_average,
        "top_3_best_runs": {
            "selection_metric": "overall_f1",
            "top_k": top_k,
            "average": average_metrics(top_runs),
            "runs": top_runs,
        },
        "runs": runs,
        "error_runs": run_errors,
        "score_policy": SCORE_POLICY,
    }

def parse_vocabulary_types(raw: list[str]) -> list[str]:
    values: list[str] = []
    for value in raw:
        key = value.strip().lower()
        if key in FIELD_NAMES and key not in values:
            values.append(key)
    return values or list(TYPE_NAMES)


def build_status(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    metadata = load_metadata(args.pred_jsonl)
    blockers: list[str] = []
    failure_category = None
    gold_rows_raw: list[dict[str, Any]] = []
    pred_rows_raw: list[dict[str, Any]] = []
    gold: list[ExtractedRow] = []
    pred: list[ExtractedRow] = []
    metrics: dict[str, Any] | None = None

    if metadata and metadata.get("metadata_parse_error"):
        blockers.append(f"prediction metadata is invalid JSON: {metadata['metadata_parse_error']}")
        failure_category = "invalid_binding_eval_input"
    if metadata and metadata.get("runtime_blocked"):
        blockers.append("prediction JSONL was not produced by a completed model run")
        if metadata.get("error"):
            blockers.append(str(metadata["error"]))
        failure_category = metadata.get("failure_category") or "blocked_model_generation_runtime"
    if metadata and not args.allow_controlled_predictions and (
        metadata.get("controlled_prediction") or metadata.get("controlled_smoke")
    ):
        blockers.append("controlled prediction scaffold cannot be scored as a formal Experiment 6 result")
        failure_category = failure_category or "blocked_controlled_prediction"

    try:
        gold_rows_raw = read_jsonl(args.gold_jsonl)
    except FileNotFoundError as exc:
        blockers.append(f"gold JSONL missing: {exc}")
        failure_category = failure_category or "blocked_missing_narrative_dataset"
    except Exception as exc:
        blockers.append(f"gold JSONL cannot be read: {exc}")
        failure_category = failure_category or "invalid_binding_eval_input"

    try:
        pred_rows_raw = read_jsonl(args.pred_jsonl)
    except FileNotFoundError as exc:
        blockers.append(f"prediction JSONL missing: {exc}")
        failure_category = failure_category or "blocked_missing_narrative_dataset"
    except Exception as exc:
        blockers.append(f"prediction JSONL cannot be read: {exc}")
        failure_category = failure_category or "invalid_binding_eval_input"

    vocabulary_types = parse_vocabulary_types(args.vocabulary_types)
    if gold_rows_raw:
        try:
            gold = extract_rows(gold_rows_raw, strict=True)
        except Exception as exc:
            blockers.append(f"gold result extraction failed: {exc}")
            failure_category = failure_category or "invalid_binding_eval_input"
    if pred_rows_raw:
        try:
            pred = extract_rows(pred_rows_raw, strict=False)
        except Exception as exc:
            blockers.append(f"prediction result extraction failed: {exc}")
            failure_category = failure_category or "invalid_binding_eval_input"

    alignment_blockers = validate_case_alignment(gold, pred, require_data=args.require_data)
    if alignment_blockers:
        blockers.extend(alignment_blockers)
        failure_category = failure_category or "invalid_binding_eval_input"

    if not blockers:
        metrics = metrics_from_extracted(gold, pred, vocabulary_types)
        runs = run_aggregate_metrics(metadata, gold, vocabulary_types)
        if runs is not None:
            metrics["run_aggregate"] = runs

    metrics_payload: dict[str, Any]
    if metrics is None:
        metrics_payload = {
            "status": "not_scored",
            "failure_category": failure_category or "invalid_binding_eval_input",
            "blockers": blockers,
            "score_policy": SCORE_POLICY,
        }
    else:
        metrics_payload = metrics

    write_json(args.metrics_json, metrics_payload)
    write_jsonl(args.metrics_json.parent / "gold_extracted.jsonl", extracted_jsonl_rows(gold))
    write_jsonl(args.metrics_json.parent / "pred_extracted.jsonl", extracted_jsonl_rows(pred))
    write_json(args.metrics_json.parent / "extraction_report.json", extraction_report(gold, pred, metadata))

    status = "runtime_blocked" if blockers else "completed"
    if blockers and failure_category is None:
        failure_category = "invalid_binding_eval_input"
    payload = {
        "time": utc_now(),
        "experiment": "6",
        "experiment_id": args.experiment_id,
        "stage": "data_binding_evaluation",
        "source_id": args.source_id,
        "narrative_route": args.narrative_route,
        "status": status,
        "failure_category": failure_category,
        "metrics": metrics,
        "score_policy": SCORE_POLICY,
        "blockers": blockers,
        "next_step": (
            "Resolve the data/runtime blocker and rerun Experiment 6; do not substitute missing or invalid inputs as scores."
            if blockers
            else "Use metrics_json and extracted JSONL artifacts for Experiment 6 reporting."
        ),
        "artifacts": {
            "gold_jsonl": str(args.gold_jsonl),
            "pred_jsonl": str(args.pred_jsonl),
            "metrics_json": str(args.metrics_json),
            "status_json": str(args.status_json),
            "gold_extracted_jsonl": str(args.metrics_json.parent / "gold_extracted.jsonl"),
            "pred_extracted_jsonl": str(args.metrics_json.parent / "pred_extracted.jsonl"),
            "extraction_report_json": str(args.metrics_json.parent / "extraction_report.json"),
        },
        "rows": {"gold": len(gold_rows_raw), "pred": len(pred_rows_raw)},
        "preflight": {
            "gold_rows": len(gold_rows_raw),
            "pred_rows": len(pred_rows_raw),
            "case_ids_aligned": aligned(gold, pred),
            "case_ids_unique": not duplicate_case_ids(gold) and not duplicate_case_ids(pred),
            "prediction_metadata": metadata or {},
            "controlled_prediction": bool(metadata and metadata.get("controlled_prediction")),
            "runtime_blocked_prediction": bool(metadata and metadata.get("runtime_blocked")),
        },
        "controlled_predictions_allowed": args.allow_controlled_predictions,
    }
    write_json(args.status_json, payload)
    return payload, 2 if blockers else 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--narrative-route", required=True)
    parser.add_argument("--gold-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--vocabulary-types", nargs="+", default=list(TYPE_NAMES))
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--allow-controlled-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    payload, exit_code = build_status(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
