#!/usr/bin/env python3
"""Build Experiment 6 formal prediction JSONL from narratives1 CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from binding_extraction import extract_result_items, item_dict

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
ROUTE_CSV_PATHS = {
    "narrative_original": WORKSPACE_ROOT / "data" / "finqa_original" / "narratives1_rel_fact_instruction.csv",
    "narrative_zero_shot": WORKSPACE_ROOT / "data" / "finqa_zero_shot" / "narratives1_rel_fact_instruction.csv",
    "narrative_many_shot": WORKSPACE_ROOT / "data" / "finqa_many_shot" / "narratives1_rel_fact_instruction.csv",
    "narrative_dynamic_shot": WORKSPACE_ROOT / "data" / "finqa_dynamic_shot" / "narratives1_rel_fact_instruction.csv",
}
ROUTE_GOLD_PATHS = {
    "narrative_original": WORKSPACE_ROOT / "data" / "finqa_original" / "narratives_gold.jsonl",
    "narrative_zero_shot": WORKSPACE_ROOT / "data" / "finqa_zero_shot" / "narratives_gold.jsonl",
    "narrative_many_shot": WORKSPACE_ROOT / "data" / "finqa_many_shot" / "narratives_gold.jsonl",
    "narrative_dynamic_shot": WORKSPACE_ROOT / "data" / "finqa_dynamic_shot" / "narratives_gold.jsonl",
}


@dataclass(frozen=True)
class MatrixCase:
    experiment_id: str
    source_id: str
    narrative_route: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_matrix(raw: str) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for item in raw.split():
        parts = item.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid matrix item: {item}")
        cases.append(MatrixCase(parts[0], parts[1], parts[2]))
    return cases


def read_jsonl_case_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    case_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = str(payload.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"{path}:{line_number} missing case_id")
            case_ids.append(case_id)
    return case_ids


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    required = {"Source", "Binding_Result"}
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metadata_path_for(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(jsonl_path.suffix + ".metadata.json")


def route_rows_from_csv(route: str, path: Path, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_rows = read_csv_rows(path)
    if expected_rows > 0 and len(csv_rows) != expected_rows:
        raise ValueError(f"{path} expected {expected_rows} rows, found {len(csv_rows)}")

    seen_sources: set[str] = set()
    duplicate_sources: list[str] = []
    output_rows: list[dict[str, Any]] = []
    row_reports: list[dict[str, Any]] = []
    parse_error_rows: list[dict[str, Any]] = []
    none_of_predict_rows = 0
    metric_items = 0

    for row_number, row in enumerate(csv_rows, start=2):
        case_id = str(row.get("Source") or "").strip()
        if not case_id:
            case_id = f"csv_row_{row_number}"
        if case_id in seen_sources:
            duplicate_sources.append(case_id)
        seen_sources.add(case_id)

        binding_result = row.get("Binding_Result") or ""
        record = {"case_id": case_id, "result": binding_result}
        items, _records, report = extract_result_items(record, fallback_case_id=case_id, row_number=row_number, strict=False)
        clean_items = [item_dict(item) for item in items]
        output_rows.append({"case_id": case_id, "items": clean_items})
        row_reports.append(report)
        metric_items += len(clean_items)
        if not clean_items:
            none_of_predict_rows += 1
        if report.get("errors"):
            parse_error_rows.append(
                {
                    "case_id": case_id,
                    "source_row": row_number,
                    "errors": report.get("errors"),
                    "binding_result_preview": binding_result[:300],
                }
            )

    if duplicate_sources:
        raise ValueError(f"{path} has duplicate Source values: {duplicate_sources[:10]}")

    route_report = {
        "route": route,
        "source_csv": str(path),
        "rows": len(output_rows),
        "unique_sources": len(seen_sources),
        "metric_items": metric_items,
        "none_of_predict_rows": none_of_predict_rows,
        "parse_error_count": len(parse_error_rows),
        "parse_error_rows": parse_error_rows[:25],
        "row_report_sample": row_reports[:5],
    }
    return output_rows, route_report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = parse_matrix(args.matrix)
    routes = sorted({case.narrative_route for case in cases})
    unknown_routes = [route for route in routes if route not in ROUTE_CSV_PATHS]
    if unknown_routes:
        raise ValueError(f"Unsupported narrative_route values: {unknown_routes}")

    route_rows: dict[str, list[dict[str, Any]]] = {}
    route_reports: dict[str, dict[str, Any]] = {}
    for route in routes:
        rows, route_report = route_rows_from_csv(route, ROUTE_CSV_PATHS[route], args.expected_rows)
        gold_case_ids = read_jsonl_case_ids(ROUTE_GOLD_PATHS[route])
        pred_case_ids = [row["case_id"] for row in rows]
        aligned = gold_case_ids == pred_case_ids
        route_report["gold_jsonl"] = str(ROUTE_GOLD_PATHS[route])
        route_report["gold_rows"] = len(gold_case_ids)
        route_report["case_ids_aligned_with_gold"] = aligned
        if not aligned:
            route_report["first_gold_case_ids"] = gold_case_ids[:10]
            route_report["first_prediction_case_ids"] = pred_case_ids[:10]
            raise ValueError(f"{route} prediction case_id order does not match shared gold")
        route_rows[route] = rows
        route_reports[route] = route_report

    case_reports: list[dict[str, Any]] = []
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        rows = route_rows[case.narrative_route]
        pred_path = args.pred_dir / f"{case.experiment_id}.jsonl"
        write_jsonl(pred_path, rows)
        metadata = {
            "controlled_smoke": False,
            "formal_result": True,
            "prediction_source": "csv_binding_result",
            "source_csv": str(ROUTE_CSV_PATHS[case.narrative_route]),
            "prompt_mode": case.narrative_route,
            "narrative_route": case.narrative_route,
            "experiment_id": case.experiment_id,
            "source_id": case.source_id,
            "rows": len(rows),
            "case_ids_aligned_with_gold": True,
            "created_at": utc_now(),
        }
        write_json(metadata_path_for(pred_path), metadata)
        case_reports.append(
            {
                "experiment_id": case.experiment_id,
                "source_id": case.source_id,
                "narrative_route": case.narrative_route,
                "prediction_jsonl": str(pred_path),
                "metadata_json": str(metadata_path_for(pred_path)),
                "rows": len(rows),
                "source_csv": str(ROUTE_CSV_PATHS[case.narrative_route]),
            }
        )

    report = {
        "time": utc_now(),
        "kind": "experiment6_real_prediction_build",
        "status": "completed",
        "prediction_source": "csv_binding_result",
        "pred_dir": str(args.pred_dir),
        "expected_rows": args.expected_rows,
        "matrix": args.matrix.split(),
        "routes": route_reports,
        "cases": case_reports,
    }
    write_json(args.report_json or (args.pred_dir / "prediction_build_report.json"), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=85)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
