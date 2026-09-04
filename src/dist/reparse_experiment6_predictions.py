#!/usr/bin/env python3
"""Rebuild Experiment 6 prediction JSONL files from saved raw model outputs.

This is intentionally model-free: it uses the raw_output paths recorded in each
case metadata file, reapplies binding_extraction, rewrites run_XX JSONL files,
then rebuilds the top-k aggregate JSONL and metadata extraction reports.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_experiment6_binding_generation import (
    MatrixCase,
    ROUTE_CSV_PATHS,
    aggregate_top_k_rows,
    prediction_rows_from_texts,
    read_prompt_rows,
    read_raw_prediction_lines,
    write_json,
    write_prediction_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_files(pred_dir: Path, cases: set[str] | None) -> list[Path]:
    files = sorted(pred_dir.glob("*.jsonl.metadata.json"))
    if cases is None:
        return files
    return [path for path in files if path.name.removesuffix(".jsonl.metadata.json") in cases]


def runtime_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = metadata.get("runtime")
    if isinstance(runtime, list):
        return [entry for entry in runtime if isinstance(entry, dict)]
    if isinstance(runtime, dict):
        return [runtime]
    return []


def reparse_case(metadata_path: Path) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    experiment_id = str(metadata.get("experiment_id") or metadata_path.name.removesuffix(".jsonl.metadata.json"))
    source_id = str(metadata.get("source_id") or "")
    narrative_route = str(metadata.get("narrative_route") or "")
    if not metadata.get("formal_result"):
        return {"case": experiment_id, "status": "skipped_not_formal_result"}
    if narrative_route not in ROUTE_CSV_PATHS:
        return {"case": experiment_id, "status": "skipped_unknown_route", "narrative_route": narrative_route}

    run_jsonls = [Path(path) for path in metadata.get("run_prediction_jsonls") or []]
    runtimes = runtime_entries(metadata)
    if not run_jsonls or not runtimes:
        return {"case": experiment_id, "status": "skipped_missing_run_metadata"}
    if len(run_jsonls) != len(runtimes):
        return {
            "case": experiment_id,
            "status": "skipped_run_metadata_mismatch",
            "run_prediction_jsonls": len(run_jsonls),
            "runtimes": len(runtimes),
        }

    prompt_rows = read_prompt_rows(ROUTE_CSV_PATHS[narrative_route], 0)
    case = MatrixCase(experiment_id, source_id, narrative_route)
    run_rows: list[list[dict[str, Any]]] = []
    run_reports: list[dict[str, Any]] = []
    raw_outputs: list[str] = []

    for run_index, (run_jsonl, runtime) in enumerate(zip(run_jsonls, runtimes), start=1):
        raw_output = runtime.get("raw_output")
        if not raw_output:
            return {"case": experiment_id, "status": "failed_missing_raw_output", "run": run_index}
        raw_path = Path(raw_output)
        if not raw_path.is_absolute():
            raw_path = Path.cwd() / raw_path
        if not raw_path.is_file():
            return {"case": experiment_id, "status": "failed_raw_missing", "run": run_index, "raw_output": str(raw_path)}
        predictions = read_raw_prediction_lines(raw_path)
        if len(predictions) != len(prompt_rows):
            return {
                "case": experiment_id,
                "status": "failed_prediction_row_count_mismatch",
                "run": run_index,
                "predictions": len(predictions),
                "prompts": len(prompt_rows),
                "raw_output": str(raw_path),
            }
        rows, extraction_reports = prediction_rows_from_texts(case, prompt_rows, predictions)
        write_prediction_jsonl(run_jsonl, rows)
        run_rows.append(rows)
        run_reports.append({"run": run_index, "extraction_reports": extraction_reports})
        raw_outputs.append(str(raw_path))

    top_k = int(metadata.get("top_k") or 3)
    top_rows = aggregate_top_k_rows(prompt_rows, run_rows, top_k)
    top_path = Path(metadata.get("top_k_prediction_jsonl") or metadata_path.name.removesuffix(".metadata.json"))
    if not top_path.is_absolute():
        top_path = metadata_path.parent / top_path.name
    write_prediction_jsonl(top_path, top_rows)

    metadata["reparsed_at"] = utc_now()
    metadata["reparse_source"] = "raw_model_outputs"
    metadata["rows"] = len(top_rows)
    metadata["extraction_reports"] = [report for run_report in run_reports for report in run_report["extraction_reports"]]
    metadata["run_prediction_jsonls"] = [str(path) for path in run_jsonls]
    metadata["top_k_prediction_jsonl"] = str(top_path)
    write_json(metadata_path, metadata)

    return {
        "case": experiment_id,
        "status": "reparsed",
        "runs": len(run_rows),
        "top_k": top_k,
        "top_k_prediction_jsonl": str(top_path),
        "items": sum(len(row.get("items") or []) for rows in run_rows for row in rows),
        "rows_with_items": sum(bool(row.get("items")) for rows in run_rows for row in rows),
        "raw_outputs": raw_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = set(args.case) if args.case else None
    results = [reparse_case(path) for path in metadata_files(args.pred_dir, cases)]
    report = {"time": utc_now(), "pred_dir": str(args.pred_dir), "cases": results}
    if args.report_json:
        write_json(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
