#!/usr/bin/env python3
"""Import Experiment 6 BTC converter predictions into FinFlier B/C payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import build_btc_finflier_demo as demo
import run_btc_demo_model_predictions as runner


DEFAULT_FINFLIER_ROOT = REPO_ROOT / "FinFlier"
DEFAULT_PRED_DIR = REPO_ROOT / "Experiment" / "btc_finflier_custom" / "binding_eval_predictions"
FOLDER_TO_EXPERIMENT = {
    "B": "btc_flan_m",
    "C": "btc_mistral_m",
}


def read_converter_rows(metadata: dict[str, Any]) -> list[list[dict[str, Any]]]:
    runtimes = metadata.get("runtime")
    if isinstance(runtimes, dict):
        runtimes = [runtimes]
    if not isinstance(runtimes, list):
        return []
    attempts: list[list[dict[str, Any]]] = []
    for runtime in runtimes:
        conversion = runtime.get("binding_conversion") if isinstance(runtime, dict) else None
        raw_path = conversion.get("converter_raw_output") if isinstance(conversion, dict) else None
        if not raw_path:
            continue
        rows: list[dict[str, Any]] = []
        path = Path(raw_path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(payload)
        attempts.append(rows)
    return attempts


def import_folder(folder: str, finflier_root: Path, pred_dir: Path) -> dict[str, Any]:
    experiment_id = FOLDER_TO_EXPERIMENT[folder]
    metadata_path = pred_dir / f"{experiment_id}.jsonl.metadata.json"
    payload_path = finflier_root / folder / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not metadata_path.is_file():
        payload["runtime_status"] = {
            "status": "runtime_blocked",
            "reason": f"Experiment 6 metadata not found: {metadata_path}",
            "route_status": "missing_experiment6_metadata",
            "missing": [str(metadata_path)],
        }
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"folder": folder, "status": "runtime_blocked", "missing": str(metadata_path)}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    attempts_by_run = read_converter_rows(metadata)
    predictions_by_case: list[list[dict[str, Any]]] = []
    for case_index, case in enumerate(payload.get("cases", [])):
        best_prediction: list[dict[str, Any]] = []
        best_comparison = demo.compare_binding_fields(case["gold_binding"], best_prediction)
        best_score = runner.comparison_score(best_comparison)
        best_attempt = None
        attempts = []
        for run_index, rows in enumerate(attempts_by_run, start=1):
            raw_row = rows[case_index] if case_index < len(rows) else {}
            raw_output = raw_row.get("prediction") or ""
            prediction = runner.parse_binding_output(raw_output)
            comparison = demo.compare_binding_fields(case["gold_binding"], prediction)
            score = runner.comparison_score(comparison)
            attempts.append(
                {
                    "attempt": run_index,
                    "status": raw_row.get("status", "completed"),
                    "raw_output": raw_output,
                    "retriever_prediction": raw_row.get("retriever_prediction"),
                    "parsed_prediction": prediction,
                    "comparison": comparison,
                    "field_match_score": score,
                }
            )
            if score["matched_fields"] > best_score["matched_fields"] or (
                score["matched_fields"] == best_score["matched_fields"] and len(prediction) > len(best_prediction)
            ):
                best_prediction = prediction
                best_comparison = comparison
                best_score = score
                best_attempt = run_index
        case["prediction_attempts"] = attempts
        case["model_prediction"] = best_prediction
        case["model_comparison"] = best_comparison
        case["best_attempt"] = best_attempt
        case["field_match_score"] = best_score
        predictions_by_case.append(best_prediction)
    comparison = demo.compare_cases(payload.get("cases", []), predictions_by_case)
    payload["comparison"] = comparison
    if metadata.get("status") in {"completed", "resumed_completed"} or metadata.get("formal_result"):
        status = "completed_exact_match" if comparison["exact_match"] else "completed_mismatch"
        reason = "Experiment 6 retriever-to-binding conversion completed; best closest attempt was kept."
    else:
        status = "runtime_blocked"
        reason = f"Experiment 6 status={metadata.get('status')} failure={metadata.get('failure_category')}"
    payload["runtime_status"] = {
        "status": status,
        "reason": reason,
        "route_status": metadata.get("status"),
        "missing": [],
        "max_attempts": len(attempts_by_run),
        "experiment6_metadata": str(metadata_path),
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (finflier_root / folder / "index.html").write_text(demo.render_html(payload), encoding="utf-8")
    return {
        "folder": folder,
        "status": status,
        "exact_match": comparison["exact_match"],
        "best_attempts": [case.get("best_attempt") for case in payload.get("cases", [])],
        "field_match_scores": [case.get("field_match_score") for case in payload.get("cases", [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finflier-root", type=Path, default=DEFAULT_FINFLIER_ROOT)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--folders", nargs="*", default=["B", "C"])
    args = parser.parse_args()
    report = [import_folder(folder, args.finflier_root, args.pred_dir) for folder in args.folders]
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
