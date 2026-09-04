#!/usr/bin/env python3
"""Import external GPT-4.1 results into Experiment 6 prediction artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment6_binding_generation as runner  # noqa: E402


GPT41_CASES = [
    runner.MatrixCase("6_FinFlier_gpt4.1", "gpt4_1", "narrative_original"),
    runner.MatrixCase("6_gpt4.1_z", "gpt4_1", "narrative_zero_shot"),
    runner.MatrixCase("6_gpt4.1_m", "gpt4_1", "narrative_many_shot"),
    runner.MatrixCase("6_gpt4.1_d", "gpt4_1", "narrative_dynamic_shot"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def task_id(case: runner.MatrixCase, run_number: int, row_index: int) -> str:
    return f"{case.experiment_id}:run_{run_number:02d}:row_{row_index:03d}"


def completed_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "completed" and row.get("prediction"):
            results[str(row["task_id"])] = row
    return results


def raw_skeleton(case: runner.MatrixCase, prompt_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "case_id": row.get("Source") or f"row_{index + 1}",
            "stage": "direct_binding_generation",
            "status": "not_started",
            "input_prompt": runner.direct_binding_prompt(row),
            "prediction": None,
            "error": None,
        }
        for index, row in enumerate(prompt_rows)
    ]


def merge_run(
    *,
    case: runner.MatrixCase,
    prompt_rows: list[dict[str, str]],
    run_number: int,
    pred_dir: Path,
    results: dict[str, dict[str, Any]],
    max_tokens: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    raw_path = pred_dir / "raw" / f"{case.experiment_id}.run_{run_number:02d}.jsonl"
    raw_rows = read_jsonl(raw_path)
    if len(raw_rows) != len(prompt_rows):
        raw_rows = raw_skeleton(case, prompt_rows)
    by_index = {int(row["index"]): row for row in raw_rows}
    imported = 0
    for index in range(len(prompt_rows)):
        result = results.get(task_id(case, run_number, index))
        if not result:
            continue
        current = by_index[index]
        if current.get("status") == "completed" and current.get("prediction"):
            continue
        by_index[index] = {
            **current,
            "status": "completed",
            "prediction": result["prediction"],
            "error": None,
            "external_import": {
                "imported_at": runner.utc_now(),
                "route": result.get("route"),
                "model": result.get("model"),
            },
        }
        imported += 1
    merged_rows = [by_index[index] for index in range(len(prompt_rows))]
    write_jsonl(raw_path, merged_rows)
    incomplete = [row for row in merged_rows if row.get("status") != "completed" or not row.get("prediction")]
    status = {
        "run": run_number,
        "raw_jsonl": str(raw_path),
        "imported_rows": imported,
        "completed_rows": len(merged_rows) - len(incomplete),
        "expected_rows": len(prompt_rows),
        "complete": not incomplete,
    }
    if incomplete:
        return None, status

    predictions = [str(row.get("prediction") or "") for row in merged_rows]
    output_rows, extraction_reports = runner.prediction_rows_from_texts(case, prompt_rows, predictions)
    run_pred_jsonl = runner.run_pred_path(
        argparse.Namespace(pred_dir=pred_dir, num_runs=10),
        case,
        run_number,
    )
    runner.write_prediction_jsonl(run_pred_jsonl, output_rows)
    runner.write_json(
        runner.run_metadata_path(run_pred_jsonl),
        {
            "created_at": runner.utc_now(),
            "status": "completed",
            "formal_result": True,
            "experiment_id": case.experiment_id,
            "source_id": case.source_id,
            "narrative_route": case.narrative_route,
            "prompt_csv": str(runner.ROUTE_CSV_PATHS[case.narrative_route]),
            "run": run_number,
            "rows": len(output_rows),
            "runtime": {
                "stage": "direct_binding_generation",
                "engine": "gpt4_1",
                "actual_engine": "gpt4_1",
                "prediction_contract": "data_binding_generator",
                "raw_output": str(raw_path),
                "system_prompt": "experiment6_binding_json_v1",
                "max_tokens": max_tokens,
                "parallelism": 1,
                "external_import": True,
            },
            "extraction_reports": extraction_reports,
        },
    )
    status["run_prediction_jsonl"] = str(run_pred_jsonl)
    return output_rows, status


def run_evaluation(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.update(
        {
            "EXPT_ID": args.expt_id,
            "EXPERIMENT6_NUM_RUNS": str(args.num_runs),
            "EXPERIMENT6_TOP_K": str(args.top_k),
            "EXPERIMENT6_GENERATE_BINDING_PREDICTIONS": "0",
            "EXPERIMENT6_PREPARE_PROMPT_DATA": "0",
            "EXPERIMENT6_BINDING_MATRIX": " ".join(f"{c.experiment_id}:{c.source_id}:{c.narrative_route}" for c in GPT41_CASES),
            "NARRATIVE_PRED_DIR": str(REPO_ROOT / "Experiment" / args.expt_id / "binding_eval_predictions"),
            "STRICT_INPUTS": "1",
        }
    )
    completed = subprocess.run(
        ["bash", "dist/experiment_6_api_key.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    log_path = REPO_ROOT / "Experiment" / args.expt_id / "external_gpt41_import_evaluation.log"
    log_path.write_text(
        f"returncode={completed.returncode}\n\n[stdout]\n{completed.stdout}\n\n[stderr]\n{completed.stderr}",
        encoding="utf-8",
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expt-id", default="experiment_6_api_key_gpt41_formal_20260614T1800Z")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = REPO_ROOT / "Experiment" / args.expt_id / "binding_eval_predictions"
    results = completed_results(args.results_jsonl)
    report: dict[str, Any] = {
        "time": runner.utc_now(),
        "expt_id": args.expt_id,
        "results_jsonl": str(args.results_jsonl),
        "completed_result_tasks": len(results),
        "cases": [],
    }
    all_cases_complete = True
    for case in GPT41_CASES:
        prompt_rows = runner.read_prompt_rows(runner.ROUTE_CSV_PATHS[case.narrative_route], args.limit)
        run_rows: list[list[dict[str, Any]]] = []
        run_prediction_jsonls: list[str] = []
        case_status: dict[str, Any] = {"experiment_id": case.experiment_id, "runs": []}
        for run_number in range(1, args.num_runs + 1):
            rows, status = merge_run(
                case=case,
                prompt_rows=prompt_rows,
                run_number=run_number,
                pred_dir=pred_dir,
                results=results,
                max_tokens=args.max_tokens,
            )
            case_status["runs"].append(status)
            if rows is None:
                all_cases_complete = False
                continue
            run_rows.append(rows)
            run_prediction_jsonls.append(str(runner.run_pred_path(argparse.Namespace(pred_dir=pred_dir, num_runs=args.num_runs), case, run_number)))
        case_status["complete"] = len(run_rows) == args.num_runs
        if case_status["complete"]:
            top_rows = runner.aggregate_top_k_rows(prompt_rows, run_rows, args.top_k)
            top_path = pred_dir / f"{case.experiment_id}.jsonl"
            runner.write_prediction_jsonl(top_path, top_rows)
            runner.write_json(
                top_path.with_suffix(top_path.suffix + ".metadata.json"),
                {
                    "created_at": runner.utc_now(),
                    "formal_result": True,
                    "controlled_smoke": False,
                    "prediction_source": "external_model_generated_from_input_prompt",
                    "generation_mode": "no-adapter",
                    "experiment_id": case.experiment_id,
                    "source_id": case.source_id,
                    "narrative_route": case.narrative_route,
                    "prompt_csv": str(runner.ROUTE_CSV_PATHS[case.narrative_route]),
                    "rows": len(top_rows),
                    "num_runs": args.num_runs,
                    "top_k": args.top_k,
                    "top_k_prediction_jsonl": str(top_path),
                    "run_prediction_jsonls": run_prediction_jsonls,
                    "runtime": {
                        "stage": "direct_binding_generation",
                        "engine": "gpt4_1",
                        "actual_engine": "gpt4_1",
                        "prediction_contract": "data_binding_generator",
                        "parallelism": 1,
                        "external_import": True,
                    },
                    "resume_runs": True,
                },
            )
            case_status["top_k_prediction_jsonl"] = str(top_path)
        report["cases"].append(case_status)

    if all_cases_complete and not args.skip_evaluation:
        report["evaluation_returncode"] = run_evaluation(args)
    else:
        report["evaluation_returncode"] = None
        report["evaluation_skipped_reason"] = "incomplete_cases" if not all_cases_complete else "skip_evaluation"
    report_path = REPO_ROOT / "Experiment" / args.expt_id / "external_gpt41_import_report.json"
    runner.write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
