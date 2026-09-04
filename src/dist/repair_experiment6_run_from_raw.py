#!/usr/bin/env python3
"""Repair a partially completed Experiment 6 run from raw model output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment6_binding_generation as runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--narrative-route", required=True)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--run-pred-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--engine", default="gpt5_5")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--row-timeout-seconds", type=int, default=900)
    parser.add_argument("--retry-max", type=int, default=0)
    parser.add_argument("--retry-wait-seconds", type=int, default=600)
    parser.add_argument("--request-spacing-seconds", type=int, default=0)
    parser.add_argument("--load-variables-md", type=int, default=1)
    parser.add_argument("--variables-md", type=Path, default=runner.WORKSPACE_ROOT / "src" / "doc" / "workspace" / "variables.md")
    return parser.parse_args()


def write_raw_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.load_variables_md = bool(args.load_variables_md)
    variables_md_status = runner.load_workspace_variables_md(args)
    case = runner.MatrixCase(args.experiment_id, args.source_id, args.narrative_route)
    prompt_rows = runner.read_prompt_rows(runner.ROUTE_CSV_PATHS[args.narrative_route], args.limit)
    if args.raw_jsonl.is_file():
        raw_rows = [
            json.loads(line)
            for line in args.raw_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        args.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = [
            {
                "index": index,
                "case_id": row.get("case_id") or row.get("id") or str(index),
                "stage": "direct_binding_generation",
                "status": "not_started",
                "input_prompt": runner.direct_binding_prompt(row),
                "prediction": None,
                "error": None,
            }
            for index, row in enumerate(prompt_rows)
        ]
        write_raw_rows(args.raw_jsonl, raw_rows)
    if len(raw_rows) != len(prompt_rows):
        raise SystemExit(f"raw row count mismatch: {len(raw_rows)} != {len(prompt_rows)}")

    generator_runtime, config, route_status = runner.resolve_generator_runtime(args.engine)
    fixed_rows = []
    repaired_indexes = []
    wait_seconds = runner.rounded_wait_seconds(max(0, args.retry_wait_seconds))
    attempts = max(0, args.retry_max) + 1
    for row_position, raw in enumerate(raw_rows):
        if raw.get("status") == "completed" and raw.get("prediction"):
            fixed_rows.append(raw)
            continue
        index = int(raw["index"])
        prompt = runner.direct_binding_prompt(prompt_rows[index])
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                outputs = runner.generate_text_with_timeout(
                    generator_runtime,
                    config,
                    prompt,
                    runner.BINDING_JSON_SYSTEM_PROMPT,
                    args.max_tokens,
                    args.row_timeout_seconds,
                )
                repaired_raw = {
                    **raw,
                    "status": "completed",
                    "prediction": outputs[0] if outputs else "",
                    "error": None,
                    "repair": {
                        "created_at": runner.utc_now(),
                        "method": "row_level_timeout_repair",
                        "row_timeout_seconds": args.row_timeout_seconds,
                        "attempt": attempt,
                    },
                }
                fixed_rows.append(repaired_raw)
                raw_rows[row_position] = repaired_raw
                write_raw_rows(args.raw_jsonl, raw_rows)
                break
            except Exception as exc:  # noqa: BLE001 - keep row-level resume artifact on all failures
                last_error = exc
                transient = runner.is_transient_error(exc)
                if not transient or attempt >= attempts:
                    raw_rows[row_position] = {
                        **raw,
                        "status": "runtime_blocked",
                        "error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "transient": transient,
                            "attempt": attempt,
                        },
                    }
                    write_raw_rows(args.raw_jsonl, raw_rows)
                    raise
                print(
                    json.dumps(
                        {
                            "time": runner.utc_now(),
                            "status": "repair_retry_wait",
                            "experiment_id": case.experiment_id,
                            "raw_jsonl": str(args.raw_jsonl),
                            "row_index": index,
                            "attempt": attempt,
                            "wait_seconds": wait_seconds,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
        else:
            raise RuntimeError(f"unreachable repair retry loop: {last_error}")
        repaired_indexes.append(index)
        if args.request_spacing_seconds > 0 and row_position < len(raw_rows) - 1:
            time.sleep(args.request_spacing_seconds)

    predictions = [str(row.get("prediction") or "") for row in fixed_rows]
    rows, extraction_reports = runner.prediction_rows_from_texts(case, prompt_rows, predictions)
    args.raw_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in fixed_rows) + "\n",
        encoding="utf-8",
    )
    runner.write_prediction_jsonl(args.run_pred_jsonl, rows)
    runner.write_json(
        runner.run_metadata_path(args.run_pred_jsonl),
        {
            "created_at": runner.utc_now(),
            "status": "completed",
            "formal_result": True,
            "experiment_id": case.experiment_id,
            "source_id": case.source_id,
            "narrative_route": case.narrative_route,
            "prompt_csv": str(runner.ROUTE_CSV_PATHS[case.narrative_route]),
            "rows": len(rows),
            "runtime": {
                "stage": "direct_binding_generation",
                "engine": args.engine,
                "actual_engine": args.engine,
                "prediction_contract": "data_binding_generator",
                "route_status": route_status,
                "raw_output": str(args.raw_jsonl),
                "system_prompt": "experiment6_binding_json_v1",
                "max_tokens": args.max_tokens,
                "row_timeout_seconds": args.row_timeout_seconds,
                "parallelism": 1,
                "config": config.to_public_dict(),
                "repair": "row_level_timeout_repair",
                "repaired_indexes": repaired_indexes,
                "variables_md_status": variables_md_status,
            },
            "extraction_reports": extraction_reports,
        },
    )
    print(
        json.dumps(
            {
                "run_pred_jsonl": str(args.run_pred_jsonl),
                "rows": len(rows),
                "repaired_indexes": repaired_indexes,
                "variables_md_status": variables_md_status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
