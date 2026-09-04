#!/usr/bin/env python3
"""Build a portable GPT-4.1 task package for unfinished Experiment 6 rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment6_binding_generation as runner  # noqa: E402


GPT41_CASES = [
    runner.MatrixCase("6_FinFlier_gpt4.1", "gpt4_1", "narrative_original"),
    runner.MatrixCase("6_gpt4.1_z", "gpt4_1", "narrative_zero_shot"),
    runner.MatrixCase("6_gpt4.1_m", "gpt4_1", "narrative_many_shot"),
    runner.MatrixCase("6_gpt4.1_d", "gpt4_1", "narrative_dynamic_shot"),
]


def read_raw_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def is_completed(raw: dict[str, Any]) -> bool:
    return raw.get("status") == "completed" and bool(raw.get("prediction"))


def task_id(case: runner.MatrixCase, run_number: int, row_index: int) -> str:
    return f"{case.experiment_id}:run_{run_number:02d}:row_{row_index:03d}"


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pred_dir = runner.REPO_ROOT / "Experiment" / args.expt_id / "binding_eval_predictions"
    tasks: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for case in GPT41_CASES:
        prompt_rows = runner.read_prompt_rows(runner.ROUTE_CSV_PATHS[case.narrative_route], args.limit)
        case_missing = 0
        case_completed = 0
        for run_number in range(1, args.num_runs + 1):
            raw_path = pred_dir / "raw" / f"{case.experiment_id}.run_{run_number:02d}.jsonl"
            raw_rows = read_raw_rows(raw_path)
            raw_by_index = {
                int(raw["index"]): raw
                for raw in raw_rows
                if isinstance(raw, dict) and str(raw.get("index", "")).isdigit()
            }
            for row_index, row in enumerate(prompt_rows):
                existing = raw_by_index.get(row_index)
                if existing and is_completed(existing):
                    case_completed += 1
                    continue
                case_missing += 1
                prompt = runner.direct_binding_prompt(row)
                tasks.append(
                    {
                        "task_id": task_id(case, run_number, row_index),
                        "experiment_id": case.experiment_id,
                        "source_id": case.source_id,
                        "narrative_route": case.narrative_route,
                        "run": run_number,
                        "row_index": row_index,
                        "case_id": row.get("Source") or f"row_{row_index + 1}",
                        "prompt": prompt,
                        "system_prompt": runner.BINDING_JSON_SYSTEM_PROMPT,
                        "max_tokens": args.max_tokens,
                    }
                )
        summary.append(
            {
                "experiment_id": case.experiment_id,
                "narrative_route": case.narrative_route,
                "completed_rows": case_completed,
                "missing_rows": case_missing,
            }
        )
    return tasks, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expt-id", default="experiment_6_api_key_gpt41_formal_20260614T1800Z")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (runner.REPO_ROOT / "Experiment" / args.expt_id / "external_gpt41_package")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, summary = build_tasks(args)
    tasks_jsonl = output_dir / "tasks.jsonl"
    tasks_jsonl.write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__).with_name("run_experiment6_gpt41_external_tasks.py"), output_dir / "run_external_gpt41.py")
    manifest = {
        "created_at": runner.utc_now(),
        "expt_id": args.expt_id,
        "num_runs": args.num_runs,
        "top_k": args.top_k,
        "task_count": len(tasks),
        "tasks_jsonl": str(tasks_jsonl),
        "results_jsonl": str(output_dir / "results.jsonl"),
        "summary": summary,
        "run_command": (
            "python run_external_gpt41.py --tasks-jsonl tasks.jsonl --results-jsonl results.jsonl "
            "--request-spacing-seconds 20 --retry-max 3 --retry-wait-seconds 600"
        ),
        "import_command": (
            f"conda run -n fnqa python src/dist/import_experiment6_gpt41_external_results.py "
            f"--expt-id {args.expt_id} --results-jsonl {output_dir / 'results.jsonl'}"
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 6 GPT-4.1 External Package",
                "",
                "Run this directory in an environment that can call GPT-4.1.",
                "",
                "Required route: set either `OPENAI_API_KEY`, `OPENAI_BASE_URL + OPENAI_API_KEY`, or Azure OpenAI variables.",
                "",
                "```bash",
                manifest["run_command"],
                "```",
                "",
                "Copy `results.jsonl` back to this directory in the FQAN workspace, then run:",
                "",
                "```bash",
                manifest["import_command"],
                "```",
                "",
                "The runner is serial and sends one GPT-4.1 request at a time.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
