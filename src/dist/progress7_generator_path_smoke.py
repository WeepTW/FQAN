"""Dry-run Progress 7 generator experiment paths and credential gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_ENGINE = "qwen3_6"


@dataclass(frozen=True)
class Route:
    experiment: str
    route_id: str
    engine: str
    input_json: Path
    output_jsonl: Path
    expected_blocker: str | None = None


def route_matrix(experiment5_input: Path) -> list[Route]:
    flan_o = REPO_ROOT / "Experiment/finqa_flan_o/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json"
    flan_d = REPO_ROOT / "Experiment/finqa_flan_d/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json"
    mistral_o = REPO_ROOT / "Experiment/finqa_Mistral_o/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json"
    apollo = REPO_ROOT / "Data_Target_Module/Apollo/output/best_matched_with_retrieved_facts_and_questions_apollo.json"
    missing_dev = REPO_ROOT / "Data_Target_Module/Finqa/dev_not_configured/best_matched.json"
    out_root = REPO_ROOT / "Experiment/progress7_path_smoke/generator_outputs"
    routes = [
        Route("5", "exp5_qwen_few10", QWEN_ENGINE, experiment5_input, out_root / "exp5_qwen_few10.jsonl"),
        Route("6", "exp6_qwen_flan_o_test", QWEN_ENGINE, flan_o, out_root / "exp6_qwen_flan_o_test.jsonl"),
        Route("6", "exp6_qwen_flan_o_dev", QWEN_ENGINE, missing_dev, out_root / "exp6_qwen_flan_o_dev.jsonl", "missing_dev_artifact"),
        Route("6", "exp6_qwen_flan_d_test", QWEN_ENGINE, flan_d, out_root / "exp6_qwen_flan_d_test.jsonl"),
        Route("6", "exp6_qwen_flan_d_dev", QWEN_ENGINE, missing_dev, out_root / "exp6_qwen_flan_d_dev.jsonl", "missing_dev_artifact"),
        Route("6", "exp6_qwen_apollo_test", QWEN_ENGINE, apollo, out_root / "exp6_qwen_apollo_test.jsonl"),
        Route("6", "exp6_qwen_apollo_dev", QWEN_ENGINE, missing_dev, out_root / "exp6_qwen_apollo_dev.jsonl", "missing_dev_artifact"),
        Route("progress7", "progress7_qwen_mistral_o_current_test", QWEN_ENGINE, mistral_o, out_root / "progress7_qwen_mistral_o_current_test.jsonl"),
        Route("7", "exp7_mistral4_flan_o_test", "mistral4", flan_o, out_root / "exp7_mistral4_flan_o_test.jsonl", "mistral4_frozen"),
        Route("7", "exp7_mistral4_flan_d_test_dynamic", "mistral4", flan_d, out_root / "exp7_mistral4_flan_d_test_dynamic.jsonl", "mistral4_frozen"),
        Route("7", "exp7_mistral4_apollo_test", "mistral4", apollo, out_root / "exp7_mistral4_apollo_test.jsonl", "mistral4_frozen"),
    ]
    for retriever_name, input_json in [("flan_o", flan_o), ("flan_d", flan_d)]:
        for engine in [QWEN_ENGINE, "mistral4", "gpt4_1", "gpt5_3_codexS", "gpt5_5"]:
            routes.append(
                Route(
                    "8",
                    f"exp8_test_{retriever_name}_{engine}",
                    engine,
                    input_json,
                    out_root / f"exp8_test_{retriever_name}_{engine}.jsonl",
                    "mistral4_frozen" if engine == "mistral4" else None,
                )
            )
            routes.append(
                Route(
                    "9",
                    f"exp9_dev_{retriever_name}_{engine}",
                    engine,
                    missing_dev,
                    out_root / f"exp9_dev_{retriever_name}_{engine}.jsonl",
                    "missing_dev_artifact",
                )
            )
    return routes


def run_dry_run(route: Route) -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        str(REPO_ROOT / "new_full_finqa_run.py"),
        "--engine",
        route.engine,
        "--input-json",
        str(route.input_json),
        "--output-jsonl",
        str(route.output_jsonl),
        "--limit",
        "1",
    ]
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError:
        parsed = None

    input_exists = route.input_json.exists()
    engine_available = bool(parsed and parsed.get("engine", {}).get("available"))
    missing_credentials = parsed.get("engine", {}).get("missing_credentials", []) if parsed else []
    if not input_exists:
        status = "blocked_missing_input"
    elif route.expected_blocker == "mistral4_frozen":
        status = "blocked_frozen_runtime"
    elif not engine_available:
        status = "blocked_missing_credentials"
    elif proc.returncode == 0:
        status = "ready_to_execute"
    else:
        status = "dry_run_failed"

    return {
        "experiment": route.experiment,
        "route_id": route.route_id,
        "engine": route.engine,
        "input_json": str(route.input_json),
        "input_exists": input_exists,
        "output_jsonl": str(route.output_jsonl),
        "expected_blocker": route.expected_blocker,
        "status": status,
        "missing_credentials": missing_credentials,
        "returncode": proc.returncode,
        "stdout_json": parsed,
        "stderr_tail": proc.stderr[-1200:],
        "command": command,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progress 7 generator path dry-run matrix.")
    parser.add_argument("--experiment5-input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes = route_matrix(args.experiment5_input)
    results = [run_dry_run(route) for route in routes]
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    payload = {"routes": results, "status_counts": counts}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
