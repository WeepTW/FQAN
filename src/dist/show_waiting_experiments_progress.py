#!/usr/bin/env python3
"""Read-only progress summary for the ordered Experiment 6/7 queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def latest_queue_root() -> Path:
    roots = sorted((SRC_ROOT / "Experiment").glob("waiting_experiments_*"))
    if not roots:
        raise SystemExit("No waiting_experiments_* queue was found.")
    return roots[-1]


def tail(path: Path, count: int = 8) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-count:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue_root", nargs="?", type=Path)
    args = parser.parse_args()
    root = (args.queue_root or latest_queue_root()).resolve()
    paths = load_json(root / "queue_paths.json")
    exp6_raw = str(paths.get("experiment6Root") or "")
    exp6_root = Path(exp6_raw) if exp6_raw else None
    exp7_id = str(paths.get("experiment7Id") or "")
    exp7_root = SRC_ROOT / "Experiment" / exp7_id if exp7_id else None
    stop_file = Path(str(paths.get("stopRequestFile") or root / "STOP_AFTER_CURRENT_CASE"))

    print(f"queue_root: {root}")
    print(f"status: {paths.get('status', 'unknown')}")
    print(f"terminal_stage: {paths.get('terminalStage', '-')}")
    print(f"stop_requested: {stop_file.is_file()}")
    print(f"safe_stop_command: touch {stop_file}")

    manifests = sorted((exp6_root / "manifests").glob("*.json")) if exp6_root else []
    counts: Counter[str] = Counter()
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        status = str(manifest.get("status") or "unknown")
        case_id = str(manifest.get("caseId") or manifest_path.name.split("__run_")[0])
        if status.startswith("completed"):
            counts[case_id] += 1
    print(f"experiment6_manifests: {len(manifests)}/30")
    for case_id in (
        "6_finflier_prompt_flan_z_adapter_long_context",
        "6_finflier_prompt_flan_m_adapter_long_context",
        "6_finflier_prompt_flan_d_adapter_long_context",
    ):
        print(f"  {case_id}: {counts[case_id]}/10")
    exp6_eval = Path(str(exp6_root).replace("_generation_", "_evaluation_v6_1_0_")) if exp6_root else None
    exp6_report = exp6_eval / "evaluation_report.json" if exp6_eval else None
    print(
        "experiment6_evaluation_report: "
        + (str(exp6_report) if exp6_report and exp6_report.is_file() else "pending")
    )

    score_report = exp7_root / "generator" / "score_report.json" if exp7_root else None
    score = load_json(score_report) if score_report else {}
    print(f"experiment7_completed_cases: {score.get('completed_cases', 0)}/16")
    print(f"experiment7_mean_EA: {score.get('mean_execution_accuracy_unweighted')}")
    print(
        "experiment7_score_report: "
        + (str(score_report) if score_report and score_report.is_file() else "pending")
    )
    for item in score.get("items", []):
        print(
            "  {retriever_id}:{dataset} EA={execution_accuracy} status={score_status}".format(
                **item
            )
        )

    stop_status = root / "stop_status.json"
    if stop_status.is_file():
        print(f"stop_status: {stop_status}")
    for label, log_path in (
        ("experiment6", root / "experiment6_queue.log"),
        ("experiment7", root / "experiment7_queue.log"),
    ):
        lines = tail(log_path)
        if lines:
            print(f"{label}_log_tail:")
            for line in lines:
                print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
