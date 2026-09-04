#!/usr/bin/env python3
"""Orchestrate the No-adaptor × FinFlier Experiment 6 comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiment6_paths import PATHS
import run_experiment6_narrative2_generation as generation


PYTHON = Path(sys.executable).resolve()
RUNNER = PATHS.resolve("dist", "run_experiment6_narrative2_generation.py")
CONFIG = PATHS.resolve(
    "repo", "config/experiment6_narrative2_generation_finflier_no_adapter.json"
)
CASE_DEVICES = {
    "6_finflier_prompt_flan_base": "1",
    "6_finflier_prompt_mistral_base": "0",
    "6_finflier_prompt_t5gemma2_base": "1",
}
ALLOWED_STATUSES = {"completed", "completed_with_format_errors"}


class OrchestrationError(RuntimeError):
    """Raised when a run cannot proceed without violating the protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_event(root: Path, value: Mapping[str, Any]) -> None:
    path = root / "scheduler" / "agent_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def cases() -> list[generation.MatrixCase]:
    config = generation.load_config(CONFIG)
    resolved = generation.expand_matrix(config)
    if {item.output_id for item in resolved} != set(CASE_DEVICES):
        raise OrchestrationError("FinFlier config is not the exact three-case matrix")
    if any(item.route != "direct-binding" for item in resolved):
        raise OrchestrationError("every FinFlier case must use direct-binding")
    return resolved


def runner_command(
    root: Path,
    output_id: str,
    *,
    device: str,
    run: int | None = None,
    preflight_only: bool = False,
    fresh: bool = False,
) -> list[str]:
    command = [
        str(PYTHON),
        "-B",
        str(RUNNER),
        "--config",
        str(CONFIG),
        "--output-root",
        str(root),
        "--case",
        output_id,
        "--cuda-visible-devices",
        device,
    ]
    if run is not None:
        command.extend(["--run", str(run)])
    if preflight_only:
        command.append("--preflight-only")
    if fresh:
        command.append("--no-resume")
    return command


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PATHS.repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    ordered = [
        "6_finflier_prompt_mistral_base",
        "6_finflier_prompt_t5gemma2_base",
        "6_finflier_prompt_flan_base",
    ]
    results: dict[str, Any] = {}
    for output_id in ordered:
        log_path = root / "scheduler" / "preflight" / f"{output_id}.log"
        returncode = run_logged(
            runner_command(
                root,
                output_id,
                device=CASE_DEVICES[output_id],
                preflight_only=True,
            ),
            log_path,
        )
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        results[output_id] = {
            "status": "passed" if returncode == 0 else "runtime-blocked",
            "returncode": returncode,
            "device": CASE_DEVICES[output_id],
            "log": str(log_path.relative_to(root)),
            "messageTail": tail,
        }
    summary = {
        "protocol": "experiment6-finflier-preflight-v1",
        "time": utc_now(),
        "inputType": "FinFlier",
        "configSha256": generation.sha256_file(CONFIG),
        "cases": results,
        "passedCases": [key for key, item in results.items() if item["status"] == "passed"],
        "blockedCases": [key for key, item in results.items() if item["status"] != "passed"],
        "status": (
            "passed" if all(item["status"] == "passed" for item in results.values())
            else "passed_with_runtime_blockers"
        ),
    }
    write_json(root / "scheduler" / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def manifest_path(root: Path, output_id: str, run: int) -> Path:
    return root / "manifests" / f"{output_id}__run_{run:02d}.json"


def valid_manifest(root: Path, output_id: str, run: int) -> bool:
    path = manifest_path(root, output_id, run)
    if not path.is_file():
        return False
    try:
        item = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if item.get("status") not in ALLOWED_STATUSES:
        return False
    if item.get("outputId") != output_id or item.get("run") != run:
        return False
    if item.get("inputType") != "FinFlier":
        return False
    if item.get("declaredRoute") != "direct-binding" or item.get("effectiveRoute") != "direct-binding":
        return False
    if item.get("adapter") is not None or item.get("converterModel") is not None:
        return False
    if item.get("expectedRows") != 85 or item.get("runtimeBlockedRows") != 0:
        return False
    prediction_path = Path(str(item.get("files", {}).get("predictions") or ""))
    if not prediction_path.is_file():
        return False
    try:
        records = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    return len(records) == 85 and len({row.get("source") for row in records}) == 85


def execute_job(root: Path, output_id: str, run: int, device: str, fresh: bool) -> dict[str, Any]:
    log_path = root / "scheduler" / "job_logs" / f"{output_id}__run_{run:02d}__gpu_{device}.log"
    started = time.monotonic()
    returncode = run_logged(
        runner_command(root, output_id, device=device, run=run, fresh=fresh),
        log_path,
    )
    return {
        "outputId": output_id,
        "run": run,
        "device": device,
        "returncode": returncode,
        "valid": valid_manifest(root, output_id, run),
        "runtimeSeconds": time.monotonic() - started,
        "log": str(log_path.relative_to(root)),
    }


def execute(root: Path, *, fresh: bool) -> dict[str, Any]:
    if fresh and any((root / "manifests").glob("*.json")):
        raise OrchestrationError("start refuses a root that already has manifests; use resume")
    summary_path = root / "scheduler" / "preflight_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else preflight(root)
    if summary.get("configSha256") != generation.sha256_file(CONFIG):
        raise OrchestrationError(
            "preflight config hash differs; use a fresh output root"
        )
    runnable = [str(item) for item in summary.get("passedCases", [])]
    blocked = [str(item) for item in summary.get("blockedCases", [])]
    if not runnable:
        raise OrchestrationError("no FinFlier case passed native-tokenizer preflight")
    for run in range(1, 11):
        pending = [item for item in runnable if not valid_manifest(root, item, run)]
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = [
                pool.submit(
                    execute_job,
                    root,
                    output_id,
                    run,
                    CASE_DEVICES[output_id],
                    fresh,
                )
                for output_id in pending
            ]
            results = [future.result() for future in futures]
        failed = [item for item in results if not item["valid"]]
        if failed:
            event = {
                "event": "run_failed",
                "time": utc_now(),
                "run": run,
                "results": results,
                "blockedCases": blocked,
            }
            append_event(root, event)
            write_json(root / "scheduler" / "failure.json", event)
            raise OrchestrationError(f"run {run} failed; queue paused")
        fingerprints = {
            read_json(manifest_path(root, item, run)).get("compatibilityFingerprint")
            for item in runnable
        }
        event = {
            "event": "run_completed",
            "time": utc_now(),
            "run": run,
            "casesCompleted": len(runnable),
            "rows": 85 * len(runnable),
            "results": results,
            "blockedCases": blocked,
            "compatibilityFingerprints": sorted(str(item) for item in fingerprints),
        }
        append_event(root, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
    result = status(root)
    write_json(root / "scheduler" / "completion_summary.json", result)
    return result


def status(root: Path) -> dict[str, Any]:
    matrix = [item.output_id for item in cases()]
    summary_path = root / "scheduler" / "preflight_summary.json"
    preflight_summary = read_json(summary_path) if summary_path.is_file() else {}
    by_case = {
        output_id: sum(valid_manifest(root, output_id, run) for run in range(1, 11))
        for output_id in matrix
    }
    passed = [str(item) for item in preflight_summary.get("passedCases", [])]
    blocked = [str(item) for item in preflight_summary.get("blockedCases", [])]
    complete_runnable = bool(passed) and all(by_case[item] == 10 for item in passed)
    result = {
        "protocol": "experiment6-finflier-status-v1",
        "time": utc_now(),
        "outputRoot": str(root),
        "inputType": "FinFlier",
        "manifestsByCase": by_case,
        "completedManifests": sum(by_case.values()),
        "completedRows": 85 * sum(by_case.values()),
        "passedCases": passed,
        "blockedCases": blocked,
        "status": (
            "completed" if complete_runnable and not blocked
            else "completed_with_runtime_blockers" if complete_runnable
            else "incomplete"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "start", "resume", "status"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    if args.command == "preflight":
        preflight(root)
    elif args.command == "status":
        status(root)
    else:
        execute(root, fresh=args.command == "start")


if __name__ == "__main__":
    main()
