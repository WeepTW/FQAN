#!/usr/bin/env python3
"""Two-GPU, seed-barrier scheduler for the corrected Experiment 6 matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from experiment6_paths import PATHS
import experiment6_corrected12 as corrected
import run_experiment6_narrative2_generation as generation


PYTHON = Path(sys.executable).resolve()
CORRECTED = PATHS.resolve("dist", "experiment6_corrected12.py")
GENERATION_CONFIG = PATHS.resolve(
    "repo", "config/experiment6_narrative2_generation_corrected_12.json"
)
DEFAULT_RUNTIME_CONFIG = PATHS.resolve(
    "repo", "config/experiment6_corrected12_dual_gpu_runtime.json"
)
ALLOWED_STATUSES = {"completed", "completed_with_format_errors"}


class SchedulerError(RuntimeError):
    """Raised when the queue cannot continue without invalidating evidence."""


@dataclass(frozen=True)
class Job:
    output_id: str
    family: str
    run: int
    estimate_seconds: float
    previous_device: str | None = None

    @property
    def key(self) -> str:
        return f"{self.output_id}__run_{self.run:02d}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def load_runtime_config(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schemaVersion") != 1:
        raise SchedulerError("dual-GPU runtime config must use schemaVersion=1")
    devices = [str(item) for item in value.get("devices", [])]
    if devices != ["0", "1"]:
        raise SchedulerError("dual-GPU scheduler requires devices [0, 1]")
    if int(value.get("expectedCases", 0)) != 12:
        raise SchedulerError("runtime config must expect 12 cases")
    if int(value.get("expectedRuns", 0)) != 10:
        raise SchedulerError("runtime config must expect 10 runs")
    if int(value.get("maxJobAttempts", 0)) != 2:
        raise SchedulerError("runtime config must allow exactly two job attempts")
    return value


def cases_and_families() -> list[tuple[str, str]]:
    config = generation.load_config(GENERATION_CONFIG)
    cases = generation.expand_matrix(config)
    if len(cases) != 12 or any(case.route != "direct-binding" for case in cases):
        raise SchedulerError("corrected generation config is not the exact 12-case matrix")
    resolved: list[tuple[str, str]] = []
    for case in cases:
        family = generation.family_for_source(case.source_id)
        if family not in {"flan", "mistral", "t5gemma2"}:
            raise SchedulerError(f"registry family missing for {case.output_id}")
        resolved.append((case.output_id, family))
    return sorted(resolved)


def manifest_path(root: Path, output_id: str, run: int) -> Path:
    return root / "manifests" / f"{output_id}__run_{run:02d}.json"


def valid_completed_manifest(root: Path, output_id: str, run: int) -> bool:
    path = manifest_path(root, output_id, run)
    if not path.is_file():
        return False
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") not in ALLOWED_STATUSES:
        return False
    return not corrected.validate_manifest_artifacts(root, manifest)


def runtime_estimates(
    root: Path,
    cases: Sequence[tuple[str, str]],
    defaults: Mapping[str, Any],
) -> dict[str, float]:
    by_case: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    family_by_case = dict(cases)
    for path in sorted((root / "manifests").glob("*.json")):
        try:
            item = read_json(path)
            seconds = float(item.get("runtimeSeconds") or 0)
            output_id = str(item.get("outputId") or "")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if seconds <= 0 or output_id not in family_by_case:
            continue
        by_case[output_id].append(seconds)
        by_family[family_by_case[output_id]].append(seconds)
    estimates: dict[str, float] = {}
    for output_id, family in cases:
        if by_case[output_id]:
            estimates[output_id] = float(median(by_case[output_id]))
        elif by_family[family]:
            estimates[output_id] = float(median(by_family[family]))
        else:
            estimates[output_id] = float(defaults[family])
    return estimates


def partition_lpt(jobs: Sequence[Job], devices: Sequence[str]) -> dict[str, list[Job]]:
    assignments = {device: [] for device in devices}
    loads = {device: 0.0 for device in devices}
    for job in sorted(jobs, key=lambda item: (-item.estimate_seconds, item.output_id)):
        eligible = [device for device in devices if device != job.previous_device]
        if not eligible:
            eligible = list(devices)
        device = min(eligible, key=lambda item: (loads[item], item))
        assignments[device].append(job)
        loads[device] += job.estimate_seconds
    return assignments


def progress_signature(run_dir: Path) -> tuple[int, int]:
    newest_ns = 0
    checkpoint_rows = 0
    if not run_dir.is_dir():
        return newest_ns, checkpoint_rows
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            newest_ns = max(newest_ns, path.stat().st_mtime_ns)
            if path.name.endswith(".checkpoint.jsonl"):
                checkpoint_rows += sum(
                    1 for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        except OSError:
            continue
    return newest_ns, checkpoint_rows


def job_command(root: Path, job: Job, device: str, fresh: bool) -> list[str]:
    command = [
        str(PYTHON),
        "-B",
        str(CORRECTED),
        "generate",
        "--output-root",
        str(root),
        "--case",
        job.output_id,
        "--run",
        str(job.run),
        "--cuda-visible-devices",
        device,
    ]
    if fresh:
        command.append("--no-resume")
    return command


def execute_job(
    root: Path,
    job: Job,
    device: str,
    fresh: bool,
    attempt: int,
    active: dict[str, dict[str, Any]],
    active_lock: threading.Lock,
) -> dict[str, Any]:
    log_path = (
        root / "scheduler" / "job_logs"
        / f"{job.key}__attempt_{attempt}__gpu_{device}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with active_lock:
        active[device] = {
            "job": job.key,
            "outputId": job.output_id,
            "run": job.run,
            "device": device,
            "attempt": attempt,
            "startedAt": utc_now(),
        }
    try:
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                job_command(root, job, device, fresh),
                cwd=PATHS.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        valid = valid_completed_manifest(root, job.output_id, job.run)
        return {
            "job": job,
            "device": device,
            "attempt": attempt,
            "returncode": completed.returncode,
            "valid": valid,
            "runtimeSeconds": time.monotonic() - started,
            "log": str(log_path),
        }
    finally:
        with active_lock:
            active.pop(device, None)


def run_assignments(
    root: Path,
    assignments: Mapping[str, Sequence[Job]],
    fresh: bool,
    attempt: int,
    heartbeat_seconds: int,
    stall_seconds: int,
) -> list[dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    active_lock = threading.Lock()
    progress: dict[str, tuple[str, tuple[int, int], float]] = {}

    def device_worker(device: str, jobs: Sequence[Job]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in jobs:
            results.append(
                execute_job(root, job, device, fresh, attempt, active, active_lock)
            )
        return results

    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(device_worker, device, jobs)
            for device, jobs in assignments.items() if jobs
        ]
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=heartbeat_seconds)
            now = time.monotonic()
            with active_lock:
                snapshot = dict(active)
            write_json(root / "scheduler" / "heartbeat.json", {
                "time": utc_now(),
                "active": snapshot,
                "pendingDeviceWorkers": len(pending),
            })
            for device, item in snapshot.items():
                run_dir = (
                    root / "cases" / str(item["outputId"])
                    / f"run_{int(item['run']):02d}"
                )
                signature = progress_signature(run_dir)
                previous = progress.get(device)
                if (
                    previous is None
                    or previous[0] != item["job"]
                    or previous[1] != signature
                ):
                    progress[device] = (item["job"], signature, now)
                elif now - previous[2] >= stall_seconds:
                    append_event(root, {
                        "time": utc_now(),
                        "event": "stall_warning",
                        **item,
                        "secondsWithoutCheckpointProgress": int(now - previous[2]),
                    })
                    progress[device] = (item["job"], signature, now)
            del done
        results: list[dict[str, Any]] = []
        for future in futures:
            results.extend(future.result())
        return results


def remaining_eta_seconds(
    root: Path,
    cases: Sequence[tuple[str, str]],
    defaults: Mapping[str, Any],
    completed_run: int,
    device_count: int,
) -> int:
    estimates = runtime_estimates(root, cases, defaults)
    remaining_runs = max(0, 10 - completed_run)
    total = remaining_runs * sum(estimates[output_id] for output_id, _ in cases)
    return int(total / max(1, device_count))


def validate_preflight(root: Path) -> str:
    path = root / "preflight.json"
    if not path.is_file():
        raise SchedulerError("corrected-12 preflight.json is missing")
    value = read_json(path)
    if value.get("status") != "passed":
        raise SchedulerError("corrected-12 preflight did not pass")
    compatibility = value.get("compatibility")
    fingerprint = compatibility.get("sha256") if isinstance(compatibility, dict) else None
    if not fingerprint:
        raise SchedulerError("preflight compatibility fingerprint is missing")
    return str(fingerprint)


def run_queue(root: Path, runtime: Mapping[str, Any], action: str) -> int:
    root = root.resolve()
    fingerprint = validate_preflight(root)
    state_path = root / "scheduler" / "state.json"
    existing_state = read_json(state_path) if state_path.is_file() else None
    if action == "start":
        if existing_state is not None or any((root / "manifests").glob("*.json")):
            raise SchedulerError("start requires a fresh preflight root; use resume")
    elif existing_state is None:
        raise SchedulerError("resume requires scheduler/state.json")
    if existing_state and existing_state.get("compatibilityFingerprint") != fingerprint:
        raise SchedulerError("scheduler resume fingerprint mismatch")

    devices = [str(item) for item in runtime["devices"]]
    fresh_first_attempts = action == "start"
    identities = [generation.gpu_execution_identity(device) for device in devices]
    if any(item is None or item.get("status") != "resolved" for item in identities):
        raise SchedulerError("both GPU identities must resolve before generation")
    if len({item.get("name") for item in identities if item}) != 1:
        raise SchedulerError("dual-GPU run requires identical GPU models")

    cases = cases_and_families()
    state = {
        "schemaVersion": 1,
        "status": "running",
        "action": action,
        "startedAt": (existing_state or {}).get("startedAt", utc_now()),
        "updatedAt": utc_now(),
        "outputRoot": str(root),
        "compatibilityFingerprint": fingerprint,
        "runtimeConfig": str(runtime["_runtimeConfigPath"]),
        "runtimeConfigSha256": str(runtime["_runtimeConfigSha256"]),
        "devices": identities,
        "completedRuns": list((existing_state or {}).get("completedRuns", [])),
    }
    write_json(state_path, state)

    defaults = runtime["initialEstimatedSeconds"]
    heartbeat_seconds = int(runtime["heartbeatSeconds"])
    stall_seconds = int(runtime["stallWarningSeconds"])
    for run_number in range(1, 11):
        if run_number in state["completedRuns"]:
            continue
        estimates = runtime_estimates(root, cases, defaults)
        jobs = [
            Job(output_id, family, run_number, estimates[output_id])
            for output_id, family in cases
            if not valid_completed_manifest(root, output_id, run_number)
        ]
        initial = partition_lpt(jobs, devices)
        results = run_assignments(
            root,
            initial,
            fresh=fresh_first_attempts,
            attempt=1,
            heartbeat_seconds=heartbeat_seconds,
            stall_seconds=stall_seconds,
        )
        all_results = list(results)
        failures = [result for result in results if not result["valid"]]
        if failures:
            retry_jobs = [
                Job(
                    result["job"].output_id,
                    result["job"].family,
                    result["job"].run,
                    result["job"].estimate_seconds,
                    previous_device=str(result["device"]),
                )
                for result in failures
            ]
            retry_assignments = partition_lpt(retry_jobs, devices)
            retry_results = run_assignments(
                root,
                retry_assignments,
                fresh=False,
                attempt=2,
                heartbeat_seconds=heartbeat_seconds,
                stall_seconds=stall_seconds,
            )
            all_results.extend(retry_results)
            failures = [result for result in retry_results if not result["valid"]]
        if failures:
            state.update({"status": "paused_failure", "updatedAt": utc_now()})
            write_json(state_path, state)
            append_event(root, {
                "time": utc_now(),
                "event": "run_failed",
                "run": run_number,
                "failedJobs": [result["job"].key for result in failures],
                "compatibilityFingerprint": fingerprint,
            })
            return 3

        manifests = [
            read_json(manifest_path(root, output_id, run_number))
            for output_id, _ in cases
        ]
        if len(manifests) != 12 or any(
            corrected.validate_manifest_artifacts(root, item) for item in manifests
        ):
            raise SchedulerError(f"run {run_number} failed its 12-manifest evidence gate")
        state["completedRuns"].append(run_number)
        state["completedRuns"] = sorted(set(state["completedRuns"]))
        state["updatedAt"] = utc_now()
        write_json(state_path, state)
        append_event(root, {
            "time": utc_now(),
            "event": "run_completed",
            "run": run_number,
            "seed": 2026073100 + run_number,
            "casesCompleted": 12,
            "rows": 1020,
            "statusCounts": dict(Counter(str(item["status"]) for item in manifests)),
            "gpuRuntimeSeconds": {
                device: sum(
                    float(result["runtimeSeconds"])
                    for result in all_results if result["device"] == device
                )
                for device in devices
            },
            "cumulativeManifests": len(state["completedRuns"]) * 12,
            "expectedManifests": 120,
            "estimatedRemainingSeconds": remaining_eta_seconds(
                root, cases, defaults, run_number, len(devices)
            ),
            "compatibilityFingerprint": fingerprint,
        })

    state.update({"status": "generation_completed", "updatedAt": utc_now()})
    write_json(state_path, state)
    return 0


def command_status(root: Path) -> int:
    state_path = root.resolve() / "scheduler" / "state.json"
    if not state_path.is_file():
        raise SchedulerError("scheduler state is missing")
    print(json.dumps(read_json(state_path), ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "status"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "status":
            return command_status(args.output_root)
        runtime = load_runtime_config(args.runtime_config.resolve())
        runtime["_runtimeConfigPath"] = str(args.runtime_config.resolve())
        runtime["_runtimeConfigSha256"] = corrected.file_sha256(
            args.runtime_config.resolve()
        )
        return run_queue(args.output_root, runtime, args.action)
    except (SchedulerError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({
            "time": utc_now(),
            "status": "blocked",
            "error": str(error),
        }, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
