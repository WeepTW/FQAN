#!/usr/bin/env python3
"""Run the single-case FLAN + full FinFlier long-context diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment6_paths import PATHS
import run_experiment6_narrative2_generation as generation


PYTHON = Path(sys.executable).resolve()
RUNNER = PATHS.resolve("dist", "run_experiment6_narrative2_generation.py")
MATERIALIZER = PATHS.resolve("dist", "materialize_experiment6_binding_candidates.py")
VALIDATOR = PATHS.resolve("dist", "validate_experiment6_binding_candidates.py")
EVALUATOR = PATHS.resolve("dist", "evaluate_experiment6_binding_candidates_v1.py")
TABLE_BUILDER = PATHS.resolve("dist", "build_experiment6_binding_candidate_score_tables.py")
CONFIG = PATHS.resolve(
    "repo", "config/experiment6_narrative2_generation_finflier_flan_long_context.json"
)
EVALUATION_CONFIG = PATHS.resolve(
    "repo", "config/experiment6_finflier_flan_long_context_evaluation_v6_1.json"
)
SOURCE_REGISTRY = PATHS.resolve("repo", "config/experiment6_source_registry.json")
CASE_ID = "6_finflier_prompt_flan_base_long_context"
GPU_DEVICE = "1"
LONGEST_SOURCE = "FT_005"
RUNS = tuple(range(1, 11))
ROWS = 85
ALLOWED_STATUSES = {"completed", "completed_with_format_errors"}
OOM_MARKERS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


class OrchestrationError(RuntimeError):
    """Raised when continuing would violate the diagnostic contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OrchestrationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def matrix_case() -> generation.MatrixCase:
    config = generation.load_config(CONFIG)
    cases = generation.expand_matrix(config)
    if len(cases) != 1 or cases[0].output_id != CASE_ID:
        raise OrchestrationError("config must contain the exact one-case FLAN matrix")
    case = cases[0]
    if case.source_id != "flan_t5_large" or case.route != "direct-binding":
        raise OrchestrationError("long-context case must be FLAN base direct-binding")
    source = generation.ACTIVE_SOURCE_REGISTRY.resolve_source(case.source_id, case.route)
    if source.get("kind") != "base" or source.get("family") != "flan":
        raise OrchestrationError("registry identity is not FLAN base")
    return case


def runner_command(
    root: Path,
    *,
    device: str,
    run: int | None = None,
    preflight_only: bool = False,
    smoke: bool = False,
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
        CASE_ID,
        "--cuda-visible-devices",
        device,
    ]
    if run is not None:
        command.extend(["--run", str(run)])
    if preflight_only:
        command.append("--preflight-only")
    if smoke:
        command.extend(["--smoke-only", "--row-source", LONGEST_SOURCE])
    if fresh:
        command.append("--no-resume")
    return command


def run_logged(
    command: Sequence[str], log_path: Path, *, environment: Mapping[str, str] | None = None
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if environment:
        env.update(environment)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            cwd=PATHS.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def manifest_path(root: Path, run: int) -> Path:
    return root / "manifests" / f"{CASE_ID}__run_{run:02d}.json"


def run_dir(root: Path, run: int) -> Path:
    return root / "cases" / CASE_ID / f"run_{run:02d}"


def valid_manifest(root: Path, run: int, *, expected_rows: int = ROWS) -> bool:
    path = manifest_path(root, run)
    if not path.is_file():
        return False
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError, OrchestrationError):
        return False
    checks = (
        manifest.get("status") in ALLOWED_STATUSES,
        manifest.get("official") is True,
        manifest.get("outputId") == CASE_ID,
        manifest.get("run") == run,
        manifest.get("inputType") == "FinFlier",
        manifest.get("declaredRoute") == "direct-binding",
        manifest.get("effectiveRoute") == "direct-binding",
        manifest.get("adapter") is None,
        manifest.get("converterModel") is None,
        manifest.get("expectedRows") == expected_rows,
        manifest.get("runtimeBlockedRows") == 0,
    )
    if not all(checks):
        return False
    files = manifest.get("files") or {}
    hashes = manifest.get("hashes") or {}
    for name in (
        "predictions",
        "rawResponse",
        "prompts",
        "runtime",
        "formatReport",
        "nonformalRepair",
        "stage1Raw",
    ):
        path_value = Path(str(files.get(name) or ""))
        if not path_value.is_file() or sha256_file(path_value) != hashes.get(name):
            return False
    predictions = [
        json.loads(line)
        for line in Path(files["predictions"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        len(predictions) == expected_rows
        and len({row.get("source") for row in predictions}) == expected_rows
    )


def latest_preflight(root: Path) -> dict[str, Any]:
    paths = sorted((root / "preflight_invocations").glob("*.json"))
    if not paths:
        raise OrchestrationError("preflight invocation was not written")
    return read_json(paths[-1])


def validate_token_report(report: Mapping[str, Any], *, measurements: int) -> None:
    direct = ((report.get("tokens") or {}).get("directBinding") or {}).get(CASE_ID)
    if not isinstance(direct, Mapping):
        raise OrchestrationError("FLAN direct-binding token report is missing")
    expected = {
        "family": "flan",
        "route": "direct-binding",
        "maxInputAllowed": 16896,
        "contextWindow": 20992,
        "maxNewTokens": 4096,
        "measurements": measurements,
        "truncationAllowed": False,
        "structuredOutput": "off",
        "adapter": None,
        "converter": None,
    }
    mismatches = {
        key: {"expected": value, "actual": direct.get(key)}
        for key, value in expected.items()
        if direct.get(key) != value
    }
    if mismatches:
        raise OrchestrationError(f"token contract mismatch: {mismatches}")
    if int(direct["maxObserved"]) > 16896:
        raise OrchestrationError("a frozen prompt exceeds the 16,896-token input gate")
    if int(direct["maxPromptPlusCompletion"]) > 20992:
        raise OrchestrationError("prompt plus completion exceeds the 20,992-token gate")
    if measurements == 85 and (
        int(direct["minObserved"]) != 9675 or int(direct["maxObserved"]) != 16574
    ):
        raise OrchestrationError(
            "frozen prompt token range drifted from 9,675..16,574"
        )


def preflight(root: Path) -> dict[str, Any]:
    matrix_case()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "scheduler" / "preflight.log"
    returncode = run_logged(
        runner_command(root, device=GPU_DEVICE, preflight_only=True), log_path
    )
    if returncode != 0:
        raise OrchestrationError(f"preflight failed; see {log_path}")
    report = latest_preflight(root)
    validate_token_report(report, measurements=ROWS)
    summary = {
        "protocol": "experiment6-finflier-flan-long-context-preflight-v1",
        "status": "passed",
        "time": utc_now(),
        "caseId": CASE_ID,
        "rows": ROWS,
        "configSha256": sha256_file(CONFIG),
        "promptAssetSha256": "948de4863d1a3901c82682b81875aa848b20e03efeff41c2274feb1dc04a5051",
        "tokenReport": report["tokens"]["directBinding"][CASE_ID],
        "log": str(log_path),
    }
    write_json(root / "scheduler" / "preflight_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def checkpoint_records(root: Path, run: int) -> list[dict[str, Any]]:
    files = sorted((run_dir(root, run) / "raw").glob("*.checkpoint.jsonl"))
    if not files:
        return []
    records = []
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def diagnostic_text(root: Path, run: int, log_path: Path) -> str:
    paths = [log_path]
    paths.extend(sorted((run_dir(root, run) / "raw").glob("*.log")))
    chunks = []
    for path in paths:
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks).lower()


def attempt_path(root: Path, run: int) -> Path:
    return root / "scheduler" / "run_attempts" / f"{CASE_ID}__run_{run:02d}.json"


def read_attempts(root: Path, run: int) -> list[dict[str, Any]]:
    path = attempt_path(root, run)
    return list(read_json(path).get("attempts") or []) if path.is_file() else []


def write_attempts(root: Path, run: int, attempts: Sequence[Mapping[str, Any]]) -> None:
    write_json(
        attempt_path(root, run),
        {
            "protocol": "experiment6-finflier-flan-device-attempts-v1",
            "caseId": CASE_ID,
            "run": run,
            "attempts": list(attempts),
        },
    )


def invocation(
    root: Path,
    run: int,
    *,
    device: str,
    fresh: bool,
    smoke: bool,
) -> dict[str, Any]:
    label = "cpu" if device == "cpu" else f"gpu_{device}"
    log_path = root / "scheduler" / "job_logs" / f"{CASE_ID}__run_{run:02d}__{label}.log"
    before = checkpoint_records(root, run)
    started_at = utc_now()
    started = time.monotonic()
    returncode = run_logged(
        runner_command(
            root,
            device=device,
            run=run,
            smoke=smoke,
            fresh=fresh,
        ),
        log_path,
    )
    after = checkpoint_records(root, run)
    text = diagnostic_text(root, run, log_path)
    return {
        "device": "cpu" if device == "cpu" else "gpu",
        "cudaVisibleDevices": "" if device == "cpu" else device,
        "numericPrecision": "float32" if device == "cpu" else "bfloat16",
        "startedAt": started_at,
        "finishedAt": utc_now(),
        "runtimeSeconds": time.monotonic() - started,
        "returncode": returncode,
        "checkpointRowsBefore": len(before),
        "checkpointRowsAfter": len(after),
        "checkpointIndicesAfter": sorted(int(item["index"]) for item in after),
        "oom": any(marker in text for marker in OOM_MARKERS),
        "log": str(log_path),
    }


def write_execution_manifest(
    root: Path,
    run: int,
    attempts: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
) -> Path:
    prompts = [
        json.loads(line)
        for line in (run_dir(root, run) / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cpu_indices: set[int] = set()
    gpu_indices: set[int] = set()
    previous: set[int] = set()
    for attempt in attempts:
        current = {int(value) for value in attempt.get("checkpointIndicesAfter") or []}
        produced = current - previous
        (cpu_indices if attempt["device"] == "cpu" else gpu_indices).update(produced)
        previous = current
    if len(previous) != expected_rows:
        raise OrchestrationError(
            f"device manifest has {len(previous)} checkpoint rows, expected {expected_rows}"
        )
    by_index = {int(item["index"]): item for item in prompts}
    rows = []
    for index in range(expected_rows):
        if index not in by_index:
            raise OrchestrationError(f"prompt record missing index {index}")
        device = "cpu" if index in cpu_indices else "gpu"
        precision = "float32" if device == "cpu" else "bfloat16"
        rows.append(
            {
                "index": index,
                "source": by_index[index]["source"],
                "directPromptSha256": by_index[index]["directPromptSha256"],
                "device": device,
                "numericPrecision": precision,
            }
        )
    payload = {
        "protocol": "experiment6-finflier-flan-row-execution-v1",
        "caseId": CASE_ID,
        "run": run,
        "fallbackReason": "CUDA OOM" if cpu_indices else None,
        "attempts": list(attempts),
        "rows": rows,
    }
    path = run_dir(root, run) / "row_execution_manifest.json"
    write_json(path, payload)
    return path


def execute_run(
    root: Path,
    run: int,
    *,
    fresh: bool,
    smoke: bool = False,
    expected_rows: int = ROWS,
) -> dict[str, Any]:
    if valid_manifest(root, run, expected_rows=expected_rows):
        return {"run": run, "status": "already_completed"}
    attempts = read_attempts(root, run)
    previous_gpu_oom = any(item.get("device") == "gpu" and item.get("oom") for item in attempts)
    device = "cpu" if previous_gpu_oom else GPU_DEVICE
    attempt = invocation(
        root,
        run,
        device=device,
        fresh=fresh and not attempts,
        smoke=smoke,
    )
    attempts.append(attempt)
    write_attempts(root, run, attempts)
    if not valid_manifest(root, run, expected_rows=expected_rows):
        if device != "cpu" and attempt["oom"]:
            cpu_attempt = invocation(
                root,
                run,
                device="cpu",
                fresh=False,
                smoke=smoke,
            )
            attempts.append(cpu_attempt)
            write_attempts(root, run, attempts)
        if not valid_manifest(root, run, expected_rows=expected_rows):
            latest = attempts[-1]
            reason = "CPU fallback failed" if latest["device"] == "cpu" else "non-OOM GPU failure"
            raise OrchestrationError(
                f"{CASE_ID} run {run} is runtime-blocked: {reason}; see {latest['log']}"
            )
    execution_path = write_execution_manifest(
        root, run, attempts, expected_rows=expected_rows
    )
    manifest = read_json(manifest_path(root, run))
    event = {
        "event": "run_completed",
        "time": utc_now(),
        "run": run,
        "seed": 2026073100 + run,
        "casesCompleted": 1,
        "rows": expected_rows,
        "gpuRows": sum(
            1
            for row in read_json(execution_path)["rows"]
            if row["device"] == "gpu"
        ),
        "cpuRows": sum(
            1
            for row in read_json(execution_path)["rows"]
            if row["device"] == "cpu"
        ),
        "compatibilityFingerprint": manifest["compatibilityFingerprint"],
        "executionManifest": str(execution_path),
        "executionManifestSha256": sha256_file(execution_path),
    }
    append_jsonl(root / "scheduler" / "agent_events.jsonl", event)
    print(json.dumps(event, ensure_ascii=False), flush=True)
    return event


def smoke(root: Path) -> dict[str, Any]:
    # A one-row smoke has a deliberately distinct scientific fingerprint from
    # the formal 85-row root, so it must not inherit a full preflight snapshot.
    event = execute_run(
        root, 1, fresh=not manifest_path(root, 1).exists(), smoke=True, expected_rows=1
    )
    report = latest_preflight(root)
    validate_token_report(report, measurements=1)
    direct = report["tokens"]["directBinding"][CASE_ID]
    if int(direct["maxObserved"]) != 16574:
        raise OrchestrationError(
            f"{LONGEST_SOURCE} no longer measures 16,574 FLAN tokens"
        )
    summary = {
        "protocol": "experiment6-finflier-flan-long-context-smoke-v1",
        "status": "passed",
        "time": utc_now(),
        "source": LONGEST_SOURCE,
        "tokens": int(direct["maxObserved"]),
        "event": event,
    }
    write_json(root / "scheduler" / "smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def status(root: Path) -> dict[str, Any]:
    complete_runs = [run for run in RUNS if valid_manifest(root, run)]
    blocked = []
    for run in RUNS:
        path = manifest_path(root, run)
        if path.is_file() and read_json(path).get("status") == "runtime_blocked":
            blocked.append(run)
    result = {
        "protocol": "experiment6-finflier-flan-long-context-status-v1",
        "time": utc_now(),
        "outputRoot": str(root),
        "caseId": CASE_ID,
        "completedRuns": complete_runs,
        "completedManifests": len(complete_runs),
        "completedRows": len(complete_runs) * ROWS,
        "runtimeBlockedRuns": blocked,
        "status": "completed" if len(complete_runs) == 10 else "incomplete",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def materialization_config(root: Path) -> Path:
    manifests = [read_json(manifest_path(root, run)) for run in RUNS]
    protocols = {str(item["protocol"]) for item in manifests}
    fingerprints = {str(item["compatibilityFingerprint"]) for item in manifests}
    if len(protocols) != 1 or len(fingerprints) != 1:
        raise OrchestrationError("generation protocol/fingerprint differs across runs")
    path = root / "finalization" / "materialization_config.json"
    write_json(
        path,
        {
            "schemaVersion": 1,
            "protocol": "experiment6-binding-candidate-materialization-v1",
            "sourceProtocol": next(iter(protocols)),
            "sourceCompatibilityFingerprint": next(iter(fingerprints)),
            "expectedCases": 1,
            "expectedRuns": 10,
            "expectedRows": ROWS,
            "caseIds": [CASE_ID],
            "requiredBindingKeys": [
                "ObjectName",
                "DataName",
                "Position",
                "Trend",
                "Num",
                "Text",
            ],
            "requireRepairCoverage": True,
        },
    )
    return path


def finalization_command(
    command: Sequence[str], log_path: Path, *, cpu_only: bool = False
) -> None:
    environment = None
    if cpu_only:
        environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    returncode = run_logged(command, log_path, environment=environment)
    if returncode != 0:
        raise OrchestrationError(f"finalization failed; see {log_path}")


def finalize(root: Path, evaluation_root: Path) -> dict[str, Any]:
    if status(root)["status"] != "completed":
        raise OrchestrationError("all ten 85-row runs must complete before evaluation")
    candidate_root = root.with_name(root.name + "_binding_candidates_v1")
    config_path = materialization_config(root)
    final_logs = root / "finalization" / "logs"
    if not candidate_root.exists():
        finalization_command(
            [
                str(PYTHON),
                "-B",
                str(MATERIALIZER),
                "--generation-root",
                str(root),
                "--config",
                str(config_path),
                "--output-root",
                str(candidate_root),
            ],
            final_logs / "materialize.log",
        )
    finalization_command(
        [str(PYTHON), "-B", str(VALIDATOR), "--root", str(candidate_root)],
        final_logs / "validate_candidates.log",
    )
    if evaluation_root.exists() and (evaluation_root / "evaluation_report.json").exists():
        raise OrchestrationError(
            f"evaluation output already exists; refuse overwrite: {evaluation_root}"
        )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    finalization_command(
        [
            str(PYTHON),
            "-B",
            str(EVALUATOR),
            "--version",
            "v6.1.0",
            "--scope",
            "flan-long-context",
            "--candidate-root",
            str(candidate_root),
            "--evaluation-root",
            str(evaluation_root),
            "--config",
            str(EVALUATION_CONFIG),
        ],
        final_logs / "evaluate.log",
        cpu_only=True,
    )
    finalization_command(
        [
            str(PYTHON),
            "-B",
            str(TABLE_BUILDER),
            "--evaluation-report",
            str(evaluation_root / "evaluation_report.json"),
            "--evaluation-root",
            str(evaluation_root),
            "--source-registry",
            str(SOURCE_REGISTRY),
            "--output-dir",
            str(evaluation_root),
        ],
        final_logs / "build_tables.log",
        cpu_only=True,
    )
    report = read_json(evaluation_root / "evaluation_report.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = PATHS.log / f"{stamp}_experiment6_finflier_flan_long_context_complete.md"
    log_lines = [
        "# Experiment 6 FLAN + full FinFlier prompt long-context diagnostic",
        "",
        f"- Completed: `{utc_now()}`.",
        f"- Generation root: `{root}`.",
        f"- Candidate root: `{candidate_root}`.",
        f"- Evaluation root: `{evaluation_root}`.",
        "- Scope: one FLAN no-adaptor case, 10 runs × 85 Sources.",
        "- Context contract: 16,896 input + 4,096 completion = 20,992; truncation=false.",
        "- Result role: long-context diagnostic only; official=false; claimEligible=false.",
        f"- Five-field case-mean macro-F1: `{report['overall']['caseMeanMacroF1']}`.",
        "- Text: NA (judge deferred).",
        "",
        "Primary files:",
        "",
        f"- `{evaluation_root / 'experiment_6_v6_欄位分數_mean.md'}`",
        f"- `{evaluation_root / 'evaluation_report.json'}`",
        f"- `{evaluation_root / 'evaluation_report.md'}`",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    summary = {
        "status": "completed",
        "generationRoot": str(root),
        "candidateRoot": str(candidate_root),
        "evaluationRoot": str(evaluation_root),
        "caseMeanMacroF1": report["overall"]["caseMeanMacroF1"],
        "completionLog": str(log_path),
        "completionLogSha256": sha256_file(log_path),
    }
    write_json(root / "finalization" / "completion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def execute(root: Path, evaluation_root: Path, *, fresh: bool) -> dict[str, Any]:
    if fresh and any((root / "manifests").glob("*.json")):
        raise OrchestrationError("start refuses existing manifests; use resume")
    if not (root / "scheduler" / "preflight_summary.json").is_file():
        preflight(root)
    for run in RUNS:
        execute_run(root, run, fresh=fresh)
    final_status = status(root)
    write_json(root / "scheduler" / "completion_summary.json", final_status)
    finalize(root, evaluation_root)
    return final_status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "smoke", "start", "resume", "status", "finalize")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_root.resolve()
    try:
        if args.command == "preflight":
            preflight(root)
        elif args.command == "smoke":
            smoke(root)
        elif args.command == "status":
            status(root)
        else:
            if args.evaluation_root is None:
                raise OrchestrationError(f"{args.command} requires --evaluation-root")
            evaluation_root = args.evaluation_root.resolve()
            if args.command == "finalize":
                finalize(root, evaluation_root)
            else:
                execute(root, evaluation_root, fresh=args.command == "start")
    except (OrchestrationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"status": "runtime-blocked", "error": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
