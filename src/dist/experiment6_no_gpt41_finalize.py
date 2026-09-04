#!/usr/bin/env python3
"""Monitor, freeze, and evaluate the 34-case Experiment 6 no-GPT-4.1 scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_narrative2_hybrid_v4_no_gpt41 as evaluator


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "Experiment"
DEFAULT_CONFIG = (
    REPO_ROOT / "config" / "experiment6_narrative2_hybrid_v4_no_gpt41.json"
)
ALLOWED_STATUSES = {"completed", "completed_with_format_errors"}
ROW_FILES = {"predictions", "rawResponse", "prompts"}


class FinalizerError(RuntimeError):
    """Raised when formal generation artifacts violate the no-GPT-4.1 scope."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise FinalizerError(f"{path}:{line_number}: {error}") from error
            count += 1
    return count


def pid_alive(pid: object) -> bool:
    try:
        value = int(pid)
        os.kill(value, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def expected_scope(
    config: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> tuple[dict[str, str], set[str]]:
    allowed_snapshot_counts = {
        int(value)
        for value in config.get(
            "generationSnapshotAllowedOfficialCases",
            [config["generationSnapshotExpectedOfficialCases"]],
        )
    }
    if int(snapshot.get("expectedOfficialCases", -1)) not in allowed_snapshot_counts:
        raise FinalizerError("generation snapshot matrix size mismatch")
    excluded = {str(value) for value in config.get("excludedSourceIds", [])}
    outputs = evaluator.expected_output_sources(snapshot, excluded)
    if len(outputs) != int(config["expectedOfficialCases"]):
        raise FinalizerError("derived no-GPT-4.1 case count mismatch")
    return outputs, excluded


def verify_manifest(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_rows: int,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if manifest.get("status") not in ALLOWED_STATUSES:
        errors.append({"manifest": str(manifest_path), "error": "nonfinal_status"})
    if not manifest.get("official"):
        errors.append({"manifest": str(manifest_path), "error": "not_official"})
    if int(manifest.get("expectedRows", -1)) != expected_rows:
        errors.append({"manifest": str(manifest_path), "error": "expected_rows"})
    row_total = sum(
        int(manifest.get(key) or 0)
        for key in ("acceptedRows", "rejectedRows", "runtimeBlockedRows")
    )
    if row_total != expected_rows:
        errors.append({
            "manifest": str(manifest_path),
            "error": "row_accounting",
            "reported": row_total,
        })
    files = manifest.get("files")
    hashes = manifest.get("hashes")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        errors.append({"manifest": str(manifest_path), "error": "files_or_hashes"})
        return errors
    for name, raw_path in files.items():
        path = Path(str(raw_path))
        if not path.is_file():
            errors.append({"manifest": str(manifest_path), "error": "missing_file", "file": str(path)})
            continue
        expected_hash = hashes.get(name)
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            errors.append({"manifest": str(manifest_path), "error": "hash_mismatch", "file": str(path)})
            continue
        if name in ROW_FILES:
            try:
                rows = count_jsonl(path)
            except FinalizerError as error:
                errors.append({"manifest": str(manifest_path), "error": str(error)})
                continue
            if rows != expected_rows:
                errors.append({
                    "manifest": str(manifest_path),
                    "error": "jsonl_row_count",
                    "file": str(path),
                    "reported": rows,
                })
    return errors


def worker_report(
    output_root: Path,
    expected_outputs: Mapping[str, str],
    complete_keys: set[tuple[str, int]],
    expected_runs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry_path = output_root / "runtime" / "generation_workers" / "workers.json"
    if not registry_path.is_file():
        return {"registry": str(registry_path), "workers": {}}, []
    registry = read_json(registry_path)
    report: dict[str, Any] = {"registry": str(registry_path), "workers": {}}
    dead_incomplete: list[dict[str, Any]] = []
    for name, item in registry.get("workers", {}).items():
        sources = {str(value) for value in item.get("sourceIds", [])}
        output_ids = {
            output_id
            for output_id, source_id in expected_outputs.items()
            if source_id in sources
        }
        expected = len(output_ids) * expected_runs
        completed = sum(
            (output_id, run) in complete_keys
            for output_id in output_ids
            for run in range(1, expected_runs + 1)
        )
        alive = pid_alive(item.get("pid"))
        worker = {
            "pid": item.get("pid"),
            "alive": alive,
            "sourceIds": sorted(sources),
            "outputIds": sorted(output_ids),
            "completedCaseRuns": completed,
            "expectedCaseRuns": expected,
            "log": item.get("log"),
        }
        report["workers"][str(name)] = worker
        if completed < expected and not alive:
            dead_incomplete.append({"worker": name, **worker})
    return report, dead_incomplete


def inspect(output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    snapshot_path = output_root / "generation_config.snapshot.json"
    if not snapshot_path.is_file():
        raise FinalizerError(f"missing generation snapshot: {snapshot_path}")
    snapshot = read_json(snapshot_path)
    expected_outputs, excluded = expected_scope(config, snapshot)
    expected_runs = int(config["expectedRuns"])
    expected_rows = int(config["expectedRows"])
    expected_keys = {
        (output_id, run)
        for output_id in expected_outputs
        for run in range(1, expected_runs + 1)
    }
    manifests: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for path in sorted((output_root / "manifests").glob("*.json")):
        try:
            item = read_json(path)
            output_id = str(item.get("outputId"))
            source_id = str(item.get("sourceId"))
            run = int(item.get("run"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            errors.append({"manifest": str(path), "error": f"unreadable: {error}"})
            continue
        if source_id in excluded:
            errors.append({"manifest": str(path), "error": "excluded_source"})
            continue
        if expected_outputs.get(output_id) != source_id:
            errors.append({"manifest": str(path), "error": "outside_scope"})
            continue
        key = (output_id, run)
        if key not in expected_keys:
            errors.append({"manifest": str(path), "error": "unexpected_run"})
            continue
        if key in manifests:
            errors.append({"manifest": str(path), "error": "duplicate_case_run"})
            continue
        manifests[key] = (path, item)
        errors.extend(verify_manifest(path, item, expected_rows))
    complete_keys = {
        key
        for key, (_, item) in manifests.items()
        if item.get("status") in ALLOWED_STATUSES
    }
    missing = sorted(expected_keys - complete_keys)
    worker_state, dead_incomplete = worker_report(
        output_root, expected_outputs, complete_keys, expected_runs
    )
    complete = not missing and not errors and len(manifests) == len(expected_keys)
    status = "ready_to_evaluate" if complete else "generation_in_progress"
    if errors or dead_incomplete:
        status = "generation_blocked"
    return {
        "time": utc_now(),
        "protocol": "experiment6-no-gpt41-finalizer-v1",
        "status": status,
        "outputRoot": str(output_root),
        "expectedOfficialCases": len(expected_outputs),
        "expectedRuns": expected_runs,
        "expectedRows": expected_rows,
        "expectedCaseRuns": len(expected_keys),
        "completedCaseRuns": len(complete_keys),
        "remainingCaseRuns": len(missing),
        "formalPredictionsCompleted": len(complete_keys) * expected_rows,
        "expectedFormalPredictions": int(config["expectedFormalPredictions"]),
        "expectedOutputIds": sorted(expected_outputs),
        "missingCaseRuns": [
            {"outputId": output_id, "run": run} for output_id, run in missing
        ],
        "artifactErrors": errors,
        "workers": worker_state,
        "deadIncompleteWorkers": dead_incomplete,
        "generationSnapshot": str(snapshot_path),
        "generationSnapshotSha256": sha256_file(snapshot_path),
    }


def freeze_predictions(
    output_root: Path, config_path: Path, config: Mapping[str, Any], status: Mapping[str, Any]
) -> dict[str, Any]:
    freeze_root = output_root / "freeze_no_gpt41"
    freeze_manifest_path = freeze_root / "prediction_files.json"
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted((output_root / "manifests").glob("*.json")):
        manifest = read_json(manifest_path)
        entries.append({
            "role": "run_manifest",
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "outputId": manifest["outputId"],
            "run": manifest["run"],
        })
        for name, raw_path in sorted(manifest["files"].items()):
            path = Path(str(raw_path))
            entries.append({
                "role": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "outputId": manifest["outputId"],
                "run": manifest["run"],
            })
    freeze = {
        "time": utc_now(),
        "protocol": "experiment6-no-gpt41-prediction-freeze-v1",
        "outputRoot": str(output_root),
        "officialCases": int(config["expectedOfficialCases"]),
        "caseRuns": int(status["completedCaseRuns"]),
        "formalPredictions": int(status["formalPredictionsCompleted"]),
        "files": entries,
    }
    write_json(freeze_manifest_path, freeze)
    source_contract = output_root / "experiment6_v4_contract.json"
    amendment = {
        "time": utc_now(),
        "protocol": "experiment6-no-gpt41-scope-amendment-v1",
        "scope": {
            "includedParts": [1, 2, 3],
            "excludedPart": 4,
            "excludedModel": "GPT-4.1",
            "officialCases": int(config["expectedOfficialCases"]),
            "runsPerCase": int(config["expectedRuns"]),
            "rowsPerRun": int(config["expectedRows"]),
            "formalPredictions": int(config["expectedFormalPredictions"]),
        },
        "provenance": {
            "reusePolicy": "same-root-original-independent-calls-no-copy",
            "sourceGenerationSnapshot": status["generationSnapshot"],
            "sourceGenerationSnapshotSha256": status["generationSnapshotSha256"],
            "sourceContract": str(source_contract),
            "sourceContractSha256": (
                sha256_file(source_contract) if source_contract.is_file() else None
            ),
            "noGpt41Config": str(config_path),
            "noGpt41ConfigSha256": sha256_file(config_path),
            "noGpt41Evaluator": str(Path(evaluator.__file__).resolve()),
            "noGpt41EvaluatorSha256": sha256_file(Path(evaluator.__file__).resolve()),
            "predictionFreeze": str(freeze_manifest_path),
            "predictionFreezeSha256": sha256_file(freeze_manifest_path),
        },
        "expectedOutputIds": status["expectedOutputIds"],
    }
    amendment_path = output_root / "experiment6_no_gpt41_scope_amendment.json"
    write_json(amendment_path, amendment)
    return {
        "predictionFreeze": str(freeze_manifest_path),
        "predictionFreezeSha256": sha256_file(freeze_manifest_path),
        "scopeAmendment": str(amendment_path),
        "scopeAmendmentSha256": sha256_file(amendment_path),
    }


def run_evaluation(
    output_root: Path, config_path: Path, attempts: int
) -> tuple[bool, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(evaluator.__file__).resolve()),
                "--config",
                str(config_path),
                "--output-root",
                str(output_root),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        log_path = (
            output_root
            / "runtime"
            / f"no_gpt41_evaluation_attempt_{attempt:02d}.log"
        )
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        record = {
            "attempt": attempt,
            "returnCode": completed.returncode,
            "runtimeSeconds": time.monotonic() - started,
            "log": str(log_path),
            "logSha256": sha256_file(log_path),
        }
        records.append(record)
        if completed.returncode == 0:
            return True, records
        if attempt < attempts:
            time.sleep(60)
    return False, records


def verify_results(output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    results_path = output_root / "experiment6_results.json"
    tsv_path = output_root / "experiment6_results.tsv"
    per_run_path = output_root / "experiment6_per_run_scores.tsv"
    field_summary_path = output_root / "experiment6_field_summary.tsv"
    report_path = output_root / "evaluation_v4_no_gpt41" / "evaluation_report.json"
    for path in (
        results_path,
        tsv_path,
        per_run_path,
        field_summary_path,
        report_path,
    ):
        if not path.is_file():
            raise FinalizerError(f"missing final evaluation artifact: {path}")
    results = read_json(results_path)
    if not isinstance(results, list) or len(results) != int(config["expectedOfficialCases"]):
        raise FinalizerError("final result count mismatch")
    required = {"model", "prompt", "runtime", "output_file", "scores"}
    if any(set(item) != required for item in results):
        raise FinalizerError("final result interface mismatch")
    snapshot = read_json(output_root / "generation_config.snapshot.json")
    excluded = {str(value) for value in config.get("excludedSourceIds", [])}
    expected_outputs = set(
        evaluator.expected_output_sources(snapshot, excluded)
    )
    result_ids = {str(item["model"]["output_id"]) for item in results}
    if result_ids != expected_outputs:
        raise FinalizerError("final result output IDs mismatch")
    expected_run_ids = list(range(1, int(config["expectedRuns"]) + 1))
    for item in results:
        scores = item["scores"]
        if set(scores["fields"]) != set(evaluator.FIELDS):
            raise FinalizerError("final result field set mismatch")
        run_ids = sorted(int(run["run"]) for run in scores["runs"])
        if run_ids != expected_run_ids:
            raise FinalizerError("final result run coverage mismatch")
        if not item["output_file"]:
            raise FinalizerError("final result has no saved output paths")
        for field in evaluator.FIELDS:
            summary = scores["fields"][field]
            for group in ("mean", "top1", "top3"):
                for metric_name in ("precision", "recall", "f1"):
                    value = float(summary[group][metric_name])
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise FinalizerError("non-finite or out-of-range field score")
    if sum(1 for _ in per_run_path.open(encoding="utf-8")) - 1 != 340:
        raise FinalizerError("per-run score TSV row count mismatch")
    if sum(1 for _ in field_summary_path.open(encoding="utf-8")) - 1 != 204:
        raise FinalizerError("field-summary TSV row count mismatch")
    report = read_json(report_path)
    if report.get("status") != "completed":
        raise FinalizerError("evaluation report is not completed")
    if int(report.get("formalPredictions", -1)) != int(config["expectedFormalPredictions"]):
        raise FinalizerError("evaluation formal prediction count mismatch")
    return {
        "results": str(results_path),
        "resultsSha256": sha256_file(results_path),
        "tsv": str(tsv_path),
        "tsvSha256": sha256_file(tsv_path),
        "perRunScores": str(per_run_path),
        "perRunScoresSha256": sha256_file(per_run_path),
        "fieldSummary": str(field_summary_path),
        "fieldSummarySha256": sha256_file(field_summary_path),
        "report": str(report_path),
        "reportSha256": sha256_file(report_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--evaluation-attempts", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    status_path = output_root / "runtime" / "no_gpt41_finalizer_status.json"
    while True:
        try:
            status = inspect(output_root, config)
        except (FinalizerError, evaluator.ProtocolError, OSError, KeyError, ValueError) as error:
            write_json(status_path, {
                "time": utc_now(),
                "protocol": "experiment6-no-gpt41-finalizer-v1",
                "status": "finalizer_blocked",
                "error": str(error),
            })
            return 2
        write_json(status_path, status)
        if status["status"] == "generation_blocked":
            return 3
        if status["status"] == "ready_to_evaluate":
            frozen = freeze_predictions(output_root, config_path, config, status)
            status = {**status, "status": "evaluating", "freeze": frozen, "time": utc_now()}
            write_json(status_path, status)
            succeeded, attempts = run_evaluation(
                output_root, config_path, max(1, int(args.evaluation_attempts))
            )
            if not succeeded:
                write_json(status_path, {
                    **status,
                    "time": utc_now(),
                    "status": "evaluation_blocked",
                    "evaluationAttempts": attempts,
                })
                return 4
            outputs = verify_results(output_root, config)
            completed_status = {
                **status,
                "time": utc_now(),
                "status": "completed",
                "evaluationAttempts": attempts,
                "outputs": outputs,
            }
            write_json(status_path, completed_status)
            write_json(EXPERIMENT_ROOT / "experiment_6_no_gpt41_current.json", {
                "time": completed_status["time"],
                "status": "completed",
                "root": str(output_root),
                "results": outputs["results"],
                "report": outputs["report"],
            })
            return 0
        if args.once:
            return 0
        time.sleep(max(10, int(args.poll_seconds)))


if __name__ == "__main__":
    sys.exit(main())
