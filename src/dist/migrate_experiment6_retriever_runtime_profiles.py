#!/usr/bin/env python3
"""Backfill auditable retriever batch/runtime profiles without changing predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment6_narrative2_generation as generation  # noqa: E402


class MigrationError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        for value in values
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def infer_batch_size(
    raw: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    configured_default: int,
) -> tuple[int, list[str]]:
    observed: list[tuple[int, str]] = []
    if raw.get("batch_size") is not None:
        observed.append((int(raw["batch_size"]), "runtime.raw.batch_size"))
    candidate_values = {
        int(item["batchSize"])
        for item in candidates
        if item.get("batchSize") is not None
    }
    if len(candidate_values) > 1:
        raise MigrationError(f"candidate batchSize values disagree: {sorted(candidate_values)}")
    if candidate_values:
        observed.append((next(iter(candidate_values)), "retriever_candidates.batchSize"))
    log_value = raw.get("log")
    if log_value and Path(str(log_value)).is_file():
        first_line = Path(str(log_value)).read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        if first_line:
            match = re.search(r"(?:^|\s)--batch-size\s+(\d+)(?:\s|$)", first_line[0])
            if match:
                observed.append((int(match.group(1)), "retriever command log"))
    if not observed:
        observed.append((configured_default, "formal config default"))
    values = {item[0] for item in observed}
    if len(values) != 1:
        raise MigrationError(f"batch evidence disagrees: {observed}")
    return next(iter(values)), [item[1] for item in observed]


def backfill_converter_runs(
    converters: Sequence[dict[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    run: int,
) -> None:
    if len(converters) != len(candidates):
        raise MigrationError("converter/candidate row counts disagree")
    for index, (converter, candidate) in enumerate(zip(converters, candidates)):
        if (
            converter.get("index") != index
            or converter.get("source") != candidate.get("source")
            or converter.get("seed") != candidate.get("seed") + index
            or converter.get("candidateSha256") != candidate.get("candidateSha256")
        ):
            raise MigrationError(f"converter provenance mismatch at row {index}")
        observed_run = converter.get("run")
        if observed_run is not None and observed_run != run:
            raise MigrationError(
                f"converter run mismatch at row {index}: {observed_run} != {run}"
            )
        converter["run"] = run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "experiment6_narrative2_generation.json",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    config = generation.load_config(args.config.resolve())
    cases = [
        case
        for case in generation.expand_matrix(config)
        if generation.family_for_source(case.source_id) is not None
    ]
    if len(cases) != 30:
        raise MigrationError(f"retriever case count={len(cases)}, expected 30")
    run_numbers = [int(value) for value in config["runs"]]
    if run_numbers != list(range(1, 11)):
        raise MigrationError(f"formal runs={run_numbers}, expected 1..10")
    expected_runs = len(cases) * len(run_numbers)
    plans: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in cases:
        family = generation.family_for_source(case.source_id)
        assert family is not None
        for run in run_numbers:
            manifest_path = output_root / "manifests" / f"{case.output_id}__run_{run:02d}.json"
            if not manifest_path.is_file():
                missing.append(f"{case.output_id}/run_{run:02d}")
                continue
            manifest = generation.read_json(manifest_path)
            files = manifest.get("files") or {}
            hashes = manifest.get("hashes") or {}
            paths = {
                name: Path(str(files[name]))
                for name in (
                    "predictions", "runtime", "retrieverCandidates",
                    "converterRawResponses",
                )
            }
            status_path = output_root / "cases" / case.output_id / f"run_{run:02d}" / "status.json"
            paths["manifest"] = manifest_path
            paths["status"] = status_path
            for name in (
                "predictions", "runtime", "retrieverCandidates",
                "converterRawResponses",
            ):
                if not paths[name].is_file():
                    raise MigrationError(f"missing {name}: {paths[name]}")
                if hashes.get(name) != generation.sha256_file(paths[name]):
                    raise MigrationError(f"pre-migration SHA mismatch: {paths[name]}")
            predictions = read_jsonl(paths["predictions"])
            runtime = generation.read_json(paths["runtime"])
            candidates = read_jsonl(paths["retrieverCandidates"])
            converters = read_jsonl(paths["converterRawResponses"])
            if (
                len(predictions) != 85
                or len(candidates) != 85
                or len(converters) != 85
            ):
                raise MigrationError(f"row count mismatch: {case.output_id}/run_{run:02d}")
            stages = runtime.get("stages") or []
            if len(stages) < 2 or not isinstance(stages[0], dict):
                raise MigrationError(f"runtime stages missing: {case.output_id}/run_{run:02d}")
            raw = stages[0].get("raw")
            if not isinstance(raw, dict):
                raise MigrationError(f"retriever raw runtime missing: {case.output_id}/run_{run:02d}")
            batch_size, evidence = infer_batch_size(raw, candidates, int(config["retriever"]["batchSize"]))
            profile = f"{family}-canonical-batch{batch_size}"
            for index, prediction in enumerate(predictions):
                if prediction.get("index") != index or prediction.get("run") != run:
                    raise MigrationError(f"prediction provenance mismatch: {case.output_id}/run_{run:02d}/{index}")
                prediction["runtimeProfile"] = profile
            for index, candidate in enumerate(candidates):
                if candidate.get("index") != index or candidate.get("run") != run:
                    raise MigrationError(f"candidate provenance mismatch: {case.output_id}/run_{run:02d}/{index}")
                candidate["batchSize"] = batch_size
            backfill_converter_runs(converters, candidates, run)
            stages[0]["runtimeProfile"] = profile
            raw["batch_size"] = batch_size
            raw["runtime_profile"] = profile
            runtime["stages"] = stages
            prediction_bytes = jsonl_bytes(predictions)
            runtime_bytes = json_bytes(runtime)
            candidate_bytes = jsonl_bytes(candidates)
            converter_bytes = jsonl_bytes(converters)
            manifest["runtimeProfile"] = profile
            manifest["hashes"]["predictions"] = sha256_bytes(prediction_bytes)
            manifest["hashes"]["runtime"] = sha256_bytes(runtime_bytes)
            manifest["hashes"]["retrieverCandidates"] = sha256_bytes(candidate_bytes)
            manifest["hashes"]["converterRawResponses"] = sha256_bytes(converter_bytes)
            manifest_bytes = json_bytes(manifest)
            plans.append({
                "outputId": case.output_id,
                "run": run,
                "family": family,
                "batchSize": batch_size,
                "runtimeProfile": profile,
                "batchEvidence": evidence,
                "paths": paths,
                "bytes": {
                    "predictions": prediction_bytes,
                    "runtime": runtime_bytes,
                    "retrieverCandidates": candidate_bytes,
                    "converterRawResponses": converter_bytes,
                    "manifest": manifest_bytes,
                    "status": manifest_bytes,
                },
            })
    if args.require_complete and missing:
        raise MigrationError(f"missing {len(missing)}/{expected_runs} runs; first={missing[:3]}")
    report = {
        "status": "ready" if not args.apply else "completed_metadata_only",
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outputRoot": str(output_root),
        "plannedRuns": len(plans),
        "expectedRuns": expected_runs,
        "missingRuns": missing,
        "predictionOrCandidateTextChanged": False,
        "converterRawResponseTextChanged": False,
        "fieldsBackfilled": ["predictions.runtimeProfile", "runtime.stages[0].runtimeProfile", "runtime.stages[0].raw.batch_size", "runtime.stages[0].raw.runtime_profile", "retrieverCandidates.batchSize", "converterRawResponses.run", "manifest.runtimeProfile"],
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_root = output_root / "diagnostics" / "provenance" / f"retriever_runtime_profile_{stamp}"
    originals = audit_root / "originals"
    for plan in plans:
        for name, path in plan["paths"].items():
            relative = path.resolve().relative_to(output_root)
            backup = originals / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        for name, value in plan["bytes"].items():
            atomic_write(plan["paths"][name], value)
    report["originalsPreserved"] = True
    report["originalsRoot"] = str(originals)
    report["runs"] = [{key: plan[key] for key in ("outputId", "run", "family", "batchSize", "runtimeProfile", "batchEvidence")} for plan in plans]
    generation.write_json(audit_root / "audit_report.json", report)
    print(json.dumps({**report, "runs": f"{len(plans)} records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
