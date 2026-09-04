#!/usr/bin/env python3
"""Formal corrected-12 orchestration for Experiment 6 Narrative2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from experiment6_paths import PATHS


GENERATION_CONFIG = PATHS.resolve(
    "repo", "config/experiment6_narrative2_generation_corrected_12.json"
)
EVALUATION_CONFIG = PATHS.resolve(
    "repo", "config/experiment6_narrative2_evaluation_corrected_12.json"
)
GENERATION_RUNNER = PATHS.resolve(
    "dist", "run_experiment6_narrative2_generation.py"
)
JUDGE_BUILDER = PATHS.resolve("dist", "build_experiment6_judge_examples_v4.py")
EVALUATOR = PATHS.resolve(
    "dist", "evaluate_narrative2_reference_aligned_v5.py"
)
RECORDER = PATHS.resolve("dist", "record_experiment6_corrected12.py")
EXPECTED_CASES = 12
EXPECTED_RUNS = 10
EXPECTED_ROWS = 85


class OrchestrationError(RuntimeError):
    """Raised when the corrected formal workflow cannot safely continue."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_root() -> Path:
    return PATHS.resolve(
        "repo", f"Experiment/experiment_6_narrative2_corrected12_{utc_stamp()}"
    )


def run(command: Sequence[str]) -> None:
    completed = subprocess.run(command, cwd=PATHS.repo, check=False)
    if completed.returncode:
        raise OrchestrationError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def generation_command(args: argparse.Namespace, *, preflight_only: bool) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(GENERATION_RUNNER),
        "--config",
        str(GENERATION_CONFIG),
        "--output-root",
        str(args.output_root.resolve()),
        "--base-route-mode",
        "formal",
    ]
    if preflight_only:
        command.append("--preflight-only")
    if getattr(args, "no_resume", False):
        command.append("--no-resume")
    for output_id in getattr(args, "case", []):
        command.extend(["--case", output_id])
    for source_id in getattr(args, "source_id", []):
        command.extend(["--source-id", source_id])
    for run_number in getattr(args, "run", []):
        command.extend(["--run", str(run_number)])
    device = getattr(args, "cuda_visible_devices", None)
    if device is not None:
        command.extend(["--cuda-visible-devices", str(device)])
    return command


def load_manifests(root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((root / "manifests").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            manifests.append(value)
    return manifests


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_artifact(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise OrchestrationError(
                f"{path}: line {line_number} is not a JSON object"
            )
        records.append(value)
    return records


def validate_manifest_artifacts(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Validate the evidence needed before a corrected run can be evaluated."""
    errors: list[str] = []
    run_number = int(manifest.get("run") or 0)
    if int(manifest.get("expectedRows") or 0) != EXPECTED_ROWS:
        errors.append("expectedRows is not 85")
    expected_seed = 2026073100 + run_number
    if int(manifest.get("seed") or 0) != expected_seed:
        errors.append("seed does not follow 2026073101-2026073110")
    if int(manifest.get("runtimeBlockedRows") or 0) != 0:
        errors.append("runtimeBlockedRows is non-zero")
    if (
        int(manifest.get("acceptedRows") or 0)
        + int(manifest.get("rejectedRows") or 0)
        != EXPECTED_ROWS
    ):
        errors.append("acceptedRows + rejectedRows is not 85")
    if manifest.get("baseRouteMode") != "formal":
        errors.append("baseRouteMode is not formal")
    if manifest.get("declaredRoute") != "direct-binding":
        errors.append("declaredRoute is not direct-binding")
    if manifest.get("effectiveRoute") != "direct-binding":
        errors.append("effectiveRoute is not direct-binding")
    if manifest.get("adapter") is not None:
        errors.append("formal base direct-binding loaded an adapter")
    if manifest.get("converterModel") is not None:
        errors.append("formal base direct-binding used a converter")
    if manifest.get("requestedModel") != manifest.get("sourceId"):
        errors.append("requestedModel does not equal sourceId")
    resolved_source = manifest.get("resolvedSource")
    if not isinstance(resolved_source, dict):
        errors.append("resolvedSource is missing")
    elif manifest.get("actualModel") != resolved_source.get("baseModel"):
        errors.append("actualModel does not equal the registered baseModel")
    registry = manifest.get("sourceRegistry")
    if not isinstance(registry, dict) or not registry.get("sha256"):
        errors.append("sourceRegistry SHA-256 is missing")
    if not manifest.get("compatibilityFingerprint"):
        errors.append("compatibilityFingerprint is missing")
    topology = manifest.get("executionTopology")
    if not isinstance(topology, dict):
        errors.append("executionTopology is missing")
    else:
        if topology.get("cudaVisibleDevices") not in {"0", "1"}:
            errors.append("executionTopology device is not 0 or 1")
        if topology.get("status") != "resolved":
            errors.append("executionTopology GPU identity is unresolved")

    files = manifest.get("files")
    hashes = manifest.get("hashes")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        errors.append("artifact files/hashes are missing")
        return errors
    required = ("predictions", "rawResponse", "prompts", "runtime", "formatReport")
    records_by_name: dict[str, list[dict[str, Any]]] = {}
    root_resolved = root.resolve()
    for name in required:
        raw_path = files.get(name)
        if not raw_path:
            errors.append(f"{name} artifact path is missing")
            continue
        path = Path(str(raw_path)).resolve()
        if not path.is_relative_to(root_resolved):
            errors.append(f"{name} artifact is outside the corrected root")
            continue
        if not path.is_file():
            errors.append(f"{name} artifact does not exist")
            continue
        expected_hash = hashes.get(name)
        if not expected_hash or file_sha256(path) != expected_hash:
            errors.append(f"{name} SHA-256 mismatch")
            continue
        if name in {"predictions", "rawResponse", "prompts"}:
            try:
                records_by_name[name] = read_jsonl_artifact(path)
            except (OSError, json.JSONDecodeError, OrchestrationError) as exc:
                errors.append(f"{name} JSONL invalid: {exc}")

    stage_raw_names = [
        name for name in files if name.startswith("stage") and name.endswith("Raw")
    ]
    if not stage_raw_names:
        errors.append("native model raw artifact is missing")
    for name in stage_raw_names:
        path = Path(str(files[name])).resolve()
        if (
            not path.is_relative_to(root_resolved)
            or not path.is_file()
            or not hashes.get(name)
            or file_sha256(path) != hashes[name]
        ):
            errors.append(f"{name} raw artifact/hash is invalid")

    source_sets: dict[str, set[str]] = {}
    for name in ("predictions", "rawResponse", "prompts"):
        records = records_by_name.get(name)
        if records is None:
            continue
        if len(records) != EXPECTED_ROWS:
            errors.append(f"{name} does not contain 85 records")
        sources = [str(record.get("source") or "") for record in records]
        source_sets[name] = set(sources)
        if len(source_sets[name]) != EXPECTED_ROWS or "" in source_sets[name]:
            errors.append(f"{name} does not contain 85 unique Sources")
        indices = {record.get("index") for record in records}
        if indices != set(range(EXPECTED_ROWS)):
            errors.append(f"{name} indices are not exactly 0-84")
        if any(
            {"Binding_Result", "gold", "goldTargets"}.intersection(record)
            for record in records
        ):
            errors.append(f"{name} contains a forbidden gold field")
    if source_sets and len({frozenset(value) for value in source_sets.values()}) != 1:
        errors.append("prediction/raw/prompt Source sets differ")

    direct_hashes = {
        str(record.get("source")): record.get("directPromptSha256")
        for record in records_by_name.get("prompts", [])
    }
    for name in ("predictions", "rawResponse"):
        for record in records_by_name.get(name, []):
            source = str(record.get("source") or "")
            if record.get("promptSha256") != direct_hashes.get(source):
                errors.append(f"{name} prompt SHA-256 does not match prompts.jsonl")
                break
    return errors


def status(root: Path) -> dict[str, Any]:
    manifests = load_manifests(root)
    by_case: dict[str, list[int]] = defaultdict(list)
    predictions = 0
    fingerprints: set[str] = set()
    route_mismatches: list[str] = []
    artifact_errors: dict[str, list[str]] = {}
    for manifest in manifests:
        output_id = str(manifest.get("outputId") or "")
        run_number = int(manifest.get("run") or 0)
        by_case[output_id].append(run_number)
        predictions += int(manifest.get("expectedRows") or 0)
        fingerprint = manifest.get("compatibilityFingerprint")
        if fingerprint:
            fingerprints.add(str(fingerprint))
        declared = manifest.get("declaredRoute") or manifest.get("route")
        if declared != manifest.get("effectiveRoute"):
            route_mismatches.append(
                f"{output_id}/run_{run_number:02d}"
            )
        evidence_errors = validate_manifest_artifacts(root, manifest)
        if evidence_errors:
            artifact_errors[f"{output_id}/run_{run_number:02d}"] = evidence_errors
    expected_runs = list(range(1, EXPECTED_RUNS + 1))
    incomplete = {
        output_id: sorted(runs)
        for output_id, runs in sorted(by_case.items())
        if sorted(runs) != expected_runs
    }
    report = {
        "outputRoot": str(root),
        "manifests": len(manifests),
        "expectedManifests": EXPECTED_CASES * EXPECTED_RUNS,
        "predictionRows": predictions,
        "expectedPredictionRows": EXPECTED_CASES * EXPECTED_RUNS * EXPECTED_ROWS,
        "statusCounts": dict(Counter(str(item.get("status")) for item in manifests)),
        "cases": {key: sorted(value) for key, value in sorted(by_case.items())},
        "incomplete": incomplete,
        "compatibilityFingerprints": sorted(fingerprints),
        "routeMismatches": route_mismatches,
        "artifactErrors": artifact_errors,
    }
    report["complete"] = (
        len(manifests) == EXPECTED_CASES * EXPECTED_RUNS
        and len(by_case) == EXPECTED_CASES
        and not incomplete
        and len(fingerprints) == 1
        and not route_mismatches
        and not artifact_errors
        and all(
            item.get("status") in {"completed", "completed_with_format_errors"}
            for item in manifests
        )
    )
    return report


def write_inventory(root: Path) -> Path:
    excluded = {"sha256_inventory.tsv"}
    lines = ["sha256\tbytes\tpath"]
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or "/.archive_no_resume/" in f"/{relative}/":
            continue
        lines.append(f"{file_sha256(path)}\t{path.stat().st_size}\t{relative}")
    target = root / "sha256_inventory.tsv"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def command_preflight(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    run(generation_command(args, preflight_only=True))


def command_generate(args: argparse.Namespace) -> None:
    existing = load_manifests(args.output_root)
    case_selection = set(args.case)
    source_selection = set(args.source_id)
    run_selection = set(args.run)
    overlapping = [
        manifest
        for manifest in existing
        if (not case_selection or manifest.get("outputId") in case_selection)
        and (not source_selection or manifest.get("sourceId") in source_selection)
        and (not run_selection or manifest.get("run") in run_selection)
    ]
    if args.no_resume and overlapping:
        raise OrchestrationError(
            "fresh generation selection already contains manifests; choose a new root "
            "or a non-overlapping worker shard"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    run(generation_command(args, preflight_only=False))


def command_evaluate(args: argparse.Namespace) -> None:
    report = status(args.output_root)
    if not report["complete"]:
        raise OrchestrationError(
            "corrected-12 generation is incomplete; evaluation/ranking is withheld"
        )
    judge_dir = args.output_root / "judge_examples"
    run(
        [
            sys.executable,
            "-B",
            str(JUDGE_BUILDER),
            "--config",
            str(EVALUATION_CONFIG),
            "--output-dir",
            str(judge_dir),
        ]
    )
    run(
        [
            sys.executable,
            "-B",
            str(EVALUATOR),
            "--config",
            str(EVALUATION_CONFIG),
            "--output-root",
            str(args.output_root),
            "--evaluation-root",
            str(args.output_root / "evaluation_reference_aligned_v5"),
        ]
    )
    write_inventory(args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "generate"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--output-root", type=Path, default=default_root())
        subparser.add_argument("--case", action="append", default=[])
        subparser.add_argument("--source-id", action="append", default=[])
        subparser.add_argument("--run", type=int, action="append", default=[])
        subparser.add_argument(
            "--cuda-visible-devices", choices=("0", "1")
        )
        subparser.add_argument("--no-resume", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-root", type=Path, required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, required=True)

    inventory = subparsers.add_parser("inventory")
    record = subparsers.add_parser("record")
    record.add_argument("--output-root", type=Path, required=True)
    inventory.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        command_preflight(args)
    elif args.command == "generate":
        command_generate(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    elif args.command == "status":
        print(json.dumps(status(args.output_root.resolve()), ensure_ascii=False, indent=2))
    elif args.command == "inventory":
        print(write_inventory(args.output_root.resolve()))
    elif args.command == "record":
        run(
            [
                sys.executable,
                "-B",
                str(RECORDER),
                "--output-root",
                str(args.output_root.resolve()),
            ]
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
