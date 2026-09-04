#!/usr/bin/env python3
"""Inspect Experiment 6 narrative2 v2 progress without partial rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment6_narrative2_generation as generation  # noqa: E402

COMPLETED = {"completed", "completed_with_format_errors"}


class InspectionError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise InspectionError(f"cannot load {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise InspectionError(f"cannot read {path}: {error}") from error
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise InspectionError(f"invalid JSONL {path}:{number}: {error}") from error
        if not isinstance(value, dict):
            raise InspectionError(f"non-object JSONL record {path}:{number}")
        records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def artifact_path(value: Any, output_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else output_root / path


def inspect_completed(
    manifest: Mapping[str, Any],
    output_root: Path,
    expected_rows: int,
    converter_model: str,
    converter_effort: str,
) -> list[str]:
    errors: list[str] = []
    label = f"{manifest.get('outputId')}/run_{manifest.get('run')}"
    files = manifest.get("files")
    hashes = manifest.get("hashes")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        return [f"{label}: files or hashes is not an object"]
    for name, value in files.items():
        path = artifact_path(value, output_root)
        if not path.is_file():
            errors.append(f"{label}: missing {name}: {path}")
        elif hashes.get(name) and sha256_file(path) != hashes[name]:
            errors.append(f"{label}: SHA mismatch for {name}")

    predictions = read_jsonl(artifact_path(files["predictions"], output_root))
    if len(predictions) != expected_rows:
        errors.append(f"{label}: predictions={len(predictions)} expected={expected_rows}")

    if manifest.get("effectiveRoute") == "retriever-converter":
        candidates = read_jsonl(
            artifact_path(files.get("retrieverCandidates"), output_root)
        )
        converter = read_jsonl(
            artifact_path(files.get("converterRawResponses"), output_root)
        )
        if len(candidates) != expected_rows:
            errors.append(f"{label}: candidates={len(candidates)} expected={expected_rows}")
        if len(converter) != expected_rows:
            errors.append(f"{label}: converter={len(converter)} expected={expected_rows}")
        bad_identity = sum(
            record.get("actualModel") != converter_model
            or record.get("reasoningEffort") != converter_effort
            for record in converter
        )
        if bad_identity:
            errors.append(f"{label}: converter identity errors={bad_identity}")
    return errors


def checkpoint_counts(run_dir: Path) -> dict[str, int]:
    return {
        "candidateRows": len(read_jsonl(run_dir / "retriever_candidates.jsonl")),
        "converterRows": len(read_jsonl(run_dir / "converter_raw_responses.jsonl")),
        "predictionRows": len(read_jsonl(run_dir / "predictions.jsonl")),
        "runtimeBlockedRows": len(read_jsonl(run_dir / "runtime_blocked_rows.jsonl")),
    }


def summarize_case(
    case: generation.MatrixCase,
    runs: Sequence[int],
    manifests: Mapping[tuple[str, int], Mapping[str, Any]],
    output_root: Path,
    expected_rows: int,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    completed_runs: list[int] = []
    incomplete: list[dict[str, Any]] = []
    format_rates: list[float] = []
    actual_models: set[str] = set()
    adapters: set[str] = set()
    for run_number in runs:
        manifest = manifests.get((case.output_id, run_number))
        status = str(manifest.get("status") or "unknown") if manifest else "not_started"
        status_counts[status] += 1
        if manifest and status in COMPLETED:
            completed_runs.append(run_number)
            if manifest.get("formatComplianceRate") is not None:
                format_rates.append(float(manifest["formatComplianceRate"]))
            if manifest.get("actualModel"):
                actual_models.add(str(manifest["actualModel"]))
            if manifest.get("adapter"):
                adapters.add(str(manifest["adapter"]))
        else:
            counts = checkpoint_counts(
                output_root / "cases" / case.output_id / f"run_{run_number:02d}"
            )
            if any(counts.values()) or status != "not_started":
                incomplete.append({"run": run_number, "manifestStatus": status, **counts})
    return {
        "outputId": case.output_id,
        "sourceId": case.source_id,
        "part": case.part,
        "official": case.official,
        "route": case.route,
        "promptMode": case.prompt_mode,
        "completedRuns": completed_runs,
        "completedRunCount": len(completed_runs),
        "statusCounts": dict(sorted(status_counts.items())),
        "formalRowsCompleted": len(completed_runs) * expected_rows,
        "formatComplianceMean": (
            sum(format_rates) / len(format_rates) if format_rates else None
        ),
        "actualModels": sorted(actual_models),
        "adapters": sorted(adapters),
        "incompleteCheckpoints": incomplete,
    }


def markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    local = report["localRetrievers"]
    lines = [
        "# Experiment 6 narrative2 v2 progress and integrity",
        "",
        f"- Status: {report['status']}",
        f"- Ranking published: {str(report['rankingPublished']).lower()}",
        f"- Official cases complete: {coverage['officialCasesComplete']}/54",
        f"- Official case-runs complete: {coverage['officialCaseRunsComplete']}/540",
        f"- Formal predictions complete: {coverage['formalPredictionsComplete']}/45,900",
        f"- Controls complete: {coverage['controlCasesComplete']}/4",
        f"- Local retriever case-runs complete: {local['caseRunsComplete']}/300",
        f"- Local retriever predictions complete: {local['predictionsComplete']}/25,500",
        f"- Integrity errors: {len(report['integrityErrors'])}",
        "",
        "## Per-case progress",
        "",
        "| Output ID | Part | Route | Prompt | Runs | Status | Format |",
        "| --- | ---: | --- | --- | ---: | --- | ---: |",
    ]
    for item in report["cases"]:
        rate = (
            "—" if item["formatComplianceMean"] is None
            else f"{item['formatComplianceMean']:.4f}"
        )
        statuses = ", ".join(
            f"{key}:{value}" for key, value in item["statusCounts"].items()
        )
        lines.append(
            f"| {item['outputId']} | {item['part']} | {item['route']} | "
            f"{item['promptMode']} | {item['completedRunCount']}/10 | "
            f"{statuses} | {rate} |"
        )
    lines.extend([
        "",
        "## Evaluation contract",
        "",
        "- Alignment anchor: typed, case-sensitive exact DataName + Position identity; array order is preserved.",
        "- ObjectName: normalized exact one-to-one match, then blinded GPT-5.5 medium equivalence for unresolved terms.",
        "- Trend: versioned direction normalization, then blinded GPT adjudication requiring the same direction, period, baseline, and scope.",
        "- Num: deterministic one-to-one numeric comparison with rel_tol=abs_tol=1e-9 and percentage-point semantics; no LLM.",
        "- Text: normalized exact match, then blinded GPT adjudication requiring the same complete proposition, including entity, direction, number, time, scope, baseline, and negation.",
        "- Ranking remains withheld until every official case has 10 completed runs of 85 rows.",
        "",
    ])
    if report["integrityErrors"]:
        lines.extend(["## Integrity errors", ""])
        lines.extend(f"- {value}" for value in report["integrityErrors"])
        lines.append("")
    return "\n".join(lines)


def build(output_root: Path, config_path: Path) -> dict[str, Any]:
    config = generation.load_config(config_path)
    cases = generation.expand_matrix(config)
    runs = [int(value) for value in config["runs"]]
    expected_rows = int(config["expectedRows"])
    expected_keys = {
        (case.output_id, run_number) for case in cases for run_number in runs
    }
    manifests: dict[tuple[str, int], Mapping[str, Any]] = {}
    integrity_errors: list[str] = []
    for path in sorted((output_root / "manifests").glob("*.json")):
        manifest = read_json(path)
        if not isinstance(manifest, Mapping):
            integrity_errors.append(f"non-object manifest: {path}")
            continue
        if manifest.get("protocol") != config["protocol"]:
            continue
        key = (str(manifest.get("outputId")), int(manifest.get("run")))
        if key in manifests:
            integrity_errors.append(f"duplicate manifest key: {key}")
        manifests[key] = manifest
        if key not in expected_keys:
            integrity_errors.append(f"unexpected manifest key: {key}")
        if manifest.get("status") in COMPLETED:
            integrity_errors.extend(inspect_completed(
                manifest,
                output_root,
                expected_rows,
                str(config["converter"]["actualModelRequired"]),
                str(config["converter"]["reasoningEffort"]),
            ))

    rows = [
        summarize_case(case, runs, manifests, output_root, expected_rows)
        for case in cases
    ]
    official = [item for item in rows if item["official"]]
    controls = [item for item in rows if not item["official"]]
    local = [
        item for item in official
        if generation.family_for_source(item["sourceId"]) is not None
    ]
    official_runs = sum(item["completedRunCount"] for item in official)
    control_runs = sum(item["completedRunCount"] for item in controls)
    local_runs = sum(item["completedRunCount"] for item in local)
    official_cases = sum(item["completedRunCount"] == len(runs) for item in official)
    control_cases = sum(item["completedRunCount"] == len(runs) for item in controls)
    local_cases = sum(item["completedRunCount"] == len(runs) for item in local)
    matrix_complete = (
        official_cases == int(config["expectedOfficialCases"])
        and control_cases == int(config["expectedDiagnosticCases"])
        and not integrity_errors
    )
    evaluation_path = output_root / "evaluation" / "evaluation_report.json"
    evaluation = read_json(evaluation_path) if evaluation_path.is_file() else None
    report = {
        "time": utc_now(),
        "protocol": config["protocol"],
        "status": "completed_ready_for_ranking" if matrix_complete else "running_no_ranking",
        "rankingPublished": False,
        "outputRoot": str(output_root),
        "coverage": {
            "officialCasesComplete": official_cases,
            "officialCaseRunsComplete": official_runs,
            "formalPredictionsComplete": official_runs * expected_rows,
            "controlCasesComplete": control_cases,
            "controlCaseRunsComplete": control_runs,
            "controlPredictionsComplete": control_runs * expected_rows,
            "manifestCount": len(manifests),
            "expectedManifestCount": len(expected_keys),
        },
        "localRetrievers": {
            "caseCount": len(local),
            "casesComplete": local_cases,
            "caseRunsComplete": local_runs,
            "predictionsComplete": local_runs * expected_rows,
            "expectedCases": 30,
            "expectedCaseRuns": 300,
            "expectedPredictions": 25500,
        },
        "manifestStatusCounts": dict(sorted(Counter(
            str(value.get("status") or "unknown") for value in manifests.values()
        ).items())),
        "integrityErrors": integrity_errors,
        "cases": rows,
        "evaluationReport": {
            "path": str(evaluation_path),
            "present": evaluation is not None,
            "status": evaluation.get("status") if isinstance(evaluation, Mapping) else None,
            "rankingConsumed": False,
        },
        "artifacts": {
            "generationReport": str(output_root / "generation_report.json"),
            "generationConfig": str(config_path),
            "generationConfigSha256": sha256_file(config_path),
            "generationProgramSha256": sha256_file(
                Path(__file__).resolve().parent / "run_experiment6_narrative2_generation.py"
            ),
            "inspectionProgramSha256": sha256_file(Path(__file__).resolve()),
        },
    }
    diagnostics = output_root / "diagnostics"
    write_json(diagnostics / "progress_report.json", report)
    write_text(diagnostics / "progress_report.md", markdown(report))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect narrative2 Experiment 6 v2 progress without ranking."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "experiment6_narrative2_generation.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build(args.output_root.resolve(), args.config.resolve())
    except InspectionError as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, indent=2))
        return 2
    print(json.dumps({
        "status": report["status"],
        "rankingPublished": report["rankingPublished"],
        "coverage": report["coverage"],
        "localRetrievers": report["localRetrievers"],
        "integrityErrorCount": len(report["integrityErrors"]),
        "report": str(
            args.output_root.resolve() / "diagnostics" / "progress_report.json"
        ),
    }, ensure_ascii=False, indent=2))
    return 0 if not report["integrityErrors"] else 2


if __name__ == "__main__":
    sys.exit(main())
