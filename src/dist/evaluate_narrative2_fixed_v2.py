#!/usr/bin/env python3
"""Experiment 6 fixed six-field evaluator and ten-run aggregator.

The protected narrative evaluator remains the comparison authority.  This
wrapper verifies that bundle, applies its exact field semantics, extends the
run contract from six to ten, and scores gate-rejected rows as zero over their
target binding fields so invalid output cannot shrink the denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiment6_paths import PATHS


REPO_ROOT = PATHS.repo
WORKSPACE_ROOT = PATHS.workspace
SCRIPTS_ROOT = PATHS.dist
sys.path.insert(0, str(SCRIPTS_ROOT))

FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
PRESENCE_FIELDS = ("Trend", "Num")
TARGET_FIELDS = ("ObjectName", "Trend", "Num")
MODE_SUFFIX = {"zero-shot": "z", "many-shot": "m", "dynamic-shot": "d"}
TERMINAL_GENERATION_STATUSES = {
    "completed",
    "completed_with_format_errors",
    "runtime_blocked",
}


class ProtocolError(RuntimeError):
    """Raised when an artifact violates the fixed-v2 contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def workspace_path(raw: str | Mapping[str, Any]) -> Path:
    if isinstance(raw, Mapping):
        return PATHS.resolve_locator(raw)
    path = Path(raw)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise ProtocolError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON in {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value, ensure_ascii=False, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
    temporary.replace(path)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest(bundle: Path, expected_manifest_sha: str) -> dict[str, Any]:
    manifest_path = bundle / "bundle_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ProtocolError("evaluation bundle manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ProtocolError("evaluation bundle manifest has no files[]")
    checked = []
    for entry in entries:
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolError(f"unsafe evaluation bundle path: {relative}")
        target = bundle / relative
        if not target.is_file():
            raise ProtocolError(f"evaluation bundle file missing: {relative}")
        actual_sha = sha256_file(target)
        actual_bytes = target.stat().st_size
        if actual_sha != entry.get("sha256") or actual_bytes != entry.get("bytes"):
            raise ProtocolError(f"evaluation bundle manifest mismatch: {relative}")
        checked.append(
            {
                "path": relative.as_posix(),
                "sha256": actual_sha,
                "bytes": actual_bytes,
            }
        )
    return {
        "manifest": str(manifest_path),
        "manifestSha256": expected_manifest_sha,
        "filesChecked": len(checked),
        "files": checked,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if (
        not isinstance(config, dict)
        or config.get("protocol") != "narrative2-fixed-python-v2"
    ):
        raise ProtocolError("fixed evaluation config has the wrong protocol")
    if config.get("rejectedPolicy") != "zero_target_binding_fields":
        raise ProtocolError("fixed-v2 requires zero_target_binding_fields")
    if list(config.get("fields") or []) != list(FIELDS):
        raise ProtocolError("fixed-v2 field list does not match narrative evaluator")
    if int(config.get("expectedRuns", 0)) != 10:
        raise ProtocolError("fixed-v2 formal expectedRuns must be 10")
    return config


def expand_matrix(generation_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for part_spec in generation_config["parts"]:
        part = int(part_spec["part"])
        route = str(part_spec["route"])
        if "models" in part_spec:
            for model in part_spec["models"]:
                for mode in part_spec["promptModes"]:
                    cases.append(
                        {
                            "outputId": f"{model['outputStem']}_{MODE_SUFFIX[mode]}",
                            "sourceId": str(model["sourceId"]),
                            "promptMode": str(mode),
                            "route": route,
                            "part": part,
                            "official": True,
                        }
                    )
        else:
            for item in part_spec["cases"]:
                cases.append(
                    {
                        "outputId": str(item["outputId"]),
                        "sourceId": str(item["sourceId"]),
                        "promptMode": str(
                            item.get("promptMode") or part_spec.get("promptMode")
                        ),
                        "route": str(item.get("route") or route),
                        "part": part,
                        "official": True,
                    }
                )
    for item in generation_config.get("controls", []):
        cases.append(
            {
                "outputId": str(item["outputId"]),
                "sourceId": str(item["sourceId"]),
                "promptMode": str(item["promptMode"]),
                "route": str(item["route"]),
                "part": 0,
                "official": False,
            }
        )
    output_ids = [item["outputId"] for item in cases]
    if len(output_ids) != len(set(output_ids)):
        raise ProtocolError("generation matrix has duplicate outputId values")
    official = [item for item in cases if item["official"]]
    expected = int(generation_config["expectedOfficialCases"])
    if len(official) != expected:
        raise ProtocolError(
            f"generation matrix has {len(official)} official cases, expected {expected}"
        )
    return cases


def new_counts() -> dict[str, dict[str, int]]:
    return {field: {"passes": 0, "tested": 0} for field in FIELDS}


def add_count(
    counts: dict[str, dict[str, int]], field: str, passed: bool
) -> None:
    counts[field]["tested"] += 1
    if passed:
        counts[field]["passes"] += 1


def merge_counts(
    destination: dict[str, dict[str, int]],
    source: Mapping[str, Mapping[str, int]],
) -> None:
    for field in FIELDS:
        destination[field]["passes"] += int(source[field]["passes"])
        destination[field]["tested"] += int(source[field]["tested"])


def metrics_for_counts(fixed: Any, counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    return {
        field: fixed.exact_metrics(
            int(counts[field]["passes"]), int(counts[field]["tested"])
        )
        for field in FIELDS
    }


def combined_metric(fixed: Any, counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    passes = sum(int(counts[field]["passes"]) for field in FIELDS)
    tested = sum(int(counts[field]["tested"]) for field in FIELDS)
    return fixed.exact_metrics(passes, tested)


def field_pass(fixed: Any, baseline: Mapping[str, Any], goal: Mapping[str, Any],
               generated: Mapping[str, Any], field: str) -> bool:
    if field not in PRESENCE_FIELDS:
        return fixed.same_fixed(generated[field], goal[field])
    expected_present = not fixed.is_absent(baseline[field])
    generated_present = not fixed.is_absent(generated[field])
    if expected_present:
        return generated_present and fixed.same_fixed(generated[field], goal[field])
    return not generated_present


def load_run_records(
    fixed: Any,
    predictions_path: Path,
    targets: Sequence[Mapping[str, Any]],
    run_number: int,
    expected_rows: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    raw_records, parse_rejected = fixed.load_jsonl(predictions_path)
    accepted, gate_rejected = fixed._prepare_records(raw_records, list(targets))
    rejected = list(parse_rejected) + list(gate_rejected)

    wrong_run_sources: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for record in accepted:
        if record["run"] != run_number:
            wrong_run_sources.add(record["source"])
            rejected.append(
                {
                    "line": None,
                    "source": record["source"],
                    "run": record["run"],
                    "errors": [
                        f"prediction run must equal directory run {run_number}"
                    ],
                }
            )
        else:
            filtered.append(record)

    rejected_sources = {
        str(item["source"])
        for item in rejected
        if isinstance(item.get("source"), str)
    } | wrong_run_sources
    accepted_by_source = {
        record["source"]: record
        for record in filtered
        if record["source"] not in rejected_sources
    }
    target_sources = [str(target["source"]) for target in targets]
    for source in target_sources:
        if source not in accepted_by_source and source not in rejected_sources:
            rejected.append(
                {
                    "line": None,
                    "source": source,
                    "run": run_number,
                    "errors": ["missing prediction row"],
                }
            )
            rejected_sources.add(source)
    for source in list(accepted_by_source):
        if source in rejected_sources:
            del accepted_by_source[source]

    physical_rows = len(raw_records) + len(parse_rejected)
    if physical_rows != expected_rows:
        rejected.append(
            {
                "line": None,
                "source": None,
                "run": run_number,
                "errors": [
                    f"JSONL row count mismatch: expected {expected_rows}, "
                    f"received {physical_rows}"
                ],
            }
        )
    return accepted_by_source, rejected, physical_rows


def legacy_metrics(
    legacy: Any,
    targets: Sequence[Mapping[str, Any]],
    accepted_by_source: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gold_rows = [
        {"case_id": target["source"], "result": target["targetBindings"]}
        for target in targets
    ]
    pred_rows = [
        {
            "case_id": target["source"],
            "result": (
                accepted_by_source[target["source"]]["result"]
                if target["source"] in accepted_by_source
                else []
            ),
        }
        for target in targets
    ]
    gold = legacy.extract_rows(gold_rows, strict=True)
    pred = legacy.extract_rows(pred_rows, strict=False)
    return legacy.metrics_from_extracted(
        gold, pred, ["subject", "trend", "numerical"]
    )


def evaluate_run(
    fixed: Any,
    legacy: Any,
    config: Mapping[str, Any],
    output_root: Path,
    case: Mapping[str, Any],
    run_number: int,
    targets: Sequence[Mapping[str, Any]],
    vocabulary: set[str],
) -> dict[str, Any]:
    output_id = str(case["outputId"])
    run_dir = output_root / "cases" / output_id / f"run_{run_number:02d}"
    status_path = run_dir / "status.json"
    predictions_path = run_dir / "predictions.jsonl"
    if not status_path.is_file() or not predictions_path.is_file():
        raise ProtocolError(
            f"{output_id} run {run_number} is missing status or predictions"
        )
    generation_status = read_json(status_path)
    if generation_status.get("status") not in TERMINAL_GENERATION_STATUSES:
        raise ProtocolError(
            f"{output_id} run {run_number} is not terminal: "
            f"{generation_status.get('status')}"
        )
    if generation_status.get("outputId") != output_id:
        raise ProtocolError(f"{status_path} outputId mismatch")
    if int(generation_status.get("run", 0)) != run_number:
        raise ProtocolError(f"{status_path} run mismatch")
    expected_hash = generation_status.get("hashes", {}).get("predictions")
    if expected_hash and sha256_file(predictions_path) != expected_hash:
        raise ProtocolError(f"{predictions_path} SHA-256 mismatch")

    accepted_by_source, rejected, physical_rows = load_run_records(
        fixed,
        predictions_path,
        targets,
        run_number,
        int(config["expectedRows"]),
    )
    counts = new_counts()
    row_details: list[dict[str, Any]] = []
    object_structure_passes = 0
    object_structure_tested = 0
    num_type_passes = 0
    num_type_tested = 0
    present_trends = 0
    present_trends_in_vocab = 0
    rejected_by_source: dict[str, list[str]] = {}
    for item in rejected:
        source = item.get("source")
        if isinstance(source, str):
            rejected_by_source.setdefault(source, []).extend(item.get("errors", []))

    for target in targets:
        source = str(target["source"])
        record = accepted_by_source.get(source)
        row_counts = new_counts()
        binding_details: list[dict[str, Any]] = []
        for binding_index in range(int(target["bindingCount"])):
            baseline = target["baselineBindings"][binding_index]
            goal = target["targetBindings"][binding_index]
            generated = (
                record["result"][binding_index] if record is not None else None
            )
            passes: dict[str, bool] = {}
            for field in FIELDS:
                passed = (
                    field_pass(fixed, baseline, goal, generated, field)
                    if generated is not None
                    else False
                )
                passes[field] = passed
                add_count(row_counts, field, passed)
                add_count(counts, field, passed)
            if generated is not None:
                object_structure_tested += 1
                num_type_tested += 1
                if fixed.non_empty_string_array(generated["ObjectName"]):
                    object_structure_passes += 1
                if fixed.finite_numeric_array(generated["Num"]):
                    num_type_passes += 1
                if not fixed.is_absent(generated["Trend"]):
                    present_trends += 1
                    if (
                        isinstance(generated["Trend"], str)
                        and generated["Trend"].strip().lower() in vocabulary
                    ):
                        present_trends_in_vocab += 1
            binding_details.append(
                {
                    "binding": binding_index,
                    "accepted": generated is not None,
                    "passes": passes,
                    "allFields": fixed.exact_metrics(
                        sum(passes.values()), len(FIELDS)
                    ),
                }
            )
        row_details.append(
            {
                "source": source,
                "excelRow": target.get("excelRow"),
                "accepted": record is not None,
                "rejectionErrors": rejected_by_source.get(source, []),
                "fields": metrics_for_counts(fixed, row_counts),
                "allFields": combined_metric(fixed, row_counts),
                "bindings": binding_details,
            }
        )

    expected_bindings = sum(int(target["bindingCount"]) for target in targets)
    if expected_bindings != int(config["expectedBindingsPerRun"]):
        raise ProtocolError("gold binding total differs from fixed-v2 config")
    field_metrics = metrics_for_counts(fixed, counts)
    all_fields = combined_metric(fixed, counts)
    if all_fields["tested"] != expected_bindings * len(FIELDS):
        raise ProtocolError("rejected-zero denominator is incomplete")
    legacy_result = legacy_metrics(legacy, targets, accepted_by_source)
    protocol_acceptance_rate = (
        len(accepted_by_source) / int(config["expectedRows"])
    )
    format_rate = float(
        generation_status.get("formatComplianceRate")
        if generation_status.get("formatComplianceRate") is not None
        else protocol_acceptance_rate
    )
    runtime_blocked_rows = int(generation_status.get("runtimeBlockedRows") or 0)
    evaluation_status = (
        "runtime_blocked_scored_zero"
        if runtime_blocked_rows
        else (
            "completed_with_rejected_zero"
            if rejected
            else "completed"
        )
    )
    metrics = {
        "protocol": config["protocol"],
        "outputId": output_id,
        "run": run_number,
        "status": evaluation_status,
        "generationStatus": generation_status.get("status"),
        "expectedRows": int(config["expectedRows"]),
        "physicalPredictionRows": physical_rows,
        "acceptedRows": len(accepted_by_source),
        "rejectedRows": int(config["expectedRows"]) - len(accepted_by_source),
        "runtimeBlockedRows": runtime_blocked_rows,
        "formatComplianceRate": format_rate,
        "fixedProtocolAcceptanceRate": protocol_acceptance_rate,
        "rejectedPolicy": config["rejectedPolicy"],
        "fields": field_metrics,
        "allFieldsCombined": all_fields,
        "legacyOTN": legacy_result,
        "objectNameNonEmptyStringArray": fixed.fraction(
            object_structure_passes, object_structure_tested
        ),
        "numFiniteNumericArray": fixed.fraction(
            num_type_passes, num_type_tested
        ),
        "trendVocabularyAmongPresentTrends": fixed.fraction(
            present_trends_in_vocab, present_trends
        ),
    }

    evaluation_dir = run_dir / "evaluation_fixed_v2"
    pred_extracted = []
    for target in targets:
        source = str(target["source"])
        record = accepted_by_source.get(source)
        pred_extracted.append(
            {
                "source": source,
                "run": run_number,
                "accepted": record is not None,
                "result": record["result"] if record is not None else [],
                "rejectionErrors": rejected_by_source.get(source, []),
            }
        )
    write_json(evaluation_dir / "metrics.json", metrics)
    write_json(
        evaluation_dir / "evaluation.json",
        {
            "generatedAt": utc_now(),
            "protocol": config["protocol"],
            "outputId": output_id,
            "run": run_number,
            "summary": metrics,
            "rows": row_details,
            "rejected": rejected,
        },
    )
    write_jsonl(
        evaluation_dir / "gold_extracted.jsonl",
        (
            {
                "source": target["source"],
                "baselineBindings": target["baselineBindings"],
                "targetBindings": target["targetBindings"],
            }
            for target in targets
        ),
    )
    write_jsonl(evaluation_dir / "pred_extracted.jsonl", pred_extracted)
    write_jsonl(evaluation_dir / "rejected_records.jsonl", rejected)
    write_json(
        evaluation_dir / "extraction_report.json",
        {
            "protocol": config["protocol"],
            "predictionPath": str(predictions_path),
            "predictionSha256": sha256_file(predictions_path),
            "goldRows": len(targets),
            "goldBindings": expected_bindings,
            "physicalPredictionRows": physical_rows,
            "acceptedRows": len(accepted_by_source),
            "rejectedRows": int(config["expectedRows"]) - len(accepted_by_source),
            "unmappedRejectedRecords": sum(
                not isinstance(item.get("source"), str) for item in rejected
            ),
        },
    )
    write_json(
        evaluation_dir / "status.json",
        {
            "generatedAt": utc_now(),
            "protocol": config["protocol"],
            "status": evaluation_status,
            "outputId": output_id,
            "run": run_number,
            "rankingEligible": runtime_blocked_rows == 0,
            "metrics": str(evaluation_dir / "metrics.json"),
        },
    )
    (evaluation_dir / "evaluation_summary.md").write_text(
        "\n".join(
            [
                f"# {output_id} run {run_number:02d} fixed-v2",
                "",
                f"- status: {evaluation_status}",
                f"- accepted/rejected: {len(accepted_by_source)}/"
                f"{int(config['expectedRows']) - len(accepted_by_source)}",
                f"- format compliance: {format_rate:.6f}",
                f"- fixed protocol acceptance: {protocol_acceptance_rate:.6f}",
                f"- allFieldsCombined F1: {all_fields['f1']:.6f}",
                f"- runtime-blocked rows: {runtime_blocked_rows}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "rows": row_details,
        "generationStatus": generation_status,
        "files": [
            str(predictions_path),
            str(status_path),
            str(evaluation_dir / "metrics.json"),
            str(evaluation_dir / "evaluation.json"),
            str(evaluation_dir / "evaluation_summary.md"),
            str(evaluation_dir / "rejected_records.jsonl"),
            str(evaluation_dir / "gold_extracted.jsonl"),
            str(evaluation_dir / "pred_extracted.jsonl"),
            str(evaluation_dir / "extraction_report.json"),
            str(evaluation_dir / "status.json"),
        ],
    }


def score_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "sampleSd": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    return {
        "mean": statistics.fmean(values),
        "sampleSd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def aggregate_case(
    fixed: Any,
    config: Mapping[str, Any],
    output_root: Path,
    case: Mapping[str, Any],
    run_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(run_results, key=lambda item: item["metrics"]["run"])
    expected_runs = int(config["expectedRuns"])
    if [item["metrics"]["run"] for item in ordered] != list(
        range(1, expected_runs + 1)
    ):
        raise ProtocolError(f"{case['outputId']} does not have runs 1-{expected_runs}")
    run_scores = [
        {
            "run": item["metrics"]["run"],
            "status": item["metrics"]["status"],
            "generationStatus": item["metrics"]["generationStatus"],
            "allFieldsCombined": item["metrics"]["allFieldsCombined"],
            "fields": item["metrics"]["fields"],
            "legacyOTN": item["metrics"]["legacyOTN"],
            "formatComplianceRate": item["metrics"]["formatComplianceRate"],
            "fixedProtocolAcceptanceRate": item["metrics"][
                "fixedProtocolAcceptanceRate"
            ],
            "acceptedRows": item["metrics"]["acceptedRows"],
            "rejectedRows": item["metrics"]["rejectedRows"],
            "runtimeBlockedRows": item["metrics"]["runtimeBlockedRows"],
        }
        for item in ordered
    ]
    ranked = sorted(
        run_scores,
        key=lambda item: (
            float(item["allFieldsCombined"]["f1"]),
            float(item["allFieldsCombined"]["precision"]),
            float(item["allFieldsCombined"]["recall"]),
            -int(item["run"]),
        ),
        reverse=True,
    )
    top_runs = ranked[: int(config["topK"])]
    aggregate_counts = new_counts()
    for item in ordered:
        for field in FIELDS:
            metric = item["metrics"]["fields"][field]
            aggregate_counts[field]["passes"] += int(metric["passes"])
            aggregate_counts[field]["tested"] += int(metric["tested"])

    first_status = ordered[0]["generationStatus"]
    prompt_mode = str(case["promptMode"])
    bundle_dir = {
        "original": "original",
        "zero-shot": "zero",
        "many-shot": "many",
        "dynamic-shot": "dynamic",
    }[prompt_mode]
    prompt_manifest = (
        output_root / "input_bundles" / bundle_dir / "bundle_manifest.json"
    )
    any_runtime_block = any(
        int(item["metrics"]["runtimeBlockedRows"]) > 0 for item in ordered
    )
    all_10 = {
        "allFieldsCombined": {
            metric: score_stats(
                [
                    float(item["allFieldsCombined"][metric])
                    for item in run_scores
                ]
            )
            for metric in ("precision", "recall", "f1", "exactMatchRate")
        },
        "fields": {
            field: {
                metric: score_stats(
                    [
                        float(item["fields"][field][metric])
                        for item in run_scores
                    ]
                )
                for metric in ("precision", "recall", "f1", "exactMatchRate")
            }
            for field in FIELDS
        },
        "formatComplianceRate": score_stats(
            [float(item["formatComplianceRate"]) for item in run_scores]
        ),
        "fixedProtocolAcceptanceRate": score_stats(
            [
                float(item["fixedProtocolAcceptanceRate"])
                for item in run_scores
            ]
        ),
        "microAcrossRuns": {
            "fields": metrics_for_counts(fixed, aggregate_counts),
            "allFieldsCombined": combined_metric(fixed, aggregate_counts),
        },
    }
    top_3 = {
        "runs": [int(item["run"]) for item in top_runs],
        "allFieldsCombined": {
            metric: score_stats(
                [
                    float(item["allFieldsCombined"][metric])
                    for item in top_runs
                ]
            )
            for metric in ("precision", "recall", "f1", "exactMatchRate")
        },
    }
    legacy_stats = {
        field: score_stats(
            [
                float(item["legacyOTN"]["by_field"][field]["f1"])
                for item in run_scores
            ]
        )
        for field in ("ObjectName", "Trend", "Num")
    }
    output_files = sorted(
        {path for item in ordered for path in item["files"]}
    )
    return {
        "model": {
            "output_id": case["outputId"],
            "source_id": case["sourceId"],
            "requested": first_status.get("requestedModel"),
            "actual": sorted(
                {
                    str(item["generationStatus"].get("actualModel"))
                    for item in ordered
                }
            ),
            "adapter": first_status.get("adapter"),
            "route": case["route"],
            "part": case["part"],
        },
        "prompt": {
            "mode": prompt_mode,
            "bundleManifest": str(prompt_manifest),
            "bundleManifestSha256": (
                sha256_file(prompt_manifest) if prompt_manifest.is_file() else None
            ),
        },
        "runtime": {
            "runs": expected_runs,
            "runtimeSecondsTotal": sum(
                float(item["generationStatus"].get("runtimeSeconds") or 0)
                for item in ordered
            ),
            "runtimeProfiles": sorted(
                {
                    str(item["generationStatus"].get("runtimeProfile"))
                    for item in ordered
                }
            ),
            "quantizations": sorted(
                {
                    str(item["generationStatus"].get("quantization"))
                    for item in ordered
                }
            ),
            "runtimeBlockedRows": sum(
                int(item["metrics"]["runtimeBlockedRows"]) for item in ordered
            ),
        },
        "output_file": output_files,
        "scores": {
            "protocol": config["protocol"],
            "completion_status": (
                "runtime_blocked_no_ranking"
                if any_runtime_block
                else "completed"
            ),
            "runs": run_scores,
            "all_10": all_10,
            "top_3": top_3,
            "legacy_o_t_n": legacy_stats,
            "format_compliance_rate": all_10["formatComplianceRate"],
            "fixed_protocol_acceptance_rate": all_10[
                "fixedProtocolAcceptanceRate"
            ],
            "rejected_policy": config["rejectedPolicy"],
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 6 narrative2 fixed-v2",
        "",
        f"- status: {report['status']}",
        f"- official cases: {report['completedOfficialCases']}/54",
        f"- evaluated case-runs: {report['evaluatedCaseRuns']}",
        f"- formal predictions represented: {report['formalPredictions']}",
        f"- ranking published: {str(report['rankingPublished']).lower()}",
        f"- gold SHA-256: {report['goldSha256']}",
        "",
        "## Fixed evaluation method",
        "",
        "Binding count and order are fixed. Each binding is compared by index.",
        "All six fields use the protected narrative same_fixed comparator.",
        "Trend and Num presence follows narrative1 baseline; values follow narrative2.",
        "Gate-rejected rows score zero for all target binding fields.",
        "No semantic judge, synonym map, fuzzy match, percent conversion, or tolerance is used.",
        "",
        "## Results",
        "",
        "| output_id | model | prompt | all-10 F1 | sample SD | format | status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in report["ordering"]:
        f1 = item["scores"]["all_10"]["allFieldsCombined"]["f1"]
        fmt = item["scores"]["all_10"]["formatComplianceRate"]
        lines.append(
            f"| {item['model']['output_id']} | {item['model']['requested']} | "
            f"{item['prompt']['mode']} | {f1['mean']:.6f} | "
            f"{f1['sampleSd']:.6f} | {fmt['mean']:.6f} | "
            f"{item['scores']['completion_status']} |"
        )
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "prompt", "runtime", "output_file", "scores"],
            delimiter="\t",
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    key: json.dumps(
                        item[key], ensure_ascii=False, sort_keys=True
                    )
                    for key in (
                        "model",
                        "prompt",
                        "runtime",
                        "output_file",
                        "scores",
                    )
                }
            )


def run_reference_self_test(
    evaluator_path: Path, output_path: Path
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(evaluator_path),
            "self-test",
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ProtocolError(
            f"reference evaluator self-test failed: {completed.stderr}"
        )
    result = read_json(output_path)
    if not result.get("valid") or result.get("summary", {}).get("display") != "23/23":
        raise ProtocolError("reference evaluator self-test is not 23/23")
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_config(config_path)
    evaluation_bundle = workspace_path(config["evaluationBundle"]).resolve()
    bundle_report = verify_manifest(
        evaluation_bundle, str(config["evaluationBundleManifestSha256"])
    )
    evaluator_path = workspace_path(config["referenceEvaluator"]).resolve()
    gold_path = workspace_path(config["goldPath"]).resolve()
    vocabulary_path = workspace_path(config["trendVocabularyPath"]).resolve()
    workbook_path = workspace_path(config["sourceWorkbook"]).resolve()
    for path, expected in (
        (evaluator_path, config["referenceEvaluatorSha256"]),
        (gold_path, config["goldSha256"]),
        (vocabulary_path, config["trendVocabularySha256"]),
        (workbook_path, config["sourceWorkbookSha256"]),
    ):
        if sha256_file(path) != expected:
            raise ProtocolError(f"SHA-256 mismatch: {path}")

    fixed = load_module(evaluator_path, "narrative2_fixed_reference")
    legacy = load_module(
        SCRIPTS_ROOT / "evaluate_data_binding.py",
        "experiment6_legacy_data_binding",
    )
    gold_document = fixed.load_json(gold_path)
    targets = fixed._extract_targets(gold_document)
    if len(targets) != int(config["expectedRows"]):
        raise ProtocolError("gold row count mismatch")
    if sum(target["bindingCount"] for target in targets) != int(
        config["expectedBindingsPerRun"]
    ):
        raise ProtocolError("gold binding count mismatch")
    vocabulary, _ = fixed._load_vocabulary(vocabulary_path, gold_path)

    output_root = args.output_root.resolve()
    generation_snapshot = output_root / "generation_config.snapshot.json"
    if not generation_snapshot.is_file():
        raise ProtocolError(f"generation config snapshot missing: {generation_snapshot}")
    generation_config = read_json(generation_snapshot)
    if generation_config.get("protocol") != "experiment6-narrative2-full-v2":
        raise ProtocolError("generation snapshot protocol mismatch")
    cases = expand_matrix(generation_config)
    official_cases = [case for case in cases if case["official"]]
    case_by_id = {case["outputId"]: case for case in cases}
    selected_ids = (
        list(dict.fromkeys(args.only_case))
        if args.only_case
        else [case["outputId"] for case in official_cases]
    )
    unknown = set(selected_ids) - set(case_by_id)
    if unknown:
        raise ProtocolError(f"unknown --only-case values: {sorted(unknown)}")
    if any(not case_by_id[output_id]["official"] for output_id in selected_ids):
        raise ProtocolError("fixed-v2 formal evaluator accepts official cases only")

    evaluation_root = output_root / "evaluation_fixed_v2"
    write_json(evaluation_root / "evaluation_config.snapshot.json", config)
    self_test = run_reference_self_test(
        evaluator_path, evaluation_root / "reference_self_test.json"
    )
    run_results_by_case: dict[str, list[dict[str, Any]]] = {}
    blockers: list[dict[str, Any]] = []
    for output_id in selected_ids:
        case = case_by_id[output_id]
        run_results: list[dict[str, Any]] = []
        for run_number in range(1, int(config["expectedRuns"]) + 1):
            try:
                run_results.append(
                    evaluate_run(
                        fixed,
                        legacy,
                        config,
                        output_root,
                        case,
                        run_number,
                        targets,
                        vocabulary,
                    )
                )
            except ProtocolError as error:
                blockers.append(
                    {
                        "outputId": output_id,
                        "run": run_number,
                        "error": str(error),
                    }
                )
        if len(run_results) == int(config["expectedRuns"]):
            run_results_by_case[output_id] = run_results

    aggregates = [
        aggregate_case(
            fixed, config, output_root, case_by_id[output_id], run_results
        )
        for output_id, run_results in sorted(run_results_by_case.items())
    ]
    ordering = sorted(
        aggregates,
        key=lambda item: (
            float(
                item["scores"]["all_10"]["allFieldsCombined"]["f1"]["mean"]
            ),
            float(
                item["scores"]["all_10"]["formatComplianceRate"]["mean"]
            ),
        ),
        reverse=True,
    )
    full_scope = not args.only_case
    all_cases_present = len(aggregates) == int(config["expectedOfficialCases"])
    no_runtime_blocks = all(
        item["scores"]["completion_status"] == "completed"
        for item in aggregates
    )
    ranking_published = (
        full_scope and all_cases_present and not blockers and no_runtime_blocks
    )
    status = (
        "completed"
        if ranking_published
        else (
            "development_partial_no_ranking"
            if args.only_case and not blockers
            else "incomplete_no_ranking"
        )
    )
    report = {
        "generatedAt": utc_now(),
        "protocol": config["protocol"],
        "status": status,
        "experimentId": output_root.name,
        "completedOfficialCases": len(aggregates),
        "evaluatedCaseRuns": sum(
            len(items) for items in run_results_by_case.values()
        ),
        "formalPredictions": sum(
            len(items) for items in run_results_by_case.values()
        )
        * int(config["expectedRows"]),
        "rankingPublished": ranking_published,
        "blockers": blockers,
        "runtimeBlockedCases": [
            item["model"]["output_id"]
            for item in aggregates
            if item["scores"]["completion_status"] != "completed"
        ],
        "dataSha256": generation_config["inputWorkbook"]["sha256"],
        "sourceWorkbookSha256": config["sourceWorkbookSha256"],
        "goldSha256": config["goldSha256"],
        "evaluationBundle": bundle_report,
        "referenceSelfTest": self_test["summary"],
        "ordering": ordering,
    }
    write_json(evaluation_root / "evaluation_report.json", report)
    (evaluation_root / "evaluation_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    partial_path = output_root / "experiment6_fixed_v2_results.partial.json"
    write_json(partial_path, aggregates)
    write_tsv(
        output_root / "experiment6_fixed_v2_results.partial.tsv", aggregates
    )
    if ranking_published:
        write_json(output_root / "experiment6_results.json", aggregates)
        write_tsv(output_root / "experiment6_results.tsv", aggregates)
    write_json(
        evaluation_root / "evaluation_progress.json",
        {
            "generatedAt": report["generatedAt"],
            "status": status,
            "rankingPublished": ranking_published,
            "completedOfficialCases": len(aggregates),
            "evaluatedCaseRuns": report["evaluatedCaseRuns"],
            "formalPredictions": report["formalPredictions"],
            "blockers": blockers,
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "config"
            / "experiment6_narrative2_fixed_evaluation.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--only-case", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = build(parse_args(argv))
    except ProtocolError as error:
        print(
            json.dumps(
                {
                    "generatedAt": utc_now(),
                    "protocol": "narrative2-fixed-python-v2",
                    "status": "blocked",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] in {
        "completed",
        "development_partial_no_ranking",
    } else 2


if __name__ == "__main__":
    sys.exit(main())
