#!/usr/bin/env python3
"""Materialize a unified, diagnostic Experiment 6 Binding dataset.

The materializer reads two or more frozen generation roots, verifies every
declared source hash, and emits all case-run-source rows.  Safe repair is
limited to JSON wrapper normalization and ObjectName singleton-array typing;
it never reads gold data or changes semantic field values.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from analyze_experiment6_repair_sensitivity import (
    REQUIRED_BINDING_KEYS,
    validate_binding,
)
from materialize_experiment6_binding_candidates import (
    candidate_id,
    read_json,
    read_jsonl,
    resolve_artifact,
    sha256_file,
    stable_sha256,
    utc_now,
    write_json,
    write_jsonl,
)


PROTOCOL = "experiment6-binding-materialization-v2-unified34"
SAFE_REPAIR_POLICY = "wrapper-and-objectname-shape-only-v1"
ALLOWED_REPAIR_OPERATIONS = {
    "add-missing-reason-wrapper",
    "wrap-single-binding-result-array",
    "wrap-objectname-singleton-array",
}


class UnifiedMaterializationError(RuntimeError):
    """Raised when unified source or output invariants fail."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UnifiedMaterializationError(message)


def source_row_key(row: Mapping[str, Any]) -> tuple[int, str]:
    try:
        return int(row["index"]), str(row["source"])
    except (KeyError, TypeError, ValueError) as error:
        raise UnifiedMaterializationError(
            f"row lacks a valid index/source key: {row}"
        ) from error


def verified_artifact(
    root: Path, manifest: Mapping[str, Any], name: str
) -> tuple[Path, str]:
    path = resolve_artifact(root, manifest, name)
    expected = str((manifest.get("hashes") or {}).get(name) or "")
    actual = sha256_file(path)
    require(bool(expected) and actual == expected, f"{name} SHA mismatch: {path}")
    return path, actual


def structural_payload_result(
    payload: Any,
) -> tuple[list[dict[str, Any]] | None, list[str], str]:
    """Return schema-valid bindings after a narrow, value-preserving repair."""
    operations: list[str] = []
    result: Any
    if isinstance(payload, dict) and set(payload) == {"result", "reason"}:
        if not isinstance(payload["reason"], str):
            return None, operations, "reason_not_string"
        result = payload["result"]
    elif isinstance(payload, dict) and set(payload) == {"Binding"}:
        result = payload["Binding"]
    elif isinstance(payload, list):
        result = payload
    elif isinstance(payload, dict) and set(payload) == {"result"}:
        result = payload["result"]
        operations.append("add-missing-reason-wrapper")
    elif isinstance(payload, dict) and set(payload) == REQUIRED_BINDING_KEYS:
        result = [payload]
        operations.append("wrap-single-binding-result-array")
    else:
        return None, operations, "top_level_contract"
    if not isinstance(result, list):
        return None, operations, "result_not_array"

    normalized: list[dict[str, Any]] = []
    for binding in result:
        if not isinstance(binding, dict):
            return None, operations, "binding_not_object"
        item = dict(binding)
        if (
            set(item) == REQUIRED_BINDING_KEYS
            and isinstance(item.get("ObjectName"), str)
            and item["ObjectName"].strip()
        ):
            item["ObjectName"] = [item["ObjectName"]]
            if "wrap-objectname-singleton-array" not in operations:
                operations.append("wrap-objectname-singleton-array")
        valid, reason = validate_binding(item)
        if not valid:
            return None, operations, reason
        normalized.append(item)
    require(set(operations).issubset(ALLOWED_REPAIR_OPERATIONS), "unsafe repair operation")
    return normalized, operations, "valid"


def parse_source_args(values: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise UnifiedMaterializationError("--source must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        require(bool(name) and bool(raw_path), "--source must be NAME=PATH")
        require(name not in sources, f"duplicate source name: {name}")
        path = Path(raw_path).resolve()
        require(path.is_dir(), f"source root missing: {path}")
        sources[name] = path
    return sources


def validate_config(config: Mapping[str, Any], source_roots: Mapping[str, Path]) -> None:
    require(config.get("materializationProtocol") == PROTOCOL, "protocol mismatch")
    require(config.get("safeRepairPolicy") == SAFE_REPAIR_POLICY, "repair policy mismatch")
    require(set(config.get("requiredBindingKeys") or []) == REQUIRED_BINDING_KEYS, "Binding keys mismatch")
    require(int(config.get("expectedCases", 0)) > 0, "expectedCases must be positive")
    require(int(config.get("expectedRuns", 0)) > 0, "expectedRuns must be positive")
    require(int(config.get("expectedRows", 0)) > 0, "expectedRows must be positive")
    groups = config.get("sourceGroups")
    require(isinstance(groups, list) and groups, "sourceGroups must be non-empty")
    names = [str(group.get("name")) for group in groups]
    require(len(names) == len(set(names)), "duplicate source group")
    require(set(names) == set(source_roots), "CLI/config source groups differ")
    all_cases = [str(case_id) for group in groups for case_id in group.get("caseIds", [])]
    require(len(all_cases) == len(set(all_cases)), "case assigned to multiple source groups")
    require(len(all_cases) == int(config["expectedCases"]), "case count mismatch")


def collect_entries(
    config: Mapping[str, Any], source_roots: Mapping[str, Path]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    expected_runs = list(range(1, int(config["expectedRuns"]) + 1))
    for group in config["sourceGroups"]:
        group_name = str(group["name"])
        root = source_roots[group_name]
        selected = set(str(value) for value in group["caseIds"])
        grouped: dict[str, list[int]] = defaultdict(list)
        for manifest_path in sorted((root / "manifests").glob("*.json")):
            manifest = read_json(manifest_path)
            if manifest.get("official") is not True:
                continue
            output_id = str(manifest.get("outputId") or "")
            if output_id not in selected:
                continue
            require(manifest.get("protocol") == group["sourceProtocol"], f"source protocol mismatch: {output_id}")
            require(
                manifest.get("compatibilityFingerprint")
                == group.get("sourceCompatibilityFingerprint"),
                f"source compatibility fingerprint mismatch: {output_id}",
            )
            grouped[output_id].append(int(manifest["run"]))
            entries.append(
                {
                    "sourceGroup": group_name,
                    "root": root,
                    "manifest": manifest,
                    "manifestPath": manifest_path.resolve(),
                    "requireRepairCoverage": bool(group["requireRepairCoverage"]),
                }
            )
        require(set(grouped) == selected, f"source group case coverage mismatch: {group_name}")
        incomplete = {
            output_id: sorted(runs)
            for output_id, runs in grouped.items()
            if sorted(runs) != expected_runs
        }
        require(not incomplete, f"source group run coverage mismatch: {incomplete}")
    expected_total = int(config["expectedCases"]) * int(config["expectedRuns"])
    require(len(entries) == expected_total, f"manifest count mismatch: {len(entries)} != {expected_total}")
    pairs = [(str(item["manifest"]["outputId"]), int(item["manifest"]["run"])) for item in entries]
    require(len(pairs) == len(set(pairs)), "duplicate case-run across source groups")
    return sorted(entries, key=lambda item: (str(item["manifest"]["outputId"]), int(item["manifest"]["run"])))


def materialize_run(
    *,
    entry: Mapping[str, Any],
    staging_root: Path,
    final_root: Path,
    fingerprint: str,
    expected_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(entry["root"])
    manifest = entry["manifest"]
    manifest_path = Path(entry["manifestPath"])
    source_group = str(entry["sourceGroup"])
    output_id = str(manifest["outputId"])
    run = int(manifest["run"])
    seed = manifest.get("seed")
    prediction_path, prediction_hash = verified_artifact(root, manifest, "predictions")
    repair_path, repair_hash = verified_artifact(root, manifest, "nonformalRepair")
    predictions = read_jsonl(prediction_path)
    repairs = read_jsonl(repair_path)
    require(len(predictions) == expected_rows, f"{output_id} run {run}: row count mismatch")
    prediction_keys = [source_row_key(row) for row in predictions]
    require(len(prediction_keys) == len(set(prediction_keys)), f"{output_id} run {run}: duplicate row key")
    require([key[0] for key in prediction_keys] == list(range(expected_rows)), f"{output_id} run {run}: index coverage mismatch")

    repair_by_key: dict[tuple[int, str], tuple[int, dict[str, Any]]] = {}
    for line_number, repair_row in enumerate(repairs, 1):
        require(repair_row.get("official") is False, f"formal repair row: {repair_path}:{line_number}")
        require(repair_row.get("excludedFromScores") is True, f"repair not excluded: {repair_path}:{line_number}")
        key = source_row_key(repair_row)
        require(key not in repair_by_key, f"duplicate repair key: {repair_path}:{key}")
        repair_by_key[key] = (line_number, repair_row)
    if entry["requireRepairCoverage"]:
        require(set(repair_by_key) == set(prediction_keys), f"repair coverage mismatch: {output_id} run {run}")
    else:
        require(set(repair_by_key).issubset(set(prediction_keys)), f"orphan repair row: {output_id} run {run}")

    derived_predictions: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    binding_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    operations_total: Counter[str] = Counter()
    source_manifest_hash = sha256_file(manifest_path)
    for prediction_line, prediction in enumerate(predictions, 1):
        index, source = source_row_key(prediction)
        original_format_valid = bool(prediction.get("formatValid"))
        item = dict(prediction)
        result: list[dict[str, Any]] | None = None
        operations: list[str] = []
        repair_line: int | None = None
        repair_method: str | None = None
        repair_payload: Any = None
        rejection_reason: str | None = None
        if original_format_valid and isinstance(prediction.get("result"), list):
            candidate_result = list(prediction["result"])
            for binding in candidate_result:
                valid, reason = validate_binding(binding)
                require(valid, f"source strict row violates schema: {prediction_path}:{prediction_line}:{reason}")
            result = candidate_result
            status = "source_strict_valid"
        else:
            repair_entry = repair_by_key.get((index, source))
            if repair_entry is None:
                status = "repair_missing"
                rejection_reason = status
            else:
                repair_line, repair_row = repair_entry
                repair = repair_row.get("repair") or {}
                repair_method = str(repair.get("method") or "recorded-nonformal-repair")
                repair_payload = repair.get("payload")
                if not repair.get("available"):
                    status = "repair_unavailable"
                    rejection_reason = status
                else:
                    result, operations, reason = structural_payload_result(repair_payload)
                    if result is None:
                        status = "repair_schema_invalid"
                        rejection_reason = reason
                    elif operations:
                        status = "safe_structural_repair_valid"
                    else:
                        status = "recorded_repair_schema_valid"
        counts[status] += 1
        operations_total.update(operations)
        schema_valid = result is not None
        if schema_valid:
            item["result"] = result
            item["formatValid"] = True
            counts["acceptedRows"] += 1
            counts["bindings"] += len(result)
        else:
            item["result"] = []
            item["formatValid"] = False
            counts["rejectedRows"] += 1
            reasons[str(rejection_reason)] += 1
        item["bindingCandidate"] = {
            "protocol": PROTOCOL,
            "status": status,
            "diagnosticOnly": True,
            "claimEligible": False,
            "sourceGroup": source_group,
            "sourceOfficial": original_format_valid,
            "recordedRepairMethod": repair_method,
            "safeRepairOperations": operations,
            "sourcePredictionSha256": prediction_hash,
            "sourceRepairSha256": repair_hash,
        }
        derived_predictions.append(item)

        provenance = {
            "sourceGroup": source_group,
            "sourceGenerationRoot": str(root),
            "sourceManifest": str(manifest_path),
            "sourceManifestSha256": source_manifest_hash,
            "sourcePrediction": str(prediction_path),
            "sourcePredictionSha256": prediction_hash,
            "sourcePredictionLine": prediction_line,
            "sourceRepair": str(repair_path),
            "sourceRepairSha256": repair_hash,
            "sourceRepairLine": repair_line,
            "recordedRepairMethod": repair_method,
            "safeRepairOperations": operations,
        }
        row_record = {
            "schemaVersion": 2,
            "protocol": PROTOCOL,
            "outputId": output_id,
            "run": run,
            "seed": seed,
            "index": index,
            "source": source,
            "sourceGroup": source_group,
            "candidateStatus": status,
            "originalFormatValid": original_format_valid,
            "formatValid": schema_valid,
            "schemaValid": schema_valid,
            "bindingCount": len(result or []),
            "Binding": result or [],
            "rejectionReason": rejection_reason,
            "diagnosticOnly": True,
            "claimEligible": False,
            "provenance": provenance,
        }
        row_records.append(row_record)
        if schema_valid:
            for binding_index, binding in enumerate(result or []):
                binding_records.append(
                    {
                        "schemaVersion": 2,
                        "protocol": PROTOCOL,
                        "candidateId": candidate_id(fingerprint, output_id, run, source, index, binding_index),
                        "outputId": output_id,
                        "run": run,
                        "seed": seed,
                        "index": index,
                        "source": source,
                        "sourceGroup": source_group,
                        "bindingIndex": binding_index,
                        **binding,
                        "candidateStatus": status,
                        "diagnosticOnly": True,
                        "claimEligible": False,
                        "sourceManifestSha256": source_manifest_hash,
                        "sourcePredictionSha256": prediction_hash,
                        "sourceRepairSha256": repair_hash,
                        "recordedRepairMethod": repair_method,
                        "safeRepairOperations": operations,
                    }
                )
        else:
            rejected_records.append(
                {
                    **{key: value for key, value in row_record.items() if key != "Binding"},
                    "repairPayload": repair_payload,
                }
            )

    relative_dir = Path("cases") / output_id / f"run_{run:02d}"
    stage_dir = staging_root / relative_dir
    final_dir = final_root / relative_dir
    prediction_out = stage_dir / "predictions.binding_candidates.jsonl"
    binding_rows_out = stage_dir / "binding.jsonl"
    rows_alias_out = stage_dir / "rows.jsonl"
    bindings_out = stage_dir / "bindings.jsonl"
    rejected_out = stage_dir / "rejected_rows.jsonl"
    write_jsonl(prediction_out, derived_predictions)
    write_jsonl(binding_rows_out, row_records)
    os.link(binding_rows_out, rows_alias_out)
    write_jsonl(bindings_out, binding_records)
    write_jsonl(rejected_out, rejected_records)
    files = {
        "predictions": str(final_dir / prediction_out.name),
        "binding": str(final_dir / binding_rows_out.name),
        "rows": str(final_dir / rows_alias_out.name),
        "bindings": str(final_dir / bindings_out.name),
        "rejectedRows": str(final_dir / rejected_out.name),
    }
    hashes = {
        "predictions": sha256_file(prediction_out),
        "binding": sha256_file(binding_rows_out),
        "rows": sha256_file(rows_alias_out),
        "bindings": sha256_file(bindings_out),
        "rejectedRows": sha256_file(rejected_out),
    }
    run_manifest = {
        "schemaVersion": 2,
        "protocol": PROTOCOL,
        "safeRepairPolicy": SAFE_REPAIR_POLICY,
        "status": "completed_diagnostic_binding_candidates",
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "goldAccessed": False,
        "outputId": output_id,
        "run": run,
        "seed": seed,
        "sourceGroup": source_group,
        "expectedRows": expected_rows,
        "acceptedRows": int(counts["acceptedRows"]),
        "rejectedRows": int(counts["rejectedRows"]),
        "bindingCount": int(counts["bindings"]),
        "candidateStatusCounts": dict(sorted(counts.items())),
        "safeRepairOperationCounts": dict(sorted(operations_total.items())),
        "rejectionReasons": dict(sorted(reasons.items())),
        "files": files,
        "hashes": hashes,
        "source": {
            "generationRoot": str(root),
            "manifest": str(manifest_path),
            "manifestSha256": source_manifest_hash,
            "predictions": str(prediction_path),
            "predictionsSha256": prediction_hash,
            "nonformalRepair": str(repair_path),
            "nonformalRepairSha256": repair_hash,
            "generationProtocol": manifest.get("protocol"),
            "compatibilityFingerprint": manifest.get("compatibilityFingerprint"),
        },
        "route": manifest.get("route"),
        "declaredRoute": manifest.get("declaredRoute"),
        "effectiveRoute": manifest.get("effectiveRoute"),
        "compatibilityFingerprint": fingerprint,
    }
    provenance_entry = {
        "outputId": output_id,
        "run": run,
        "sourceGroup": source_group,
        "sourceManifest": str(manifest_path),
        "sourceManifestSha256": source_manifest_hash,
        "sourcePredictions": str(prediction_path),
        "sourcePredictionsSha256": prediction_hash,
        "sourceNonformalRepair": str(repair_path),
        "sourceNonformalRepairSha256": repair_hash,
        "outputPredictions": files["predictions"],
        "outputPredictionsSha256": hashes["predictions"],
    }
    return run_manifest, row_records, binding_records, rejected_records, [provenance_entry]


def write_inventory(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_inventory.tsv":
            rows.append((str(path.relative_to(root)), path.stat().st_size, sha256_file(path)))
    with (root / "sha256_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    require(not output_root.exists(), f"output root already exists: {output_root}")
    config_path = args.config.resolve()
    config = read_json(config_path)
    source_roots = parse_source_args(args.source)
    validate_config(config, source_roots)
    entries = collect_entries(config, source_roots)
    source_manifest_hashes = [sha256_file(Path(entry["manifestPath"])) for entry in entries]
    fingerprint = stable_sha256(
        {
            "protocol": PROTOCOL,
            "safeRepairPolicy": SAFE_REPAIR_POLICY,
            "configSha256": sha256_file(config_path),
            "materializerSha256": sha256_file(Path(__file__).resolve()),
            "sourceManifestSha256": source_manifest_hashes,
        }
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    all_rows: list[dict[str, Any]] = []
    all_bindings: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    source_orders: set[str] = set()
    for entry in entries:
        run_manifest, rows, bindings, rejected, provenance_rows = materialize_run(
            entry=entry,
            staging_root=staging_root,
            final_root=output_root,
            fingerprint=fingerprint,
            expected_rows=int(config["expectedRows"]),
        )
        source_orders.add(stable_sha256([row["source"] for row in rows]))
        all_rows.extend(rows)
        all_bindings.extend(bindings)
        all_rejected.extend(rejected)
        provenance.extend(provenance_rows)
        run_manifests.append(run_manifest)
        write_json(
            staging_root / "manifests" / f"{run_manifest['outputId']}__run_{int(run_manifest['run']):02d}.json",
            run_manifest,
        )
    require(len(source_orders) == 1, "source order differs across case-runs")
    expected_rows_total = int(config["expectedCases"]) * int(config["expectedRuns"]) * int(config["expectedRows"])
    require(len(all_rows) == expected_rows_total, "aggregate row count mismatch")

    binding_path = staging_root / "binding.jsonl"
    rows_path = staging_root / "rows.jsonl"
    write_jsonl(binding_path, all_rows)
    os.link(binding_path, rows_path)
    write_jsonl(staging_root / "bindings.jsonl", all_bindings)
    write_jsonl(staging_root / "rejected_rows.jsonl", all_rejected)
    write_json(staging_root / "source_provenance.json", {"schemaVersion": 2, "protocol": PROTOCOL, "artifacts": provenance})

    rows_by_case: Counter[str] = Counter()
    accepted_by_case: Counter[str] = Counter()
    bindings_by_case: Counter[str] = Counter()
    for row in all_rows:
        rows_by_case[str(row["outputId"])] += 1
        if row["schemaValid"]:
            accepted_by_case[str(row["outputId"])] += 1
    for binding in all_bindings:
        bindings_by_case[str(binding["outputId"])] += 1
    with (staging_root / "binding_counts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["outputId", "rows", "acceptedRows", "rejectedRows", "bindings"])
        for output_id in sorted(rows_by_case):
            writer.writerow([output_id, rows_by_case[output_id], accepted_by_case[output_id], rows_by_case[output_id] - accepted_by_case[output_id], bindings_by_case[output_id]])

    statuses = Counter(str(row["candidateStatus"]) for row in all_rows)
    reasons = Counter(str(row["rejectionReason"]) for row in all_rejected)
    operations = Counter(operation for row in all_rows for operation in row["provenance"]["safeRepairOperations"])
    source_group_counts = Counter(str(row["sourceGroup"]) for row in all_rows)
    accepted_rows = [row for row in all_rows if row["schemaValid"]]
    dataset = {
        "schemaVersion": 2,
        "protocol": PROTOCOL,
        "safeRepairPolicy": SAFE_REPAIR_POLICY,
        "status": "complete",
        "createdAt": utc_now(),
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "goldAccessed": False,
        "outputRoot": str(output_root),
        "configPath": str(config_path),
        "configSha256": sha256_file(config_path),
        "materializerSha256": sha256_file(Path(__file__).resolve()),
        "compatibilityFingerprint": fingerprint,
        "sourceGroups": [
            {"name": group["name"], "root": str(source_roots[str(group["name"])]), "caseIds": group["caseIds"]}
            for group in config["sourceGroups"]
        ],
        "grain": {
            "binding": "one case-run-source row; Binding is an array and empty rows are retained",
            "bindings": "one schema-valid Binding per case-run-source-bindingIndex",
            "rejectedRows": "one schema-invalid or unavailable case-run-source row",
        },
        "keys": {
            "binding": ["outputId", "run", "source", "index"],
            "bindings": ["outputId", "run", "source", "index", "bindingIndex"],
        },
        "counts": {
            "cases": len(rows_by_case),
            "caseRuns": len(run_manifests),
            "rows": len(all_rows),
            "acceptedRows": len(accepted_rows),
            "acceptedRowsWithBindings": sum(int(row["bindingCount"]) > 0 for row in accepted_rows),
            "acceptedEmptyBindingRows": sum(int(row["bindingCount"]) == 0 for row in accepted_rows),
            "rejectedRows": len(all_rejected),
            "bindings": len(all_bindings),
            "sourceGroupRows": dict(sorted(source_group_counts.items())),
            "candidateStatus": dict(sorted(statuses.items())),
            "safeRepairOperations": dict(sorted(operations.items())),
            "rejectionReasons": dict(sorted(reasons.items())),
        },
        "files": {
            "binding": str(output_root / "binding.jsonl"),
            "rows": str(output_root / "rows.jsonl"),
            "bindings": str(output_root / "bindings.jsonl"),
            "rejectedRows": str(output_root / "rejected_rows.jsonl"),
            "bindingCounts": str(output_root / "binding_counts.tsv"),
            "sourceProvenance": str(output_root / "source_provenance.json"),
        },
        "hashes": {
            "binding": sha256_file(binding_path),
            "rows": sha256_file(rows_path),
            "bindings": sha256_file(staging_root / "bindings.jsonl"),
            "rejectedRows": sha256_file(staging_root / "rejected_rows.jsonl"),
            "bindingCounts": sha256_file(staging_root / "binding_counts.tsv"),
            "sourceProvenance": sha256_file(staging_root / "source_provenance.json"),
        },
        "limitations": [
            "This is a diagnostic materialization and does not replace formal predictions or rankings.",
            "Safe repair changes JSON shape only; semantic field values are never normalized.",
            "Numeric strings with units remain schema-invalid and are retained in rejected_rows.jsonl.",
            "Four GPT-4.1 cases are intentionally excluded.",
        ],
    }
    write_json(staging_root / "dataset_manifest.json", dataset)
    readme = [
        "# Experiment 6 unified Binding dataset (34 cases)",
        "",
        "- Diagnostic only: `official=false`, `claimEligible=false`.",
        f"- Cases / runs / rows: {len(rows_by_case)} / {len(run_manifests)} / {len(all_rows)}.",
        f"- Valid rows / rejected rows / Bindings: {len(accepted_rows)} / {len(all_rejected)} / {len(all_bindings)}.",
        "- `binding.jsonl` retains every case-run-source row, including empty and rejected outputs.",
        "- `bindings.jsonl` is the one-Binding-per-line projection.",
        "- No gold or judge data was read during materialization.",
    ]
    (staging_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    write_inventory(staging_root)
    os.replace(staging_root, output_root)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build(args)
    except (UnifiedMaterializationError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "counts": report["counts"], "outputRoot": report["outputRoot"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
