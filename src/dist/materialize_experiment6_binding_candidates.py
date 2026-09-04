#!/usr/bin/env python3
"""Materialize traceable Experiment 6 Binding candidates for later evaluation.

This tool never changes generation artifacts and never consults gold data.  It
accepts only strict official predictions plus the generation-time nonformal
repair records already named and hashed by each generation manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from analyze_experiment6_repair_sensitivity import (
    REQUIRED_BINDING_KEYS,
    SensitivityError,
    strict_payload_result,
    validate_binding,
)


PROTOCOL = "experiment6-binding-candidate-materialization-v1"


class MaterializationError(RuntimeError):
    """Raised when the source or output violates the materialization contract."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MaterializationError(
                    f"{path}:{line_number}: JSONL row is not an object"
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_artifact(
    generation_root: Path, manifest: Mapping[str, Any], name: str
) -> Path:
    raw = str((manifest.get("files") or {}).get(name) or "")
    if not raw:
        raise MaterializationError(
            f"{manifest.get('outputId')} run {manifest.get('run')}: missing files.{name}"
        )
    declared = Path(raw)
    if declared.is_file():
        return declared.resolve()
    relocated = (
        generation_root
        / "cases"
        / str(manifest["outputId"])
        / f"run_{int(manifest['run']):02d}"
        / declared.name
    )
    if not relocated.is_file():
        raise MaterializationError(f"missing {name} artifact: {relocated}")
    return relocated.resolve()


def verified_artifact(
    generation_root: Path, manifest: Mapping[str, Any], name: str
) -> tuple[Path, str]:
    path = resolve_artifact(generation_root, manifest, name)
    expected = str((manifest.get("hashes") or {}).get(name) or "")
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise MaterializationError(
            f"{manifest['outputId']} run {manifest['run']}: {name} SHA-256 mismatch"
        )
    return path, actual


def candidate_id(
    fingerprint: str,
    output_id: str,
    run: int,
    source: str,
    index: int,
    binding_index: int,
) -> str:
    return stable_sha256(
        [fingerprint, output_id, run, source, index, binding_index]
    )


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "protocol",
        "sourceProtocol",
        "sourceCompatibilityFingerprint",
        "expectedCases",
        "expectedRuns",
        "expectedRows",
        "caseIds",
        "requiredBindingKeys",
        "requireRepairCoverage",
    }
    if set(config) != required:
        raise MaterializationError(
            f"config keys mismatch: expected {sorted(required)}, got {sorted(config)}"
        )
    if config["protocol"] != PROTOCOL:
        raise MaterializationError("config protocol mismatch")
    if set(config["requiredBindingKeys"]) != REQUIRED_BINDING_KEYS:
        raise MaterializationError("required Binding keys mismatch")
    if len(config["caseIds"]) != int(config["expectedCases"]):
        raise MaterializationError("caseIds length does not match expectedCases")


def source_row_key(row: Mapping[str, Any]) -> tuple[int, str]:
    try:
        return int(row["index"]), str(row["source"])
    except (KeyError, TypeError, ValueError) as error:
        raise MaterializationError(f"row lacks a valid index/source key: {row}") from error


def materialize_run(
    *,
    generation_root: Path,
    staging_root: Path,
    final_root: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    fingerprint: str,
    expected_rows: int,
    require_repair_coverage: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_id = str(manifest["outputId"])
    run = int(manifest["run"])
    seed = manifest.get("seed")
    prediction_path, prediction_hash = verified_artifact(
        generation_root, manifest, "predictions"
    )
    repair_path, repair_hash = verified_artifact(
        generation_root, manifest, "nonformalRepair"
    )
    predictions = read_jsonl(prediction_path)
    repairs = read_jsonl(repair_path)
    if len(predictions) != expected_rows:
        raise MaterializationError(
            f"{output_id} run {run}: expected {expected_rows} predictions, got {len(predictions)}"
        )
    prediction_keys = [source_row_key(row) for row in predictions]
    if len(set(prediction_keys)) != len(prediction_keys):
        raise MaterializationError(f"{output_id} run {run}: duplicate prediction key")
    if [key[0] for key in prediction_keys] != list(range(expected_rows)):
        raise MaterializationError(f"{output_id} run {run}: indices are not 0..{expected_rows - 1}")

    repair_by_key: dict[tuple[int, str], tuple[int, dict[str, Any]]] = {}
    for line_number, repair_row in enumerate(repairs, 1):
        if repair_row.get("official") is not False or repair_row.get(
            "excludedFromScores"
        ) is not True:
            raise MaterializationError(
                f"{repair_path}:{line_number}: repair is not explicitly nonformal"
            )
        key = source_row_key(repair_row)
        if key in repair_by_key:
            raise MaterializationError(f"{repair_path}: duplicate repair key {key}")
        repair_by_key[key] = (line_number, repair_row)
    if require_repair_coverage and set(repair_by_key) != set(prediction_keys):
        raise MaterializationError(
            f"{output_id} run {run}: repair and prediction key coverage differ"
        )

    derived_predictions: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    binding_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    manifest_hash = sha256_file(manifest_path)

    for prediction_line, prediction in enumerate(predictions, 1):
        index, source = source_row_key(prediction)
        item = dict(prediction)
        repair_line: int | None = None
        repair_method: str | None = None
        repair_payload: Any = None
        rejection_reason: str | None = None
        status: str
        result: list[dict[str, Any]] | None = None

        if prediction.get("formatValid") and isinstance(prediction.get("result"), list):
            result = list(prediction["result"])
            for binding in result:
                valid, reason = validate_binding(binding)
                if not valid:
                    raise MaterializationError(
                        f"{prediction_path}:{prediction_line}: official valid row violates schema: {reason}"
                    )
            status = "official_schema_valid"
            counts[status] += 1
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
                    result, reason = strict_payload_result(repair_payload)
                    if result is None:
                        status = "repair_schema_invalid"
                        rejection_reason = reason
                    else:
                        status = "repair_schema_valid"
            counts[status] += 1

        accepted = result is not None
        if accepted:
            item["result"] = result
            item["formatValid"] = True
            item["parserDiagnostic"] = {
                "strict": status == "official_schema_valid",
                "valid": True,
                "diagnosticOnly": status != "official_schema_valid",
                "method": repair_method if status == "repair_schema_valid" else "source-strict-parser",
                "sourceOfficial": status == "official_schema_valid",
                "claimEligible": status == "official_schema_valid",
            }
            counts["acceptedRows"] += 1
            counts["bindings"] += len(result)
        else:
            item["result"] = []
            item["formatValid"] = False
            counts["rejectedRows"] += 1
            rejection_reasons[str(rejection_reason)] += 1
        item["bindingCandidate"] = {
            "protocol": PROTOCOL,
            "status": status,
            "diagnosticOnly": status != "official_schema_valid",
            "claimEligible": status == "official_schema_valid",
            "sourcePredictionSha256": prediction_hash,
            "sourceRepairSha256": repair_hash,
        }
        derived_predictions.append(item)

        row_record = {
            "schemaVersion": 1,
            "protocol": PROTOCOL,
            "outputId": output_id,
            "run": run,
            "seed": seed,
            "index": index,
            "source": source,
            "candidateStatus": status,
            "schemaValid": accepted,
            "bindingCount": len(result or []),
            "Binding": result or [],
            "rejectionReason": rejection_reason,
            "diagnosticOnly": status != "official_schema_valid",
            "claimEligible": status == "official_schema_valid",
            "provenance": {
                "sourceManifest": str(manifest_path),
                "sourceManifestSha256": manifest_hash,
                "sourcePrediction": str(prediction_path),
                "sourcePredictionSha256": prediction_hash,
                "sourcePredictionLine": prediction_line,
                "sourceRepair": str(repair_path),
                "sourceRepairSha256": repair_hash,
                "sourceRepairLine": repair_line,
                "repairMethod": repair_method,
            },
        }
        row_records.append(row_record)

        if accepted:
            for binding_index, binding in enumerate(result or []):
                binding_records.append(
                    {
                        "schemaVersion": 1,
                        "protocol": PROTOCOL,
                        "candidateId": candidate_id(
                            fingerprint,
                            output_id,
                            run,
                            source,
                            index,
                            binding_index,
                        ),
                        "outputId": output_id,
                        "run": run,
                        "seed": seed,
                        "index": index,
                        "source": source,
                        "bindingIndex": binding_index,
                        **binding,
                        "candidateStatus": status,
                        "diagnosticOnly": status != "official_schema_valid",
                        "claimEligible": status == "official_schema_valid",
                        "sourceManifestSha256": manifest_hash,
                        "sourcePredictionSha256": prediction_hash,
                        "sourceRepairSha256": repair_hash,
                        "repairMethod": repair_method,
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
    stage_run_dir = staging_root / relative_dir
    final_run_dir = final_root / relative_dir
    predictions_out = stage_run_dir / "predictions.binding_candidates.jsonl"
    rows_out = stage_run_dir / "rows.jsonl"
    bindings_out = stage_run_dir / "bindings.jsonl"
    rejected_out = stage_run_dir / "rejected_rows.jsonl"
    write_jsonl(predictions_out, derived_predictions)
    write_jsonl(rows_out, row_records)
    write_jsonl(bindings_out, binding_records)
    write_jsonl(rejected_out, rejected_records)

    files = {
        "predictions": str(final_run_dir / predictions_out.name),
        "rows": str(final_run_dir / rows_out.name),
        "bindings": str(final_run_dir / bindings_out.name),
        "rejectedRows": str(final_run_dir / rejected_out.name),
    }
    hashes = {
        "predictions": sha256_file(predictions_out),
        "rows": sha256_file(rows_out),
        "bindings": sha256_file(bindings_out),
        "rejectedRows": sha256_file(rejected_out),
    }
    run_manifest = {
        "schemaVersion": 1,
        "protocol": PROTOCOL,
        "status": "completed_diagnostic_binding_candidates",
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "outputId": output_id,
        "run": run,
        "seed": seed,
        "expectedRows": expected_rows,
        "acceptedRows": int(counts["acceptedRows"]),
        "rejectedRows": int(counts["rejectedRows"]),
        "bindingCount": int(counts["bindings"]),
        "candidateStatusCounts": dict(sorted(counts.items())),
        "rejectionReasons": dict(sorted(rejection_reasons.items())),
        "files": files,
        "hashes": hashes,
        "source": {
            "generationRoot": str(generation_root),
            "manifest": str(manifest_path),
            "manifestSha256": manifest_hash,
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
        "goldAccessed": False,
    }
    return run_manifest, row_records, binding_records, rejected_records


def build(args: argparse.Namespace) -> dict[str, Any]:
    generation_root = args.generation_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    if output_root.exists():
        raise MaterializationError(f"output root already exists: {output_root}")
    config = read_json(config_path)
    validate_config(config)
    manifest_paths = sorted((generation_root / "manifests").glob("*.json"))
    manifests_with_paths = [
        (path.resolve(), read_json(path)) for path in manifest_paths
    ]
    manifests_with_paths = [
        item for item in manifests_with_paths if item[1].get("official") is True
    ]
    selected_cases = set(str(value) for value in config["caseIds"])
    manifests_with_paths = [
        item
        for item in manifests_with_paths
        if str(item[1].get("outputId")) in selected_cases
    ]
    actual_cases = {str(item[1]["outputId"]) for item in manifests_with_paths}
    if actual_cases != selected_cases:
        raise MaterializationError(
            f"case coverage mismatch: missing={sorted(selected_cases - actual_cases)}, "
            f"extra={sorted(actual_cases - selected_cases)}"
        )
    grouped_runs: dict[str, list[int]] = defaultdict(list)
    for _, manifest in manifests_with_paths:
        if manifest.get("protocol") != config["sourceProtocol"]:
            raise MaterializationError(
                f"source protocol mismatch: {manifest.get('outputId')} run {manifest.get('run')}"
            )
        if (
            manifest.get("compatibilityFingerprint")
            != config["sourceCompatibilityFingerprint"]
        ):
            raise MaterializationError(
                f"source fingerprint mismatch: {manifest.get('outputId')} run {manifest.get('run')}"
            )
        grouped_runs[str(manifest["outputId"])].append(int(manifest["run"]))
    expected_runs = list(range(1, int(config["expectedRuns"]) + 1))
    incomplete = {
        output_id: sorted(runs)
        for output_id, runs in grouped_runs.items()
        if sorted(runs) != expected_runs
    }
    if incomplete:
        raise MaterializationError(f"run coverage mismatch: {incomplete}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    created_at = utc_now()
    fingerprint = stable_sha256(
        {
            "protocol": PROTOCOL,
            "sourceCompatibilityFingerprint": config[
                "sourceCompatibilityFingerprint"
            ],
            "configSha256": sha256_file(config_path),
            "sourceManifestSha256": [
                sha256_file(path) for path, _ in manifests_with_paths
            ],
        }
    )
    try:
        all_rows: list[dict[str, Any]] = []
        all_bindings: list[dict[str, Any]] = []
        all_rejected: list[dict[str, Any]] = []
        run_manifests: list[dict[str, Any]] = []
        source_orders: set[str] = set()
        for manifest_path, manifest in sorted(
            manifests_with_paths,
            key=lambda item: (str(item[1]["outputId"]), int(item[1]["run"])),
        ):
            run_manifest, rows, bindings, rejected = materialize_run(
                generation_root=generation_root,
                staging_root=staging_root,
                final_root=output_root,
                manifest=manifest,
                manifest_path=manifest_path,
                fingerprint=fingerprint,
                expected_rows=int(config["expectedRows"]),
                require_repair_coverage=bool(config["requireRepairCoverage"]),
            )
            source_orders.add(stable_sha256([row["source"] for row in rows]))
            all_rows.extend(rows)
            all_bindings.extend(bindings)
            all_rejected.extend(rejected)
            run_manifests.append(run_manifest)
            write_json(
                staging_root
                / "manifests"
                / f"{manifest['outputId']}__run_{int(manifest['run']):02d}.json",
                run_manifest,
            )
        if len(source_orders) != 1:
            raise MaterializationError("source order differs across case-runs")

        write_jsonl(staging_root / "rows.jsonl", all_rows)
        write_jsonl(staging_root / "bindings.jsonl", all_bindings)
        write_jsonl(staging_root / "rejected_rows.jsonl", all_rejected)
        counts_by_case: Counter[str] = Counter()
        rows_by_case: Counter[str] = Counter()
        accepted_by_case: Counter[str] = Counter()
        for row in all_rows:
            rows_by_case[str(row["outputId"])] += 1
            if row["schemaValid"]:
                accepted_by_case[str(row["outputId"])] += 1
        for binding in all_bindings:
            counts_by_case[str(binding["outputId"])] += 1
        with (staging_root / "binding_counts.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["outputId", "rows", "acceptedRows", "rejectedRows", "bindings"])
            for output_id in sorted(selected_cases):
                writer.writerow(
                    [
                        output_id,
                        rows_by_case[output_id],
                        accepted_by_case[output_id],
                        rows_by_case[output_id] - accepted_by_case[output_id],
                        counts_by_case[output_id],
                    ]
                )

        status_counts = Counter(str(row["candidateStatus"]) for row in all_rows)
        rejection_counts = Counter(
            str(row["rejectionReason"]) for row in all_rejected
        )
        dataset_manifest = {
            "schemaVersion": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "createdAt": created_at,
            "official": False,
            "diagnosticOnly": True,
            "claimEligible": False,
            "goldAccessed": False,
            "generationRoot": str(generation_root),
            "outputRoot": str(output_root),
            "configPath": str(config_path),
            "configSha256": sha256_file(config_path),
            "sourceProtocol": config["sourceProtocol"],
            "sourceCompatibilityFingerprint": config[
                "sourceCompatibilityFingerprint"
            ],
            "compatibilityFingerprint": fingerprint,
            "grain": {
                "rows": "one case-run-source prediction row",
                "bindings": "one schema-valid Binding per case-run-source-bindingIndex",
                "rejectedRows": "one rejected case-run-source row",
            },
            "keys": {
                "rows": ["outputId", "run", "source", "index"],
                "bindings": [
                    "outputId",
                    "run",
                    "source",
                    "index",
                    "bindingIndex",
                ],
            },
            "counts": {
                "cases": len(selected_cases),
                "caseRuns": len(run_manifests),
                "rows": len(all_rows),
                "acceptedRows": len(all_rows) - len(all_rejected),
                "rejectedRows": len(all_rejected),
                "bindings": len(all_bindings),
                "candidateStatus": dict(sorted(status_counts.items())),
                "rejectionReasons": dict(sorted(rejection_counts.items())),
            },
            "files": {
                "rows": str(output_root / "rows.jsonl"),
                "bindings": str(output_root / "bindings.jsonl"),
                "rejectedRows": str(output_root / "rejected_rows.jsonl"),
                "bindingCounts": str(output_root / "binding_counts.tsv"),
            },
            "hashes": {
                "rows": sha256_file(staging_root / "rows.jsonl"),
                "bindings": sha256_file(staging_root / "bindings.jsonl"),
                "rejectedRows": sha256_file(staging_root / "rejected_rows.jsonl"),
                "bindingCounts": sha256_file(staging_root / "binding_counts.tsv"),
            },
            "limitations": [
                "Only generation-time recorded repairs that pass the complete six-field schema are materialized.",
                "No gold, judge, alias, tolerance, or semantic normalization is used.",
                "This diagnostic dataset does not replace strict formal predictions or rankings.",
            ],
        }
        write_json(staging_root / "dataset_manifest.json", dataset_manifest)
        report_lines = [
            "# Experiment 6 Binding 候選資料集",
            "",
            "- 性質：diagnostic only；`official=false`、`claimEligible=false`。",
            f"- 來源：`{generation_root}`。",
            f"- cases / case-runs / rows：{len(selected_cases)} / {len(run_manifests)} / {len(all_rows)}。",
            f"- 完整 schema 合法列：{len(all_rows) - len(all_rejected)}；拒絕列：{len(all_rejected)}。",
            f"- 展開後 Binding：{len(all_bindings)}。",
            "- 物化過程未讀取 gold 或 judge 輸出。",
            "",
            "## 檔案",
            "",
            "- `rows.jsonl`：每個 case-run-source 一列；`Binding` 為陣列。",
            "- `bindings.jsonl`：每個 schema-valid Binding 一列，六欄已展平。",
            "- `rejected_rows.jsonl`：未物化列及精確拒絕原因；保留無效 repair payload 供後續檢討。",
            "- `cases/*/run_*/predictions.binding_candidates.jsonl`：維持 evaluator 的 85-row prediction 結構。",
            "- `manifests/*.json`：逐 run 來源與輸出 SHA-256。",
            "- `sha256_inventory.tsv`：除自身外的完整輸出清冊。",
            "",
            "## 限制",
            "",
            "此資料集只整理既有 generation-time repair；不得當作正式生成分數。",
        ]
        (staging_root / "README.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )

        inventory_rows: list[tuple[str, int, str]] = []
        for path in sorted(staging_root.rglob("*")):
            if path.is_file() and path.name != "sha256_inventory.tsv":
                inventory_rows.append(
                    (str(path.relative_to(staging_root)), path.stat().st_size, sha256_file(path))
                )
        with (staging_root / "sha256_inventory.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["relative_path", "size_bytes", "sha256"])
            writer.writerows(inventory_rows)
        os.replace(staging_root, output_root)
        return dataset_manifest
    except Exception:
        # Keep failed staging data for forensic inspection; never delete source data.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build(args)
    except (MaterializationError, SensitivityError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "official": report["official"],
                "diagnosticOnly": report["diagnosticOnly"],
                "counts": report["counts"],
                "outputRoot": report["outputRoot"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
