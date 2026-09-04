#!/usr/bin/env python3
"""Validate the unified Experiment 6 Binding materialization v2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from analyze_experiment6_repair_sensitivity import validate_binding
from materialize_experiment6_binding_candidates import candidate_id, read_json, read_jsonl, sha256_file
from materialize_experiment6_bindings_v2 import ALLOWED_REPAIR_OPERATIONS, PROTOCOL, SAFE_REPAIR_POLICY


class UnifiedValidationError(RuntimeError):
    """Raised when a unified Binding dataset invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UnifiedValidationError(message)


def row_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return str(row["outputId"]), int(row["run"]), str(row["source"]), int(row["index"])


def binding_key(row: Mapping[str, Any]) -> tuple[str, int, str, int, int]:
    return (*row_key(row), int(row["bindingIndex"]))


def validate_output(root: Path) -> dict[str, Any]:
    root = root.resolve()
    dataset = read_json(root / "dataset_manifest.json")
    require(dataset.get("protocol") == PROTOCOL, "dataset protocol mismatch")
    require(dataset.get("safeRepairPolicy") == SAFE_REPAIR_POLICY, "repair policy mismatch")
    require(dataset.get("official") is False, "dataset marked official")
    require(dataset.get("diagnosticOnly") is True, "dataset not diagnostic-only")
    require(dataset.get("claimEligible") is False, "dataset claimEligible")
    require(dataset.get("goldAccessed") is False, "materialization accessed gold")

    for name, raw_path in dataset["files"].items():
        path = Path(raw_path)
        require(path.is_file(), f"aggregate file missing: {path}")
        require(sha256_file(path) == dataset["hashes"][name], f"aggregate SHA mismatch: {name}")
    require(dataset["hashes"]["binding"] == dataset["hashes"]["rows"], "binding/rows alias content differs")

    rows = read_jsonl(Path(dataset["files"]["binding"]))
    rows_alias = read_jsonl(Path(dataset["files"]["rows"]))
    bindings = read_jsonl(Path(dataset["files"]["bindings"]))
    rejected = read_jsonl(Path(dataset["files"]["rejectedRows"]))
    require(rows == rows_alias, "binding.jsonl and rows.jsonl differ")
    row_keys = [row_key(row) for row in rows]
    binding_keys = [binding_key(row) for row in bindings]
    rejected_keys = [row_key(row) for row in rejected]
    require(len(row_keys) == len(set(row_keys)), "duplicate row key")
    require(len(binding_keys) == len(set(binding_keys)), "duplicate Binding key")
    require(len(rejected_keys) == len(set(rejected_keys)), "duplicate rejected-row key")
    require(len({str(row["candidateId"]) for row in bindings}) == len(bindings), "duplicate candidateId")

    expected_runs = set(range(1, 11))
    runs_by_case: dict[str, set[int]] = {}
    row_counts_by_pair: Counter[tuple[str, int]] = Counter()
    source_order_by_pair: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        output_id, run, source, index = row_key(row)
        runs_by_case.setdefault(output_id, set()).add(run)
        row_counts_by_pair[(output_id, run)] += 1
        source_order_by_pair.setdefault((output_id, run), []).append(source)
        require(index == row_counts_by_pair[(output_id, run)] - 1, f"index order mismatch: {output_id} run {run}")
        require(row.get("diagnosticOnly") is True and row.get("claimEligible") is False, f"row policy mismatch: {row_key(row)}")
        operations = set((row.get("provenance") or {}).get("safeRepairOperations") or [])
        require(operations.issubset(ALLOWED_REPAIR_OPERATIONS), f"unsafe repair operation: {row_key(row)}")
    require(len(runs_by_case) == int(dataset["counts"]["cases"]) == 34, "case count mismatch")
    require(all(runs == expected_runs for runs in runs_by_case.values()), "run coverage mismatch")
    require(all(count == 85 for count in row_counts_by_pair.values()), "85-row coverage mismatch")
    source_orders = {tuple(order) for order in source_order_by_pair.values()}
    require(len(source_orders) == 1, "source order differs across case-runs")

    rejected_set = set(rejected_keys)
    invalid_set = {row_key(row) for row in rows if not row.get("schemaValid")}
    require(rejected_set == invalid_set, "rejected rows do not exactly partition invalid rows")
    binding_counts: Counter[tuple[str, int, str, int]] = Counter(row_key(row) for row in bindings)
    accepted_rows = []
    for row in rows:
        values = row.get("Binding")
        require(isinstance(values, list), f"Binding is not array: {row_key(row)}")
        require(bool(row.get("formatValid")) == bool(row.get("schemaValid")), f"format/schema flag mismatch: {row_key(row)}")
        require(len(values) == int(row["bindingCount"]), f"bindingCount mismatch: {row_key(row)}")
        require(len(values) == binding_counts[row_key(row)], f"long-form Binding mismatch: {row_key(row)}")
        if row.get("schemaValid"):
            accepted_rows.append(row)
        for value in values:
            valid, reason = validate_binding(value)
            require(valid, f"invalid row Binding {row_key(row)}: {reason}")

    fingerprint = str(dataset["compatibilityFingerprint"])
    for binding in bindings:
        value = {key: binding[key] for key in ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")}
        valid, reason = validate_binding(value)
        require(valid, f"invalid long-form Binding {binding_key(binding)}: {reason}")
        require(binding["candidateId"] == candidate_id(fingerprint, *binding_key(binding)), f"candidateId mismatch: {binding_key(binding)}")
        require(set(binding.get("safeRepairOperations") or []).issubset(ALLOWED_REPAIR_OPERATIONS), f"unsafe Binding repair: {binding_key(binding)}")

    manifest_paths = sorted((root / "manifests").glob("*.json"))
    require(len(manifest_paths) == int(dataset["counts"]["caseRuns"]) == 340, "manifest count mismatch")
    per_run_rows: list[dict[str, Any]] = []
    per_run_bindings: list[dict[str, Any]] = []
    per_run_rejected: list[dict[str, Any]] = []
    source_artifacts: set[tuple[str, str]] = set()
    manifest_pairs = set()
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        require(manifest.get("protocol") == PROTOCOL, f"run protocol mismatch: {manifest_path}")
        require(manifest.get("official") is False and manifest.get("claimEligible") is False, f"run policy mismatch: {manifest_path}")
        pair = (str(manifest["outputId"]), int(manifest["run"]))
        require(pair not in manifest_pairs, f"duplicate run manifest: {pair}")
        manifest_pairs.add(pair)
        run_files: dict[str, list[dict[str, Any]]] = {}
        for name, raw_path in manifest["files"].items():
            path = Path(raw_path)
            require(path.is_file(), f"run file missing: {path}")
            require(sha256_file(path) == manifest["hashes"][name], f"run SHA mismatch: {path}")
            if name != "predictions":
                run_files[name] = read_jsonl(path)
        require(run_files["binding"] == run_files["rows"], f"run binding/rows differ: {pair}")
        predictions = read_jsonl(Path(manifest["files"]["predictions"]))
        run_rows = run_files["binding"]
        require(len(predictions) == len(run_rows) == int(manifest["expectedRows"]) == 85, f"run row count mismatch: {pair}")
        for prediction, row in zip(predictions, run_rows):
            require((int(prediction["index"]), str(prediction["source"])) == (int(row["index"]), str(row["source"])), f"prediction order mismatch: {pair}")
            require(prediction.get("result") == row.get("Binding"), f"prediction result mismatch: {pair}")
            require(bool(prediction.get("formatValid")) == bool(row.get("schemaValid")), f"prediction format mismatch: {pair}")
        require(sum(bool(row["schemaValid"]) for row in run_rows) == int(manifest["acceptedRows"]), f"acceptedRows mismatch: {pair}")
        require(len(run_files["bindings"]) == int(manifest["bindingCount"]), f"run Binding count mismatch: {pair}")
        require(len(run_files["rejectedRows"]) == int(manifest["rejectedRows"]), f"run rejection count mismatch: {pair}")
        for path_key, hash_key in (("manifest", "manifestSha256"), ("predictions", "predictionsSha256"), ("nonformalRepair", "nonformalRepairSha256")):
            source_path = Path(manifest["source"][path_key])
            expected_hash = str(manifest["source"][hash_key])
            require(source_path.is_file(), f"source artifact missing: {source_path}")
            require(sha256_file(source_path) == expected_hash, f"source artifact SHA mismatch: {source_path}")
            source_artifacts.add((str(source_path), expected_hash))
        per_run_rows.extend(run_rows)
        per_run_bindings.extend(run_files["bindings"])
        per_run_rejected.extend(run_files["rejectedRows"])
    require(per_run_rows == rows, "aggregate/per-run rows differ")
    require(per_run_bindings == bindings, "aggregate/per-run Bindings differ")
    require(per_run_rejected == rejected, "aggregate/per-run rejections differ")

    inventory_path = root / "sha256_inventory.tsv"
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    inventoried = {row["relative_path"] for row in inventory}
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != inventory_path}
    require(inventoried == actual_files, "inventory coverage mismatch")
    for row in inventory:
        path = root / row["relative_path"]
        require(path.stat().st_size == int(row["size_bytes"]), f"inventory size mismatch: {path}")
        require(sha256_file(path) == row["sha256"], f"inventory SHA mismatch: {path}")

    declared = dataset["counts"]
    require(len(rows) == int(declared["rows"]) == 28900, "aggregate row count mismatch")
    require(len(accepted_rows) == int(declared["acceptedRows"]), "accepted count mismatch")
    require(len(rejected) == int(declared["rejectedRows"]), "rejected count mismatch")
    require(len(bindings) == int(declared["bindings"]), "Binding count mismatch")
    return {
        "status": "valid",
        "protocol": PROTOCOL,
        "root": str(root),
        "counts": {
            "cases": len(runs_by_case),
            "caseRuns": len(manifest_pairs),
            "rows": len(rows),
            "acceptedRows": len(accepted_rows),
            "acceptedRowsWithBindings": sum(int(row["bindingCount"]) > 0 for row in accepted_rows),
            "acceptedEmptyBindingRows": sum(int(row["bindingCount"]) == 0 for row in accepted_rows),
            "rejectedRows": len(rejected),
            "bindings": len(bindings),
            "sourceArtifactsVerified": len(source_artifacts),
            "inventoryFilesVerified": len(inventory),
        },
        "candidateStatus": dict(sorted(Counter(str(row["candidateStatus"]) for row in rows).items())),
        "safeRepairOperations": dict(sorted(Counter(operation for row in rows for operation in row["provenance"]["safeRepairOperations"]).items())),
        "checks": {
            "complete34CaseMatrixExcludingGpt41": True,
            "rowAndBindingKeysUnique": True,
            "completeSixFieldSchema": True,
            "runPredictionCompatibility": True,
            "sourceArtifactHashes": True,
            "inventoryCoverageAndHashes": True,
            "repairPolicyWhitelist": True,
            "goldAccessedDuringMaterialization": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_output(args.root)
    except (UnifiedValidationError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
