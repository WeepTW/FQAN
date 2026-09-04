#!/usr/bin/env python3
"""Validate an Experiment 6 diagnostic Binding-candidate dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from analyze_experiment6_repair_sensitivity import validate_binding
from materialize_experiment6_binding_candidates import (
    PROTOCOL,
    candidate_id,
    read_json,
    read_jsonl,
    sha256_file,
)


class ValidationError(RuntimeError):
    """Raised when a candidate dataset invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def row_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["outputId"]),
        int(row["run"]),
        str(row["source"]),
        int(row["index"]),
    )


def binding_key(row: Mapping[str, Any]) -> tuple[str, int, str, int, int]:
    return (*row_key(row), int(row["bindingIndex"]))


def validate_output(root: Path) -> dict[str, Any]:
    root = root.resolve()
    dataset = read_json(root / "dataset_manifest.json")
    require(dataset.get("protocol") == PROTOCOL, "dataset protocol mismatch")
    require(dataset.get("official") is False, "diagnostic dataset marked official")
    require(dataset.get("diagnosticOnly") is True, "diagnosticOnly must be true")
    require(dataset.get("claimEligible") is False, "claimEligible must be false")
    require(dataset.get("goldAccessed") is False, "materialization must not access gold")

    for name, raw_path in dataset["files"].items():
        path = Path(raw_path)
        require(path.is_file(), f"dataset file missing: {path}")
        require(
            sha256_file(path) == dataset["hashes"][name],
            f"dataset file SHA mismatch: {name}",
        )

    rows = read_jsonl(Path(dataset["files"]["rows"]))
    bindings = read_jsonl(Path(dataset["files"]["bindings"]))
    rejected = read_jsonl(Path(dataset["files"]["rejectedRows"]))
    row_keys = [row_key(row) for row in rows]
    binding_keys = [binding_key(row) for row in bindings]
    rejected_keys = [row_key(row) for row in rejected]
    require(len(row_keys) == len(set(row_keys)), "duplicate aggregate row key")
    require(len(binding_keys) == len(set(binding_keys)), "duplicate aggregate Binding key")
    require(len(rejected_keys) == len(set(rejected_keys)), "duplicate rejected-row key")
    require(len({str(row["candidateId"]) for row in bindings}) == len(bindings), "duplicate candidateId")

    accepted_rows = [row for row in rows if row.get("schemaValid") is True]
    require(
        set(rejected_keys) == {row_key(row) for row in rows if not row.get("schemaValid")},
        "rejected rows do not exactly partition invalid rows",
    )
    bindings_by_row: Counter[tuple[str, int, str, int]] = Counter(
        row_key(binding) for binding in bindings
    )
    for row in rows:
        values = row.get("Binding")
        require(isinstance(values, list), f"Binding is not an array: {row_key(row)}")
        require(len(values) == int(row["bindingCount"]), f"bindingCount mismatch: {row_key(row)}")
        require(len(values) == bindings_by_row[row_key(row)], f"long-form Binding mismatch: {row_key(row)}")
        for value in values:
            valid, reason = validate_binding(value)
            require(valid, f"invalid row Binding {row_key(row)}: {reason}")

    fingerprint = str(dataset["compatibilityFingerprint"])
    for binding in bindings:
        value = {key: binding[key] for key in ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")}
        valid, reason = validate_binding(value)
        require(valid, f"invalid long-form Binding {binding_key(binding)}: {reason}")
        expected_id = candidate_id(fingerprint, *binding_key(binding))
        require(binding["candidateId"] == expected_id, f"candidateId mismatch: {binding_key(binding)}")

    manifest_paths = sorted((root / "manifests").glob("*.json"))
    per_run_rows: list[dict[str, Any]] = []
    per_run_bindings: list[dict[str, Any]] = []
    per_run_rejected: list[dict[str, Any]] = []
    run_pairs: set[tuple[str, int]] = set()
    source_artifacts: set[tuple[str, str]] = set()
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        require(manifest.get("official") is False, f"run manifest marked official: {manifest_path}")
        require(manifest.get("claimEligible") is False, f"run manifest claimEligible: {manifest_path}")
        pair = (str(manifest["outputId"]), int(manifest["run"]))
        require(pair not in run_pairs, f"duplicate run manifest: {pair}")
        run_pairs.add(pair)
        run_files: dict[str, list[dict[str, Any]]] = {}
        for name, raw_path in manifest["files"].items():
            path = Path(raw_path)
            require(path.is_file(), f"run file missing: {path}")
            require(sha256_file(path) == manifest["hashes"][name], f"run file SHA mismatch: {path}")
            if name != "predictions":
                run_files[name] = read_jsonl(path)
        predictions = read_jsonl(Path(manifest["files"]["predictions"]))
        run_rows = run_files["rows"]
        require(len(predictions) == len(run_rows) == int(manifest["expectedRows"]), f"run row count mismatch: {pair}")
        for prediction, row in zip(predictions, run_rows):
            require((int(prediction["index"]), str(prediction["source"])) == (int(row["index"]), str(row["source"])), f"prediction order mismatch: {pair}")
            require(prediction.get("result") == row.get("Binding"), f"prediction Binding mismatch: {pair}")
            require(bool(prediction.get("formatValid")) == bool(row.get("schemaValid")), f"formatValid mismatch: {pair}")
        require(sum(bool(row["schemaValid"]) for row in run_rows) == int(manifest["acceptedRows"]), f"acceptedRows mismatch: {pair}")
        require(len(run_files["bindings"]) == int(manifest["bindingCount"]), f"bindingCount manifest mismatch: {pair}")
        require(len(run_files["rejectedRows"]) == int(manifest["rejectedRows"]), f"rejectedRows manifest mismatch: {pair}")
        for path_key, hash_key in (("manifest", "manifestSha256"), ("predictions", "predictionsSha256"), ("nonformalRepair", "nonformalRepairSha256")):
            source_path = Path(manifest["source"][path_key])
            expected_hash = str(manifest["source"][hash_key])
            require(source_path.is_file(), f"source artifact missing: {source_path}")
            require(sha256_file(source_path) == expected_hash, f"source artifact SHA mismatch: {source_path}")
            source_artifacts.add((str(source_path), expected_hash))
        per_run_rows.extend(run_rows)
        per_run_bindings.extend(run_files["bindings"])
        per_run_rejected.extend(run_files["rejectedRows"])
    require(per_run_rows == rows, "aggregate rows differ from ordered per-run rows")
    require(per_run_bindings == bindings, "aggregate bindings differ from ordered per-run bindings")
    require(per_run_rejected == rejected, "aggregate rejected rows differ from ordered per-run rows")

    inventory_path = root / "sha256_inventory.tsv"
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    inventoried = {row["relative_path"] for row in inventory}
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != inventory_path
    }
    require(inventoried == actual_files, "inventory path coverage mismatch")
    for row in inventory:
        path = root / row["relative_path"]
        require(path.stat().st_size == int(row["size_bytes"]), f"inventory size mismatch: {path}")
        require(sha256_file(path) == row["sha256"], f"inventory SHA mismatch: {path}")

    declared = dataset["counts"]
    require(len(rows) == int(declared["rows"]), "dataset row count mismatch")
    require(len(accepted_rows) == int(declared["acceptedRows"]), "dataset accepted count mismatch")
    require(len(rejected) == int(declared["rejectedRows"]), "dataset rejected count mismatch")
    require(len(bindings) == int(declared["bindings"]), "dataset Binding count mismatch")
    positive_rows = sum(bool(row["bindingCount"]) for row in accepted_rows)
    empty_rows = len(accepted_rows) - positive_rows
    return {
        "status": "valid",
        "protocol": PROTOCOL,
        "root": str(root),
        "counts": {
            "cases": len({key[0] for key in row_keys}),
            "caseRuns": len(run_pairs),
            "rows": len(rows),
            "acceptedRows": len(accepted_rows),
            "acceptedRowsWithBindings": positive_rows,
            "acceptedEmptyBindingRows": empty_rows,
            "rejectedRows": len(rejected),
            "bindings": len(bindings),
            "sourceArtifactsVerified": len(source_artifacts),
            "inventoryFilesVerified": len(inventory),
        },
        "candidateStatus": dict(sorted(Counter(str(row["candidateStatus"]) for row in rows).items())),
        "checks": {
            "aggregateHashes": True,
            "uniqueKeys": True,
            "completeSixFieldSchema": True,
            "runPredictionCompatibility": True,
            "sourceArtifactHashes": True,
            "inventoryCoverageAndHashes": True,
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
    except (ValidationError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
