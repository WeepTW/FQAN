#!/usr/bin/env python3
"""Build a gold-free repaired Binding projection for Mistral chat reruns.

Generation artifacts are immutable.  This tool copies prediction records into
an isolated diagnostic root, preserves strict validity, applies only recorded
generation-time repair or the unique six-field-object repair, and represents
unrecoverable rows as valid empty predictions for evaluation.  It never reads
gold or judge assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiment6_mistral_chat_binding_repair import BINDING_KEYS, repair_unique_binding


PROTOCOL = "experiment6-mistral-chat-repaired-projection-v1"
EXPECTED_CASES = {"6_mistral_base_m", "6_mistral_base_d"}
EXPECTED_RUNS = set(range(1, 11))
EXPECTED_ROWS = 85


class ProjectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProjectionError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProjectionError(f"expected JSON object: {path}:{number}")
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


def valid_binding(binding: Any) -> bool:
    return isinstance(binding, dict) and set(binding) == set(BINDING_KEYS)


def valid_result(result: Any) -> bool:
    return isinstance(result, list) and all(valid_binding(binding) for binding in result)


def resolve_artifact(root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    raw = str((manifest.get("files") or {}).get(name) or "")
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    relocated = (
        root
        / "cases"
        / str(manifest["outputId"])
        / f"run_{int(manifest['run']):02d}"
        / path.name
    )
    if not relocated.is_file():
        raise ProjectionError(f"missing artifact {name}: {relocated}")
    return relocated.resolve()


def verified_artifact(root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    path = resolve_artifact(root, manifest, name)
    expected = str((manifest.get("hashes") or {}).get(name) or "")
    if not expected or sha256_file(path) != expected:
        raise ProjectionError(
            f"{manifest.get('outputId')} run {manifest.get('run')}: {name} SHA mismatch"
        )
    return path


def repair_by_key(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(row["index"]), str(row["source"])): row
        for row in read_jsonl(path)
    }


def choose_result(
    prediction: Mapping[str, Any], recorded: Mapping[str, Any] | None
) -> tuple[list[dict[str, Any]], str, list[str]]:
    original = prediction.get("result")
    if prediction.get("formatValid") is True and valid_result(original):
        return list(original), "strict-valid-preserved", []
    if recorded is not None:
        payload = recorded.get("repairedPayload") or recorded.get("repairPayload")
        result = payload.get("result") if isinstance(payload, dict) else None
        if recorded.get("formatValid") is True and valid_result(result):
            method = str(recorded.get("repairMethod") or "generation-recorded-repair")
            return list(result), "generation-repair-preserved", [method]
    recovered = repair_unique_binding(str(prediction.get("rawResponse") or ""))
    payload = recovered.get("payload") if recovered.get("available") else None
    result = payload.get("result") if isinstance(payload, dict) else None
    if valid_result(result):
        return list(result), "unique-object-recovered", [str(recovered["method"])]
    return [], "unrecoverable-as-empty", [str(recovered.get("reason") or "unavailable")]


def link_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ProjectionError(f"artifact copy changed bytes: {source}")


def write_inventory(root: Path) -> None:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_inventory.tsv":
            rows.append((str(path.relative_to(root)), path.stat().st_size, sha256_file(path)))
    with (root / "sha256_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(rows)


def build(generation_root: Path, output_root: Path) -> dict[str, Any]:
    generation_root = generation_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise ProjectionError(f"output root exists: {output_root}")
    manifest_entries = [
        (path, read_json(path))
        for path in sorted((generation_root / "manifests").glob("*.json"))
    ]
    manifest_entries = [(path, item) for path, item in manifest_entries if item.get("official")]
    cases: dict[str, set[int]] = defaultdict(set)
    for _, manifest in manifest_entries:
        cases[str(manifest["outputId"])].add(int(manifest["run"]))
    if set(cases) != EXPECTED_CASES or any(runs != EXPECTED_RUNS for runs in cases.values()):
        raise ProjectionError(f"incomplete case/run coverage: {dict(cases)}")

    source_snapshot = generation_root / "generation_config.snapshot.json"
    if not source_snapshot.is_file():
        raise ProjectionError("generation_config.snapshot.json missing")
    fingerprint = stable_sha256(
        {
            "protocol": PROTOCOL,
            "materializerSha256": sha256_file(Path(__file__).resolve()),
            "generationSnapshotSha256": sha256_file(source_snapshot),
            "sourceManifests": [sha256_file(path) for path, _ in manifest_entries],
        }
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    totals: Counter[str] = Counter()
    rows_all: list[dict[str, Any]] = []
    bindings_all: list[dict[str, Any]] = []
    audit_all: list[dict[str, Any]] = []
    try:
        link_verified(source_snapshot, staging / "generation_config.snapshot.json")
        source_fp = generation_root / "compatibility_fingerprint.json"
        if source_fp.is_file():
            link_verified(source_fp, staging / "source_compatibility_fingerprint.json")
        for source_manifest_path, source_manifest in sorted(
            manifest_entries,
            key=lambda item: (str(item[1]["outputId"]), int(item[1]["run"])),
        ):
            output_id = str(source_manifest["outputId"])
            run = int(source_manifest["run"])
            predictions_path = verified_artifact(generation_root, source_manifest, "predictions")
            repair_path = verified_artifact(generation_root, source_manifest, "nonformalRepair")
            predictions = read_jsonl(predictions_path)
            repairs = repair_by_key(repair_path)
            if len(predictions) != EXPECTED_ROWS:
                raise ProjectionError(f"{output_id} run {run}: expected 85 rows")
            sources = [str(row.get("source") or "") for row in predictions]
            if len(set(sources)) != EXPECTED_ROWS:
                raise ProjectionError(f"{output_id} run {run}: duplicate/missing Source")

            relative = Path("cases") / output_id / f"run_{run:02d}"
            stage_dir = staging / relative
            final_dir = output_root / relative
            repaired_predictions: list[dict[str, Any]] = []
            run_audit: list[dict[str, Any]] = []
            run_bindings: list[dict[str, Any]] = []
            statuses: Counter[str] = Counter()
            for prediction in predictions:
                key = (int(prediction["index"]), str(prediction["source"]))
                result, status, operations = choose_result(prediction, repairs.get(key))
                statuses[status] += 1
                totals[status] += 1
                item = dict(prediction)
                item["result"] = result
                item["formatValid"] = True
                item["strictFormatValid"] = bool(prediction.get("formatValid"))
                item["materialization"] = {
                    "protocol": PROTOCOL,
                    "status": status,
                    "operations": operations,
                    "sourcePredictionSha256": sha256_file(predictions_path),
                    "goldAccessed": False,
                    "diagnosticOnly": True,
                }
                repaired_predictions.append(item)
                audit = {
                    "outputId": output_id,
                    "run": run,
                    "seed": source_manifest["seed"],
                    "index": key[0],
                    "source": key[1],
                    "strictFormatValid": bool(prediction.get("formatValid")),
                    "status": status,
                    "bindingCount": len(result),
                    "operations": operations,
                    "rawResponseSha256": prediction.get("rawResponseSha256"),
                    "goldAccessed": False,
                }
                run_audit.append(audit)
                rows_all.append({**audit, "Binding": result})
                audit_all.append(audit)
                for binding_index, binding in enumerate(result):
                    record = {
                        "outputId": output_id,
                        "run": run,
                        "seed": source_manifest["seed"],
                        "index": key[0],
                        "source": key[1],
                        "bindingIndex": binding_index,
                        **binding,
                        "materializationStatus": status,
                    }
                    run_bindings.append(record)
                    bindings_all.append(record)

            predictions_out = stage_dir / "predictions.repaired.jsonl"
            audit_out = stage_dir / "repair_audit.jsonl"
            bindings_out = stage_dir / "bindings.jsonl"
            write_jsonl(predictions_out, repaired_predictions)
            write_jsonl(audit_out, run_audit)
            write_jsonl(bindings_out, run_bindings)

            copied_files: dict[str, Path] = {}
            for name in ("rawResponse", "prompts", "runtime", "formatReport", "nonformalRepair", "stage1Raw"):
                if name not in (source_manifest.get("files") or {}):
                    continue
                source = verified_artifact(generation_root, source_manifest, name)
                destination = stage_dir / source.name
                link_verified(source, destination)
                copied_files[name] = destination
            manifest = dict(source_manifest)
            manifest.update(
                {
                    "protocol": PROTOCOL,
                    "official": False,
                    "compatibilityFingerprint": fingerprint,
                    "status": "completed",
                    "expectedRows": EXPECTED_ROWS,
                    "acceptedRows": EXPECTED_ROWS,
                    "rejectedRows": 0,
                    "formatComplianceRate": 1.0,
                    "diagnosticOnly": True,
                    "claimEligible": False,
                    "goldAccessed": False,
                    "sourceGenerationManifest": str(source_manifest_path),
                    "sourceGenerationManifestSha256": sha256_file(source_manifest_path),
                    "materializationStatusCounts": dict(sorted(statuses.items())),
                }
            )
            files = {name: str(output_root / relative / path.name) for name, path in copied_files.items()}
            files.update(
                {
                    "predictions": str(final_dir / predictions_out.name),
                    "repairAudit": str(final_dir / audit_out.name),
                    "bindings": str(final_dir / bindings_out.name),
                }
            )
            hashes = {name: sha256_file(path) for name, path in copied_files.items()}
            hashes.update(
                {
                    "predictions": sha256_file(predictions_out),
                    "repairAudit": sha256_file(audit_out),
                    "bindings": sha256_file(bindings_out),
                }
            )
            manifest["files"] = files
            manifest["hashes"] = hashes
            write_json(staging / "manifests" / f"{output_id}__run_{run:02d}.json", manifest)

        write_jsonl(staging / "rows.jsonl", rows_all)
        write_jsonl(staging / "bindings.jsonl", bindings_all)
        write_jsonl(staging / "repair_audit.jsonl", audit_all)
        dataset = {
            "schemaVersion": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "official": False,
            "diagnosticOnly": True,
            "claimEligible": False,
            "goldAccessed": False,
            "generationRoot": str(generation_root),
            "outputRoot": str(output_root),
            "compatibilityFingerprint": fingerprint,
            "counts": {
                "cases": len(EXPECTED_CASES),
                "caseRuns": len(manifest_entries),
                "rows": len(rows_all),
                "bindings": len(bindings_all),
                "materializationStatus": dict(sorted(totals.items())),
            },
            "files": {
                "rows": str(output_root / "rows.jsonl"),
                "bindings": str(output_root / "bindings.jsonl"),
                "repairAudit": str(output_root / "repair_audit.jsonl"),
            },
            "sourceGenerationSnapshotSha256": sha256_file(source_snapshot),
            "materializerSha256": sha256_file(Path(__file__).resolve()),
            "limitations": [
                "Diagnostic only; it does not replace the formal ranking.",
                "Unrecoverable rows become valid empty predictions and therefore retain false negatives.",
                "No gold or judge asset is read during materialization.",
            ],
        }
        write_json(staging / "dataset_manifest.json", dataset)
        write_json(staging / "compatibility_fingerprint.json", {"protocol": PROTOCOL, "sha256": fingerprint})
        write_inventory(staging)
        os.replace(staging, output_root)
        return dataset
    except Exception:
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build(args.generation_root, args.output_root)
    except (ProjectionError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "complete", "counts": report["counts"], "outputRoot": report["outputRoot"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
