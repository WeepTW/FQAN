#!/usr/bin/env python3
"""Validate the three-row Mistral target-last smoke gate without gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiment6_mistral_chat_binding_repair import repair_unique_binding


EXPECTED = {
    "dynamic_echo_econ020": ("6_mistral_base_d", "Econ_020"),
    "many_echo_econ066": ("6_mistral_base_m", "Econ_066"),
    "dynamic_success_econ026": ("6_mistral_base_d", "Econ_026"),
}


class SmokeError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SmokeError(f"expected JSON object: {path}")
    return value


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SmokeError(f"expected JSON object: {path}:{number}")
        rows.append(value)
    return rows


def validate(root: Path) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    scientific_fingerprints: set[str] = set()
    for subdir, (case_id, source) in EXPECTED.items():
        smoke_root = root / subdir
        manifests = sorted((smoke_root / "manifests").glob("*.json"))
        if len(manifests) != 1:
            raise SmokeError(f"{subdir}: expected one manifest, got {len(manifests)}")
        manifest = read_json(manifests[0])
        if manifest.get("outputId") != case_id or int(manifest.get("run", 0)) != 1:
            raise SmokeError(f"{subdir}: case/run identity mismatch")
        if manifest.get("status") not in {
            "completed",
            "completed_with_format_errors",
        } or manifest.get("runtimeBlockedRows") != 0:
            raise SmokeError(f"{subdir}: generation did not complete")
        if manifest.get("declaredRoute") != "direct-binding" or manifest.get("effectiveRoute") != "direct-binding":
            raise SmokeError(f"{subdir}: route mismatch")
        if manifest.get("adapter") is not None or manifest.get("converterModel") is not None:
            raise SmokeError(f"{subdir}: direct route loaded adapter/converter")
        compatibility = read_json(smoke_root / "compatibility_fingerprint.json")
        fingerprint = str(compatibility.get("sha256") or "")
        if fingerprint != str(manifest.get("compatibilityFingerprint") or ""):
            raise SmokeError(f"{subdir}: manifest/compatibility fingerprint mismatch")
        material = compatibility.get("material")
        if not isinstance(material, dict):
            raise SmokeError(f"{subdir}: compatibility material missing")
        scientific_material = dict(material)
        scientific_material.pop("executionMode", None)
        scientific_fingerprints.add(sha256_json(scientific_material))
        fingerprints.add(str(manifest.get("compatibilityFingerprint") or ""))
        prediction_path = Path(str(manifest["files"]["predictions"]))
        predictions = read_jsonl(prediction_path)
        if len(predictions) != 1 or predictions[0].get("source") != source:
            raise SmokeError(f"{subdir}: source coverage mismatch")
        prediction = predictions[0]
        raw = str(prediction.get("rawResponse") or "")
        repaired = repair_unique_binding(raw)
        strict = prediction.get("formatValid") is True
        result = prediction.get("result")
        strict_nonempty = strict and isinstance(result, list) and bool(result)
        if not strict_nonempty and not repaired.get("available"):
            raise SmokeError(
                f"{subdir}: no strict or uniquely repairable Binding: "
                f"{repaired.get('reason')}"
            )
        if repaired.get("reason") == "prompt_echo_guard":
            raise SmokeError(f"{subdir}: prompt echo detected")
        reports.append(
            {
                "smoke": subdir,
                "case": case_id,
                "source": source,
                "strictValidNonempty": strict_nonempty,
                "repairAvailable": bool(repaired.get("available")),
                "repairMethod": repaired.get("method"),
                "rawCharacters": len(raw),
            }
        )
    if len(fingerprints) != len(EXPECTED) or "" in fingerprints:
        raise SmokeError(f"smoke subset fingerprint mismatch: {sorted(fingerprints)}")
    if len(scientific_fingerprints) != 1:
        raise SmokeError(
            "smoke scientific compatibility mismatch: "
            f"{sorted(scientific_fingerprints)}"
        )
    return {
        "status": "passed",
        "protocol": "experiment6-mistral-chat-v4-smoke-gate-v1",
        "scientificCompatibilityFingerprint": next(iter(scientific_fingerprints)),
        "smokeCompatibilityFingerprints": sorted(fingerprints),
        "smokes": reports,
        "goldAccessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.root.resolve())
    except (SmokeError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
