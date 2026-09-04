#!/usr/bin/env python3
"""Validate the isolated Experiment 6 historical/direct diagnostic reruns.

The validator is intentionally independent from generation and evaluation.  It
checks the row/source/run/seed/file/hash/route contracts required by the rerun
protocol and writes a machine-readable audit without changing any score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SOURCES = 85
EXPECTED_RUNS = tuple(range(1, 11))
SEED_BASE = 2026073100
FILE_FIELDS = ("predictions", "rawResponse", "prompts")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    return rows


def source_value(row: dict[str, Any]) -> str | None:
    for key in ("Source", "source", "sourceId"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def validate_root(
    root: Path,
    expected_cases: set[str],
    route_mode: str,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    manifests = sorted((root / "manifests").glob("*.json"))
    records: list[dict[str, Any]] = []
    per_case_runs: dict[str, set[int]] = defaultdict(set)

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = str(manifest.get("outputId"))
        run = int(manifest.get("run", -1))
        prefix = f"{manifest_path.name}"
        per_case_runs[case].add(run)

        if case not in expected_cases:
            failures.append(f"{prefix}: unexpected case {case}")
        if run not in EXPECTED_RUNS:
            failures.append(f"{prefix}: unexpected run {run}")
        if manifest.get("seed") != SEED_BASE + run:
            failures.append(f"{prefix}: seed mismatch")
        if manifest.get("expectedRows") != EXPECTED_SOURCES:
            failures.append(f"{prefix}: expectedRows != {EXPECTED_SOURCES}")
        if manifest.get("baseRouteMode") != route_mode:
            failures.append(f"{prefix}: baseRouteMode mismatch")

        if route_mode == "historical":
            if manifest.get("effectiveRoute") != "retriever-converter":
                failures.append(f"{prefix}: historical effectiveRoute mismatch")
            if manifest.get("converterModel") != "gpt-5.5":
                failures.append(f"{prefix}: converterModel is not gpt-5.5")
            if manifest.get("reasoningEffort") != "medium":
                failures.append(f"{prefix}: converter reasoning is not medium")
        else:
            if manifest.get("effectiveRoute") != "direct-diagnostic-native":
                failures.append(f"{prefix}: direct effectiveRoute mismatch")
            if manifest.get("converterModel") is not None:
                failures.append(f"{prefix}: direct route used a converter")
            if manifest.get("adapter") is not None:
                failures.append(f"{prefix}: direct route used an adapter")

        files = manifest.get("files", {})
        hashes = manifest.get("hashes", {})
        for field, expected_hash in hashes.items():
            file_value = files.get(field)
            if not isinstance(file_value, str) or not Path(file_value).is_file():
                failures.append(f"{prefix}: hashed file {field} is missing")
            elif sha256(Path(file_value)) != expected_hash:
                failures.append(f"{prefix}: {field} SHA-256 mismatch")
        for field in FILE_FIELDS:
            file_value = files.get(field)
            if not isinstance(file_value, str):
                failures.append(f"{prefix}: missing files.{field}")
                continue
            path = Path(file_value)
            if not path.is_file():
                failures.append(f"{prefix}: missing {field} file")
                continue
            if sha256(path) != hashes.get(field):
                failures.append(f"{prefix}: {field} SHA-256 mismatch")
            try:
                rows = jsonl_rows(path)
            except (ValueError, json.JSONDecodeError) as exc:
                failures.append(str(exc))
                continue
            if len(rows) != EXPECTED_SOURCES:
                failures.append(f"{prefix}: {field} rows={len(rows)}")
            sources = [source_value(row) for row in rows]
            if None in sources or len(set(sources)) != EXPECTED_SOURCES:
                failures.append(f"{prefix}: {field} lacks 85 unique Sources")

        if route_mode == "direct-diagnostic":
            runtime_path = Path(str(files.get("runtime", "")))
            if runtime_path.is_file():
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                stages = runtime.get("stages", [])
                raw = stages[0].get("raw", {}) if stages else {}
                direct_contract = {
                    "structured_output": "off",
                    "converter_used": False,
                    "generation_cache_used": False,
                    "prompt_route": "direct",
                    "max_input_tokens": 8192,
                    "max_new_tokens": 4096,
                }
                for key, expected in direct_contract.items():
                    if raw.get(key) != expected:
                        failures.append(
                            f"{prefix}: direct runtime {key}={raw.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                if raw.get("use_adapter") is not False:
                    failures.append(f"{prefix}: direct runtime used an adapter")
                if raw.get("row_checkpoint_rows") != EXPECTED_SOURCES:
                    failures.append(f"{prefix}: direct checkpoint lacks 85 rows")

        records.append(
            {
                "case": case,
                "run": run,
                "seed": manifest.get("seed"),
                "status": manifest.get("status"),
                "expectedRows": manifest.get("expectedRows"),
                "acceptedRows": manifest.get("acceptedRows"),
                "rejectedRows": manifest.get("rejectedRows"),
                "runtimeBlockedRows": manifest.get("runtimeBlockedRows"),
                "manifest": str(manifest_path.resolve()),
            }
        )

    expected_manifest_count = len(expected_cases) * len(EXPECTED_RUNS)
    if len(manifests) != expected_manifest_count:
        failures.append(
            f"{root}: manifest count={len(manifests)}, expected={expected_manifest_count}"
        )
    if set(per_case_runs) != expected_cases:
        failures.append(f"{root}: case set mismatch")
    for case in sorted(expected_cases):
        if per_case_runs.get(case, set()) != set(EXPECTED_RUNS):
            failures.append(f"{root}: {case} does not have runs 1..10")

    return (
        {
            "root": str(root.resolve()),
            "routeMode": route_mode,
            "expectedCases": sorted(expected_cases),
            "manifestCount": len(manifests),
            "predictionCount": sum(int(row.get("expectedRows") or 0) for row in records),
            "acceptedRows": sum(int(row.get("acceptedRows") or 0) for row in records),
            "rejectedRows": sum(int(row.get("rejectedRows") or 0) for row in records),
            "runtimeBlockedRows": sum(
                int(row.get("runtimeBlockedRows") or 0) for row in records
            ),
            "statuses": dict(Counter(str(row.get("status")) for row in records)),
            "records": records,
        },
        failures,
    )


def parse_cases(value: str) -> set[str]:
    cases = {item.strip() for item in value.split(",") if item.strip()}
    if not cases:
        raise argparse.ArgumentTypeError("at least one case is required")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--historical-cases", type=parse_cases, required=True)
    parser.add_argument("--direct-cases", type=parse_cases, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    historical, historical_failures = validate_root(
        args.historical_root, args.historical_cases, "historical"
    )
    direct, direct_failures = validate_root(
        args.direct_root, args.direct_cases, "direct-diagnostic"
    )
    failures = historical_failures + direct_failures
    report = {
        "protocol": "experiment6-zero-rerun-integrity-v1",
        "status": "passed" if not failures else "failed",
        "expectedFreshPredictions": 11050,
        "actualFreshPredictions": historical["predictionCount"] + direct["predictionCount"],
        "historical": historical,
        "direct": direct,
        "failures": failures,
    }
    if report["actualFreshPredictions"] != report["expectedFreshPredictions"]:
        report["status"] = "failed"
        report["failures"].append("fresh prediction total is not 11,050")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "expectedFreshPredictions", "actualFreshPredictions", "failures")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
