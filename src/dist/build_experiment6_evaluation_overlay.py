#!/usr/bin/env python3
"""Build a read-only Experiment 6 evaluation overlay under /tmp."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MISSING_GPT41_CASES = (
    "6_FinFlier_gpt4.1",
    "6_gpt4.1_z",
    "6_gpt4.1_m",
    "6_gpt4.1_d",
)
COMPLETE_STATUSES = {
    "completed",
    "completed_with_format_errors",
    "runtime_blocked",
}


class OverlayError(RuntimeError):
    """Raised when source artifacts cannot support a formal overlay."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OverlayError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(resolved)


def load_official_manifests(
    root: Path,
) -> dict[tuple[str, int], tuple[dict[str, Any], Path]]:
    manifests: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    for path in sorted((root / "manifests").glob("*.json")):
        manifest = read_json(path)
        if not manifest.get("official"):
            continue
        key = (str(manifest.get("outputId") or ""), int(manifest.get("run", -1)))
        if not key[0] or key[1] < 1:
            raise OverlayError(f"invalid manifest identity: {path}")
        if key in manifests:
            raise OverlayError(f"duplicate case/run: {key}")
        manifests[key] = (manifest, path)
    if not manifests:
        raise OverlayError(f"no official manifests: {root}")
    return manifests


def resolve_prediction(root: Path, manifest: Mapping[str, Any]) -> Path:
    raw = str((manifest.get("files") or {}).get("predictions") or "")
    declared = Path(raw)
    if declared.is_file():
        return declared
    return (
        root
        / "cases"
        / str(manifest["outputId"])
        / f"run_{int(manifest['run']):02d}"
        / declared.name
    )


def validate_selected(
    selected: Mapping[tuple[str, int], tuple[dict[str, Any], Path, Path, str]],
    expected_cases: int,
    expected_runs: int,
    expected_rows: int,
) -> list[dict[str, Any]]:
    by_case: dict[str, list[int]] = defaultdict(list)
    inventory: list[dict[str, Any]] = []
    for (output_id, run), (manifest, manifest_path, source_root, origin) in sorted(
        selected.items()
    ):
        by_case[output_id].append(run)
        status = str(manifest.get("status") or "")
        if status not in COMPLETE_STATUSES:
            raise OverlayError(f"incomplete status: {output_id} run {run}: {status}")
        prediction_path = resolve_prediction(source_root, manifest)
        if not prediction_path.is_file():
            if status != "runtime_blocked":
                raise OverlayError(f"prediction missing: {prediction_path}")
            row_count = expected_rows
            actual_hash = None
        else:
            actual_hash = sha256_file(prediction_path)
            expected_hash = str((manifest.get("hashes") or {}).get("predictions") or "")
            if actual_hash != expected_hash:
                raise OverlayError(
                    f"prediction SHA mismatch: {output_id} run {run}"
                )
            row_count = sum(
                1
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if row_count != expected_rows:
                raise OverlayError(
                    f"prediction row mismatch: {output_id} run {run}: "
                    f"{row_count} != {expected_rows}"
                )
        inventory.append(
            {
                "outputId": output_id,
                "run": run,
                "origin": origin,
                "manifestPath": logical_path(manifest_path),
                "manifestSha256": sha256_file(manifest_path),
                "predictionPath": logical_path(prediction_path),
                "predictionSha256": actual_hash,
                "rows": row_count,
            }
        )
    if len(by_case) != expected_cases:
        raise OverlayError(f"case count mismatch: {len(by_case)} != {expected_cases}")
    required_runs = list(range(1, expected_runs + 1))
    incomplete = {
        output_id: sorted(runs)
        for output_id, runs in by_case.items()
        if sorted(runs) != required_runs
    }
    if incomplete:
        raise OverlayError(f"incomplete case runs: {incomplete}")
    return inventory


def build_overlay(
    base_root: Path,
    override_root: Path,
    output_root: Path,
    *,
    expected_cases: int,
    expected_override_cases: int,
    expected_runs: int,
    expected_rows: int,
) -> dict[str, Any]:
    base_root = base_root.resolve()
    override_root = override_root.resolve()
    output_root = output_root.resolve()
    if Path("/tmp") not in output_root.parents:
        raise OverlayError(f"output root must be below /tmp: {output_root}")
    if output_root.exists():
        raise OverlayError(f"output root already exists: {output_root}")

    base = load_official_manifests(base_root)
    override = load_official_manifests(override_root)
    override_cases = {output_id for output_id, _ in override}
    base_cases = {output_id for output_id, _ in base}
    if len(override_cases) != expected_override_cases:
        raise OverlayError(
            f"override case count mismatch: {len(override_cases)} "
            f"!= {expected_override_cases}"
        )
    if not override_cases <= base_cases:
        raise OverlayError(
            f"override contains cases absent from base: "
            f"{sorted(override_cases - base_cases)}"
        )

    selected: dict[
        tuple[str, int], tuple[dict[str, Any], Path, Path, str]
    ] = {}
    for key, (manifest, path) in base.items():
        if key[0] in override_cases:
            continue
        selected[key] = (manifest, path, base_root, "base")
    for key, (manifest, path) in override.items():
        selected[key] = (manifest, path, override_root, "override")

    inventory = validate_selected(
        selected,
        expected_cases=expected_cases,
        expected_runs=expected_runs,
        expected_rows=expected_rows,
    )

    manifests_root = output_root / "manifests"
    cases_root = output_root / "cases"
    manifests_root.mkdir(parents=True)
    cases_root.mkdir()
    source_by_case: dict[str, Path] = {}
    origin_by_case: dict[str, str] = {}
    for (output_id, run), (manifest, manifest_path, source_root, origin) in sorted(
        selected.items()
    ):
        link_name = manifests_root / f"{output_id}__run_{run:02d}.json"
        link_name.symlink_to(manifest_path.resolve())
        previous = source_by_case.setdefault(output_id, source_root)
        if previous != source_root:
            raise OverlayError(f"case split across roots: {output_id}")
        origin_by_case[output_id] = origin
    for output_id, source_root in sorted(source_by_case.items()):
        source_case = source_root / "cases" / output_id
        if not source_case.is_dir():
            raise OverlayError(f"case directory missing: {source_case}")
        (cases_root / output_id).symlink_to(source_case.resolve(), target_is_directory=True)

    manifests = [
        selected[key][0]
        for key in sorted(selected, key=lambda item: (item[0], item[1]))
    ]
    status_counts = Counter(str(item.get("status") or "") for item in manifests)
    report = {
        "schemaVersion": 1,
        "protocol": "experiment6-evaluation-overlay-v1",
        "scopeLabel": "merged34-corrected12-over-old22",
        "createdAt": utc_now(),
        "updatedAt": max(str(item.get("finishedAt") or "") for item in manifests),
        "scopeComplete": True,
        "experimentMatrixComplete": False,
        "expectedFormalCases": 38,
        "selectedCases": expected_cases,
        "selectedRuns": len(manifests),
        "expectedRowsPerRun": expected_rows,
        "overrideCases": sorted(override_cases),
        "baseOnlyCases": sorted(base_cases - override_cases),
        "excludedCases": list(MISSING_GPT41_CASES),
        "sourceRoots": {
            "base": logical_path(base_root),
            "override": logical_path(override_root),
        },
        "caseOrigins": dict(sorted(origin_by_case.items())),
        "statusCounts": dict(sorted(status_counts.items())),
        "complete": True,
        "manifests": manifests,
    }
    write_json(output_root / "generation_report.json", report)
    write_json(
        output_root / "overlay_inventory.json",
        {
            "schemaVersion": 1,
            "protocol": report["protocol"],
            "createdAt": report["createdAt"],
            "scopeComplete": True,
            "experimentMatrixComplete": False,
            "artifacts": inventory,
        },
    )
    return {
        "status": "completed",
        "outputRoot": str(output_root),
        "cases": expected_cases,
        "runs": len(manifests),
        "overrideCases": len(override_cases),
        "baseCases": expected_cases - len(override_cases),
        "excludedCases": list(MISSING_GPT41_CASES),
        "generationReport": str(output_root / "generation_report.json"),
        "inventory": str(output_root / "overlay_inventory.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--override-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=34)
    parser.add_argument("--expected-override-cases", type=int, default=12)
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--expected-rows", type=int, default=85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_overlay(
            args.base_root,
            args.override_root,
            args.output_root,
            expected_cases=args.expected_cases,
            expected_override_cases=args.expected_override_cases,
            expected_runs=args.expected_runs,
            expected_rows=args.expected_rows,
        )
    except (OverlayError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
