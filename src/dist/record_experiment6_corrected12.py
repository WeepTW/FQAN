#!/usr/bin/env python3
"""Record a completed corrected-12 run in the canonical docs/assets log index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_narrative2_reference_aligned_v5 as evaluator_v5
from experiment6_corrected12 import status
from experiment6_paths import PATHS


MINIMUM_CASES = [
    "6_flan_base_z",
    "6_flan_base_m",
    "6_flan_base_d",
    "6_mistral_base_z",
    "6_mistral_base_m",
    "6_mistral_base_d",
    "6_t5gemma2_base_z",
    "6_t5gemma2_base_m",
    "6_t5gemma2_base_d",
    "6_FinFlier_flan_base",
    "6_FinFlier_mistral_base",
    "6_FinFlier_t5gemma2_base",
]


def full_38_cases() -> list[str]:
    cases = [
        f"6_{family}_{suffix}"
        for family in ("flan", "mistral", "t5gemma2")
        for suffix in ("z", "m", "d")
    ]
    cases.extend(
        f"6_{family}_base_{suffix}"
        for family in ("flan", "mistral", "t5gemma2")
        for suffix in ("z", "m", "d")
    )
    cases.extend(f"6_gpt5.5_{suffix}" for suffix in ("z", "m", "d"))
    cases.extend(
        f"6_FinFlier_{family}_{suffix}"
        for family in ("flan", "mistral", "t5gemma2")
        for suffix in ("z", "m", "d")
    )
    cases.extend(
        [
            "6_FinFlier_flan_base",
            "6_FinFlier_mistral_base",
            "6_FinFlier_t5gemma2_base",
            "6_FinFlier_gpt5.5",
            "6_FinFlier_gpt4.1",
            "6_gpt4.1_z",
            "6_gpt4.1_m",
            "6_gpt4.1_d",
        ]
    )
    if len(cases) != 38 or len(set(cases)) != 38:
        raise AssertionError("full formal case list must contain 38 unique IDs")
    return cases


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_evaluation_report(root: Path, evaluation: dict[str, Any]) -> None:
    errors: list[str] = []
    expected_method = evaluator_v5.method_metadata()
    method = evaluation.get("method")
    judge = evaluation.get("judge")
    if evaluation.get("protocol") != evaluator_v5.PROTOCOL:
        errors.append("evaluation protocol is not reference-aligned v5.1")
    if (
        not isinstance(method, dict)
        or method.get("methodCompatibilitySha256")
        != expected_method["methodCompatibilitySha256"]
    ):
        errors.append("evaluation method fingerprint mismatch")
    if not isinstance(judge, dict):
        errors.append("evaluation judge metadata missing")
    else:
        if judge.get("model") != "gpt-5.5":
            errors.append("evaluation judge model is not gpt-5.5")
        if judge.get("reasoningEffort") != "medium":
            errors.append("evaluation judge effort is not medium")
        if judge.get("minimumConfidence") != 0.8:
            errors.append("evaluation judge confidence is not 0.8")
    evaluation_config_path = PATHS.resolve(
        "repo", "config/experiment6_narrative2_evaluation_corrected_12.json"
    )
    if evaluation.get("sourceEvaluationConfigProtocol") != (
        "experiment6-narrative2-reference-aligned-v5.1-corrected12"
    ):
        errors.append("source evaluation config protocol mismatch")
    if evaluation.get("sourceEvaluationConfigSchemaVersion") != 6:
        errors.append("source evaluation config schema version mismatch")
    if evaluation.get("evaluationConfigSha256") != sha256_file(
        evaluation_config_path
    ):
        errors.append("source evaluation config SHA-256 mismatch")
    expected_scalars = {
        "status": "completed",
        "completedCases": 12,
        "completedCaseRuns": 120,
        "formalPredictions": 10200,
    }
    for name, expected in expected_scalars.items():
        if evaluation.get(name) != expected:
            errors.append(
                f"evaluation {name}={evaluation.get(name)!r} expected={expected!r}"
            )
    generation_root = evaluation.get("generationRoot")
    if not generation_root or Path(str(generation_root)).resolve() != root.resolve():
        errors.append("evaluation generationRoot mismatch")
    tables = evaluation.get("tables")
    evaluation_root = root / "evaluation_reference_aligned_v5"
    required_tables = {
        "scorecard",
        "fieldScorecard",
        "ablationScorecard",
        "perRun",
    }
    if not isinstance(tables, dict):
        errors.append("evaluation tables metadata missing")
    else:
        for name in sorted(required_tables):
            raw = tables.get(name)
            if not raw:
                errors.append(f"evaluation table missing: {name}")
                continue
            path = Path(str(raw)).resolve()
            if not path.is_relative_to(evaluation_root.resolve()) or not path.is_file():
                errors.append(f"evaluation table invalid: {name}")
    if errors:
        raise RuntimeError("refusing incompatible evaluation report: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    run_status = status(root)
    if not run_status["complete"]:
        raise RuntimeError("refusing to log incomplete corrected-12 run")

    evaluation_report_path = root / "evaluation_reference_aligned_v5" / "evaluation_report.json"
    inventory_path = root / "sha256_inventory.tsv"
    compatibility_path = root / "compatibility_fingerprint.json"
    for required in (evaluation_report_path, inventory_path, compatibility_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    evaluation = read_json(evaluation_report_path)
    validate_evaluation_report(root, evaluation)
    compatibility = read_json(compatibility_path)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    time_iso = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    log_path = PATHS.log / f"{timestamp}_experiment6_corrected12_completed.json"
    output_relative = root.relative_to(PATHS.workspace).as_posix()
    log_relative = log_path.relative_to(PATHS.workspace).as_posix()
    payload = {
        "time": time_iso,
        "repo": str(PATHS.repo),
        "kind": "experiment6_narrative2_corrected12_completed",
        "status": "completed_12_cases_120_runs_10200_predictions_reference_aligned_v5_1",
        "summary": (
            "Twelve base cases invalidated by source_id-first routing were freshly rerun "
            "with formal direct-binding routes. Declared/effective routes match; base "
            "models used no adapter, retriever converter, or reused generation-result "
            "cache. Reference-"
            "aligned v5.1 diagnostic evaluation used hard DataName/Position anchors "
            "and ChatMock "
            "gpt-5.5 medium semantic judging."
        ),
        "output_root": output_relative,
        "compatibility_fingerprint": compatibility["sha256"],
        "root_cause": (
            "source_id family mapping was evaluated before case.route, changing twelve "
            "declared direct-binding base cases into retriever-converter executions"
        ),
        "matrix": {
            "cases": 12,
            "runsPerCase": 10,
            "rowsPerRun": 85,
            "predictions": 10200,
            "minimumRerunCases": MINIMUM_CASES,
            "fullFormalRankingCases": full_38_cases(),
        },
        "validation": {
            "complete": run_status["complete"],
            "statusCounts": run_status["statusCounts"],
            "routeMismatches": run_status["routeMismatches"],
            "fingerprints": run_status["compatibilityFingerprints"],
            "modelSubstitution": False,
            "goldLeakage": False,
        },
        "evaluation": {
            "protocol": evaluation.get("protocol"),
            "status": evaluation.get("status"),
            "judge": evaluation.get("judge"),
            "method": evaluation.get("method"),
            "ordering": evaluation.get("ordering"),
        },
        "hashes": {
            "generationConfig": sha256_file(
                PATHS.resolve("repo", "config/experiment6_narrative2_generation_corrected_12.json")
            ),
            "evaluationConfig": sha256_file(
                PATHS.resolve("repo", "config/experiment6_narrative2_evaluation_corrected_12.json")
            ),
            "sourceRegistry": sha256_file(
                PATHS.resolve("repo", "config/experiment6_source_registry.json")
            ),
            "evaluationReport": sha256_file(evaluation_report_path),
            "inventory": sha256_file(inventory_path),
        },
    }
    write_json_atomic(log_path, payload)

    index_path = PATHS.log / "index.json"
    index = read_json(index_path)
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("indexed docs/log/index.json lacks entries[]")
    entry = {
        "time": time_iso,
        "repo": str(PATHS.repo),
        "kind": payload["kind"],
        "status": payload["status"],
        "summary": payload["summary"],
        "path": log_relative,
        "output_root": output_relative,
        "tags": [
            "experiment_6",
            "narrative2",
            "corrected12",
            "route_first",
            "direct_binding",
            "reference_aligned_v5_1",
            "gpt-5.5-medium",
            "completed",
        ],
    }
    if not any(item.get("output_root") == output_relative for item in entries):
        entries.append(entry)
        write_json_atomic(index_path, index)
    print(log_path)


if __name__ == "__main__":
    main()
