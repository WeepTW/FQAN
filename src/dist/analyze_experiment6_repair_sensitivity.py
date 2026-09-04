#!/usr/bin/env python3
"""Evaluate pre-existing deterministic repairs as diagnostic-only sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import evaluate_narrative2_reference_aligned_v6_0_2 as v602


PROTOCOL = "experiment6-deterministic-repair-sensitivity-v1"
REQUIRED_BINDING_KEYS = {
    "ObjectName",
    "DataName",
    "Position",
    "Trend",
    "Num",
    "Text",
}


class SensitivityError(RuntimeError):
    """Raised when diagnostic inputs violate their recorded contract."""


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
        raise SensitivityError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SensitivityError(f"{path}:{line_number}: row is not an object")
        values.append(value)
    return values


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def validate_binding(binding: Any) -> tuple[bool, str]:
    if not isinstance(binding, dict):
        return False, "binding_not_object"
    if set(binding) != REQUIRED_BINDING_KEYS:
        return False, f"binding_keys={sorted(binding)}"
    if (
        not isinstance(binding["ObjectName"], list)
        or not binding["ObjectName"]
        or not all(
            isinstance(value, str) and value.strip()
            for value in binding["ObjectName"]
        )
    ):
        return False, "ObjectName_not_nonempty_string_array"
    if not isinstance(binding["DataName"], str):
        return False, "DataName_not_string"
    if not isinstance(binding["Trend"], str):
        return False, "Trend_not_string"
    if not isinstance(binding["Text"], str):
        return False, "Text_not_string"
    if not isinstance(binding["Num"], list) or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in binding["Num"]
    ):
        return False, "Num_not_finite_number_array"
    if not isinstance(binding["Position"], list):
        return False, "Position_not_array"
    for position in binding["Position"]:
        if not isinstance(position, dict) or set(position) != {"Begin", "End"}:
            return False, "Position_item_invalid_keys"
        for key in ("Begin", "End"):
            coordinate = position[key]
            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in coordinate
                )
            ):
                return False, f"Position_{key}_not_two_integer_array"
    return True, "valid"


def strict_payload_result(payload: Any) -> tuple[list[dict[str, Any]] | None, str]:
    result: Any
    if isinstance(payload, dict) and set(payload) == {"result", "reason"}:
        if not isinstance(payload["reason"], str):
            return None, "reason_not_string"
        result = payload["result"]
    elif isinstance(payload, dict) and set(payload) == {"Binding"}:
        result = payload["Binding"]
    elif isinstance(payload, list):
        result = payload
    else:
        return None, "top_level_contract"
    if not isinstance(result, list):
        return None, "result_not_array"
    for binding in result:
        valid, reason = validate_binding(binding)
        if not valid:
            return None, reason
    return result, "valid"


def derive_predictions(
    predictions: Sequence[Mapping[str, Any]],
    repairs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repair_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for repair in repairs:
        if repair.get("official") is not False or repair.get("excludedFromScores") is not True:
            raise SensitivityError("repair row is not explicitly excluded from official scores")
        key = (int(repair["index"]), str(repair["source"]))
        if key in repair_by_key:
            raise SensitivityError(f"duplicate repair row: {key}")
        repair_by_key[key] = repair

    derived: list[dict[str, Any]] = []
    counts = Counter()
    rejection_reasons = Counter()
    for prediction in predictions:
        item = dict(prediction)
        if prediction.get("formatValid") and isinstance(prediction.get("result"), list):
            counts["officialValidRows"] += 1
            derived.append(item)
            continue
        key = (int(prediction["index"]), str(prediction["source"]))
        repair_row = repair_by_key.get(key)
        if repair_row is None:
            counts["repairMissingRows"] += 1
            derived.append(item)
            continue
        repair = repair_row.get("repair") or {}
        if not repair.get("available"):
            counts["repairUnavailableRows"] += 1
            derived.append(item)
            continue
        counts["repairAvailableRows"] += 1
        result, reason = strict_payload_result(repair.get("payload"))
        if result is None:
            counts["repairSchemaRejectedRows"] += 1
            rejection_reasons[reason] += 1
            derived.append(item)
            continue
        counts["repairSchemaValidRows"] += 1
        item["result"] = result
        item["formatValid"] = True
        item["parserDiagnostic"] = {
            "strict": False,
            "valid": True,
            "diagnosticOnly": True,
            "method": str(repair.get("method") or "recorded-nonformal-repair"),
            "sourceOfficial": False,
            "claimEligible": False,
        }
        derived.append(item)
    counts["rows"] = len(predictions)
    return derived, {
        **dict(sorted(counts.items())),
        "schemaRejectionReasons": dict(sorted(rejection_reasons.items())),
    }


def resolve_repair_path(
    generation_root: Path,
    manifest: Mapping[str, Any],
) -> Path:
    raw = str((manifest.get("files") or {}).get("nonformalRepair") or "")
    declared = Path(raw)
    if declared.is_file():
        return declared
    return (
        generation_root
        / "cases"
        / str(manifest["outputId"])
        / f"run_{int(manifest['run']):02d}"
        / declared.name
    )


def overall_case_mean(cases: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        case["aggregate"]["macro"]["f1"]["mean"]
        for case in cases
        if case["aggregate"]["macro"]["f1"]["mean"] is not None
    ]
    return statistics.fmean(float(value) for value in values) if values else None


def build(args: argparse.Namespace) -> dict[str, Any]:
    generation_root = args.generation_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    strict_path = args.strict_evaluation_report.resolve()
    config = read_json(config_path)
    v602.validate_config(config)
    strict_report = read_json(strict_path)
    if strict_report.get("protocol") != v602.PROTOCOL:
        raise SensitivityError("strict report must use v6.0.2")
    if strict_report.get("mode") != "formal":
        raise SensitivityError("strict report must be formal")

    gold_path = (v602.REPO_ROOT / config["goldPath"]).resolve()
    if sha256_file(gold_path) != config["goldSha256"]:
        raise SensitivityError("gold SHA mismatch")
    targets = read_json(gold_path).get("rows")
    if not isinstance(targets, list) or len(targets) != int(config["expectedRows"]):
        raise SensitivityError("gold row count mismatch")

    manifest_paths = sorted((generation_root / "manifests").glob("*.json"))
    manifests = [
        read_json(path)
        for path in manifest_paths
        if read_json(path).get("official")
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        grouped[str(manifest["outputId"])].append(manifest)
    expected_runs = list(range(1, int(config["expectedRuns"]) + 1))
    incomplete = {
        output_id: sorted(int(item["run"]) for item in items)
        for output_id, items in grouped.items()
        if sorted(int(item["run"]) for item in items) != expected_runs
    }
    if incomplete:
        raise SensitivityError(f"incomplete runs: {incomplete}")

    objects = v602.ObjectMatcher(config["objectName"])
    trends = v602.TrendClassifier(config["trend"], allow_model=True)
    run_results: list[dict[str, Any]] = []
    totals = Counter()
    rejection_reasons = Counter()
    input_artifacts: list[dict[str, Any]] = []
    for manifest in sorted(
        manifests, key=lambda item: (str(item["outputId"]), int(item["run"]))
    ):
        predictions, prediction_path = v602.load_predictions(
            manifest, targets, generation_root
        )
        repair_path = resolve_repair_path(generation_root, manifest)
        if not repair_path.is_file():
            raise SensitivityError(f"repair file missing: {repair_path}")
        expected_repair_hash = str(
            (manifest.get("hashes") or {}).get("nonformalRepair") or ""
        )
        if sha256_file(repair_path) != expected_repair_hash:
            raise SensitivityError(
                f"repair SHA mismatch: {manifest['outputId']} run {manifest['run']}"
            )
        repairs = read_jsonl(repair_path)
        derived, repair_summary = derive_predictions(predictions, repairs)
        for key, value in repair_summary.items():
            if key == "schemaRejectionReasons":
                rejection_reasons.update(value)
            else:
                totals[key] += int(value)
        summary, records = v602.evaluate_rows(targets, derived, objects, trends)
        output_id = str(manifest["outputId"])
        run = int(manifest["run"])
        run_result = {
            "outputId": output_id,
            "run": run,
            "seed": manifest.get("seed"),
            "primary": summary["primary"],
            "bindingLevel": summary["bindingLevel"],
            "withoutTrendAblation": summary["withoutTrendAblation"],
            "coverage": summary["coverage"],
            "trend": summary["trend"],
            "repair": repair_summary,
        }
        run_results.append(run_result)
        write_jsonl(
            output_root / "cases" / output_id / f"run_{run:02d}" / "records.jsonl",
            records,
        )
        write_json(
            output_root / "cases" / output_id / f"run_{run:02d}" / "metrics.json",
            run_result,
        )
        input_artifacts.extend(
            [
                {
                    "kind": "predictions",
                    "path": str(prediction_path),
                    "sha256": sha256_file(prediction_path),
                },
                {
                    "kind": "recorded-nonformal-repair",
                    "path": str(repair_path),
                    "sha256": sha256_file(repair_path),
                },
            ]
        )

    cases = [
        v602.aggregate_case(
            output_id,
            [item for item in run_results if item["outputId"] == output_id],
            "diagnostic",
            config,
        )
        for output_id in sorted(grouped)
    ]
    strict_cases = {
        str(case["outputId"]): case for case in strict_report.get("cases", [])
    }
    deltas = []
    for case in cases:
        output_id = str(case["outputId"])
        strict_case = strict_cases.get(output_id)
        if strict_case is None:
            raise SensitivityError(f"strict case missing: {output_id}")
        repaired_score = case["aggregate"]["macro"]["f1"]["mean"]
        strict_score = strict_case["aggregate"]["macro"]["f1"]["mean"]
        deltas.append(
            {
                "outputId": output_id,
                "strictMacroF1Mean": strict_score,
                "repairMacroF1Mean": repaired_score,
                "difference": (
                    float(repaired_score) - float(strict_score)
                    if repaired_score is not None and strict_score is not None
                    else None
                ),
            }
        )

    report = {
        "schemaVersion": 1,
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "claimEligible": False,
        "selectionEmitted": False,
        "time": utc_now(),
        "generationRoot": str(generation_root),
        "outputRoot": str(output_root),
        "configPath": str(config_path),
        "configSha256": sha256_file(config_path),
        "evaluatorProtocol": v602.PROTOCOL,
        "evaluatorSha256": sha256_file(Path(v602.__file__).resolve()),
        "strictEvaluationReport": str(strict_path),
        "strictEvaluationSha256": sha256_file(strict_path),
        "goldPath": str(gold_path),
        "goldSha256": sha256_file(gold_path),
        "completedCases": len(cases),
        "completedCaseRuns": len(run_results),
        "repair": {
            **dict(sorted(totals.items())),
            "schemaRejectionReasons": dict(sorted(rejection_reasons.items())),
        },
        "overall": {
            "strictCorrected12MacroF1Mean": overall_case_mean(
                list(strict_cases.values())
            ),
            "repairSensitivityMacroF1Mean": overall_case_mean(cases),
        },
        "caseDifferences": deltas,
        "cases": cases,
        "inputArtifacts": input_artifacts,
        "limitations": [
            "Existing test predictions were inspected; this result is exploratory only.",
            "Recorded nonformal repairs were created during generation and remain excluded from official scores.",
            "No alias, Sentence-BERT threshold, gold label, or LLM judge was used to repair outputs.",
            "No top-1 or top-3 selection is emitted.",
        ],
    }
    write_json(output_root / "repair_sensitivity_report.json", report)
    markdown = [
        "# Experiment 6 deterministic-repair sensitivity",
        "",
        "- Status: diagnostic only; claimEligible=false.",
        f"- Cases/runs: {len(cases)}/{len(run_results)}.",
        f"- Rows: {totals.get('rows', 0)}.",
        f"- Recorded repair available: {totals.get('repairAvailableRows', 0)}.",
        f"- Full-schema-valid repair: {totals.get('repairSchemaValidRows', 0)}.",
        f"- Strict corrected12 macro-F1 mean: {report['overall']['strictCorrected12MacroF1Mean']}.",
        f"- Repair sensitivity macro-F1 mean: {report['overall']['repairSensitivityMacroF1Mean']}.",
        "",
        "| case | strict macro-F1 mean | repair macro-F1 mean | difference |",
        "|---|---:|---:|---:|",
    ]
    for item in deltas:
        markdown.append(
            f"| {item['outputId']} | {item['strictMacroF1Mean']} | "
            f"{item['repairMacroF1Mean']} | {item['difference']} |"
        )
    markdown.extend(
        [
            "",
            "This exploratory analysis does not replace either v6.0.1 or v6.0.2.",
        ]
    )
    (output_root / "repair_sensitivity_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strict-evaluation-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build(args)
    except (
        SensitivityError,
        v602.ProtocolError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "claimEligible": report["claimEligible"],
                "completedCases": report["completedCases"],
                "completedCaseRuns": report["completedCaseRuns"],
                "repair": report["repair"],
                "overall": report["overall"],
                "outputRoot": report["outputRoot"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
