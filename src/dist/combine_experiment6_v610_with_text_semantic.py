#!/usr/bin/env python3
"""Combine v6.1 five-field metrics with v5.1 semantic Text metrics.

The resulting six-field report keeps each method's provenance explicit.  No
counts or judgments are recomputed: five fields come from the v6.1 report and
Text comes from the GPT-5.5-medium semantic stage of the v5.1 report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


FIVE_FIELDS = ("ObjectName", "Trend", "Num", "Position", "DataName")
ALL_FIELDS = ("ObjectName", "Trend", "Num", "Text", "Position", "DataName")
METRICS = ("precision", "recall", "f1")
PROTOCOL = "experiment6-reference-aligned-v6.1-with-semantic-text-v1"


class CombineError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CombineError(f"expected JSON object: {path}")
    return value


def metric_mean(value: Mapping[str, Any]) -> float | None:
    raw = value.get("mean")
    return None if raw is None else float(raw)


def average(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def display(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def build(v610_path: Path, semantic_path: Path) -> dict[str, Any]:
    v610 = read_json(v610_path)
    semantic = read_json(semantic_path)
    direct_v610 = v610.get("protocol") == "experiment6-reference-aligned-v6.1.0"
    wrapped_v610 = (
        v610.get("protocol") == "experiment6-binding-candidate-evaluation-v1"
        and v610.get("scoringProtocol") == "experiment6-reference-aligned-v6.1.0"
        and v610.get("scope") == "mistral-base-md"
        and v610.get("diagnosticOnly") is True
        and v610.get("claimEligible") is False
    )
    if not (direct_v610 or wrapped_v610):
        raise CombineError("unexpected v6.1 protocol")
    if semantic.get("protocol") != "narrative2-reference-aligned-hybrid-v5.1":
        raise CombineError("unexpected semantic Text protocol")
    judge = semantic.get("judge") or {}
    if (
        judge.get("model") != "gpt-5.5"
        or judge.get("reasoningEffort") != "medium"
        or judge.get("disabled") is not False
    ):
        raise CombineError("semantic Text judge identity mismatch")
    v610_cases = {str(item["outputId"]): item for item in v610.get("cases") or []}
    semantic_cases = {str(item["outputId"]): item for item in semantic.get("cases") or []}
    if set(v610_cases) != set(semantic_cases) or set(v610_cases) != {
        "6_mistral_base_m",
        "6_mistral_base_d",
    }:
        raise CombineError("case sets do not match the two-case contract")
    cases: list[dict[str, Any]] = []
    for output_id in sorted(v610_cases):
        five = v610_cases[output_id]["aggregate"]["fields"]
        text = semantic_cases[output_id]["ablations"]["semantic_gpt55_medium"]["fields"]["Text"]
        fields: dict[str, Any] = {}
        for field in FIVE_FIELDS:
            fields[field] = {
                metric: metric_mean(five[field][metric]) for metric in METRICS
            }
        fields["Text"] = {
            metric: metric_mean(text[metric]) for metric in METRICS
        }
        macro = {
            metric: average([fields[field][metric] for field in ALL_FIELDS])
            for metric in METRICS
        }
        cases.append(
            {
                "outputId": output_id,
                "runs": int(v610_cases[output_id]["runs"]),
                "fields": fields,
                "macro": macro,
                "strictSchemaValidity": semantic_cases[output_id].get("strictSchemaValidity"),
                "coverage": semantic_cases[output_id].get("coverage"),
                "rootCauses": semantic_cases[output_id].get("rootCauses"),
            }
        )
    return {
        "protocol": PROTOCOL,
        "status": "completed",
        "diagnosticOnly": True,
        "claimEligible": False,
        "counting": "TP/FP/FN only; TN undefined and not computed",
        "method": {
            "ObjectNameTrendNumPositionDataName": "reference-aligned v6.1.0",
            "Text": "v5.1 normalized exact then GPT-5.5 medium semantic judge",
            "TextMinimumConfidence": judge.get("minimumConfidence"),
            "macro": "arithmetic mean of the six field-level 10-run means; NA fields excluded",
        },
        "inputs": {
            "v610Report": str(v610_path.resolve()),
            "v610ReportSha256": sha256_file(v610_path),
            "semanticTextReport": str(semantic_path.resolve()),
            "semanticTextReportSha256": sha256_file(semantic_path),
        },
        "judge": judge,
        "cases": cases,
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 6 Mistral base_m/base_d 六欄診斷評估",
        "",
        "ObjectName、Trend、Num、Position、DataName 使用 reference-aligned v6.1；"
        "Text 先作正規化精確比對，未決項由 GPT-5.5 medium 語意判定。TN 未定義，故不計算。",
        "",
        "| Case | Annotation | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for case in report["cases"]:
        for field in ALL_FIELDS:
            metrics = case["fields"][field]
            lines.append(
                f"| {case['outputId']} | {field} | {display(metrics['precision'])} | "
                f"{display(metrics['recall'])} | {display(metrics['f1'])} |"
            )
        macro = case["macro"]
        lines.append(
            f"| {case['outputId']} | Macro (6 fields) | {display(macro['precision'])} | "
            f"{display(macro['recall'])} | {display(macro['f1'])} |"
        )
    lines.extend(
        [
            "",
            f"Judge：`{report['judge']['model']}` / `{report['judge']['reasoningEffort']}`；"
            f"confidence = `{report['judge']['minimumConfidence']}`。",
            "",
            "此為獨立診斷，不更新正式 38-case 排名。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v610-report", type=Path, required=True)
    parser.add_argument("--semantic-text-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.v610_report.resolve(), args.semantic_text_report.resolve())
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rendered = markdown(report)
    (root / "evaluation_report.md").write_text(rendered, encoding="utf-8")
    (root / "experiment_6_v6_欄位分數_mean.md").write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": "completed", "outputRoot": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
