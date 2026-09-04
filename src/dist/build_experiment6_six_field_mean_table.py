#!/usr/bin/env python3
"""Render six-field mean Precision/Recall/F1 from a v5.1 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = ("ObjectName", "Trend", "Num", "Text", "Position", "DataName")
STAGE = "semantic_gpt55_medium"


class TableError(RuntimeError):
    pass


def value(metric: dict[str, Any], name: str) -> float:
    return float(metric[name]["mean"])


def render(report: dict[str, Any]) -> str:
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise TableError("evaluation report has no cases")
    lines = [
        "# Experiment 6 v6.1 六欄分數（10-run mean）",
        "",
        "評估階段：`semantic_gpt55_medium`；不計算 TN。",
        "",
        "| Case | Annotation | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for case in sorted(cases, key=lambda item: str(item["outputId"])):
        stage = case["ablations"][STAGE]
        for field in FIELDS:
            metrics = stage["fields"][field]
            lines.append(
                f"| {case['outputId']} | {field} | "
                f"{value(metrics, 'precision'):.6f} | "
                f"{value(metrics, 'recall'):.6f} | "
                f"{value(metrics, 'f1'):.6f} |"
            )
        macro = stage["macro"]
        lines.append(
            f"| {case['outputId']} | Macro (6 fields) | "
            f"{value(macro, 'precision'):.6f} | "
            f"{value(macro, 'recall'):.6f} | "
            f"{value(macro, 'f1'):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Judge：`{report['judge']['model']}` / "
            f"`{report['judge']['reasoningEffort']}`；"
            f"confidence = `{report['judge']['minimumConfidence']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
