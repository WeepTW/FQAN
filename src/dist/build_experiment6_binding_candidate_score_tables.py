#!/usr/bin/env python3
"""Build claim-ineligible score tables for Experiment 6 Binding candidates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import build_experiment6_v6_score_tables as formal_tables
from evaluate_experiment6_binding_candidates_v1 import PROTOCOL, logical_path, sha256_file


WORKSPACE_ROOT = DIST.parents[1]
VIEW_LABELS = formal_tables.VIEW_LABELS


class CandidateTableError(RuntimeError):
    """Raised when a candidate report cannot support diagnostic tables."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateTableError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateTableError(message)


def display_metadata(
    evaluation: Mapping[str, Any], registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sources = registry.get("sources") or {}
    case_by_id = {str(case["outputId"]): case for case in evaluation.get("cases", [])}
    rows = []
    for metadata in evaluation.get("caseMetadata", []):
        output_id = str(metadata["outputId"])
        source_id = str(metadata["sourceId"])
        require(source_id in sources, f"source registry entry missing: {source_id}")
        source = sources[source_id]
        kind = str(source.get("kind"))
        fine_tuned_method = str(source.get("promptMode")) if kind == "adapter" else "no-adaptor"
        rows.append({
            **metadata,
            "fineTunedMethod": fine_tuned_method,
            "retrieverModel": str(source.get("baseModel") or source_id),
            "inputPromptType": str(metadata.get("inputType") or metadata["promptMode"]),
            "case": case_by_id[output_id],
        })
    require(len(rows) == len(case_by_id), "case metadata/evaluation mismatch")
    rows.sort(key=formal_tables.row_sort_key)
    keys = [
        (row["fineTunedMethod"], row["retrieverModel"], row["inputPromptType"])
        for row in rows
    ]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    require(not duplicates, f"duplicate report metadata keys: {duplicates}")
    return rows


def validate_evaluation(evaluation: Mapping[str, Any]) -> None:
    require(evaluation.get("protocol") == PROTOCOL, "candidate report protocol mismatch")
    require(evaluation.get("status") == "diagnostic_only", "candidate report status mismatch")
    require(evaluation.get("mode") == "diagnostic", "candidate report mode mismatch")
    require(evaluation.get("official") is False, "candidate report cannot be official")
    require(evaluation.get("claimEligible") is False, "candidate report cannot be claim-eligible")
    require(evaluation.get("selectionEmitted") is True, "descriptive selection missing")
    require(evaluation.get("selectionRole") == "diagnostic-descriptive-only", "selection role mismatch")
    require(evaluation.get("scopeComplete") is True, "scope is incomplete")
    require(evaluation.get("experimentMatrixComplete") is False, "38-case matrix must remain incomplete")
    expected_runs = 10
    cases = evaluation.get("cases")
    require(isinstance(cases, list), "evaluation cases missing")
    for case in cases:
        require(int(case.get("runs", 0)) == expected_runs, f"{case.get('outputId')} lacks ten runs")
        selection = case.get("selection")
        require(isinstance(selection, Mapping), f"{case.get('outputId')} selection missing")
        require(selection.get("role") == "diagnostic-descriptive-only", f"{case.get('outputId')} selection role mismatch")
        require(len(selection.get("sharedRunOrder") or []) == expected_runs, f"{case.get('outputId')} shared ranking incomplete")
        require(len(selection.get("top3", {}).get("runs") or []) == 3, f"{case.get('outputId')} top-3 incomplete")


def metric_score_for(
    case: Mapping[str, Any], view: str, field: str, metric_name: str
) -> Any:
    if view == "mean":
        return case["aggregate"]["fields"][field][metric_name]["mean"]
    if view == "top-1":
        return case["selection"]["top1"]["fields"][field][metric_name]
    if view == "top-3":
        return case["selection"]["top3"]["fields"][field][metric_name]
    raise CandidateTableError(f"unknown score view: {view}")


def metric_table_lines(
    rows: Sequence[Mapping[str, Any]], view: str, metric_name: str
) -> list[str]:
    output = [
        "| " + " | ".join(formal_tables.REPORT_COLUMNS) + " |",
        "|"
        + "|".join(
            "---" if index < 3 else "---:"
            for index in range(len(formal_tables.REPORT_COLUMNS))
        )
        + "|",
    ]
    for row in rows:
        values = [
            row["fineTunedMethod"],
            row["retrieverModel"],
            row["inputPromptType"],
        ]
        values.extend(
            formal_tables.format_score(
                metric_score_for(row["case"], view, field, metric_name)
            )
            for _, field in formal_tables.FIELD_MAP
        )
        values.append("NA (judge deferred)")
        output.append(
            "| "
            + " | ".join(formal_tables.markdown_escape(value) for value in values)
            + " |"
        )
    return output


def scoring_example(version: str) -> list[str]:
    trend_note = (
        "v6.1 將 increase／decrease／stable／reversal／peak／trough 與五種圖形模式分開；"
        if version == "v6.1.0"
        else "Trend 依該版凍結類別規則判定；"
    )
    return [
        "## TP／FP／FN 與指標定義",
        "",
        "先以 DataName＋Position 作一對一 hard-anchor 對齊，再逐欄累計 TP、FP、FN。"
        "答錯且有輸出時該欄同時計 FP＋FN；缺欄只計 FN；未配對 prediction 計 FP；"
        "未配對 gold 計 FN。",
        "",
        "Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，"
        "F1 = 2TP/(2TP+FP+FN)。表內三項皆由同一組 TP／FP／FN 推導，未另行估算。",
        "",
        trend_note
        + "DataName、Position 始終硬比對；Text judge deferred，故為 NA。",
        "",
    ]


def selection_lines(rows: Sequence[Mapping[str, Any]], view: str) -> list[str]:
    if view == "mean":
        return []
    lines = [
        "",
        "## 共同 run 選擇稽核（描述性、不可主張）",
        "",
        "| outputId | selected run(s) |",
        "|---|---|",
    ]
    for row in rows:
        selection = row["case"]["selection"]
        selected = (
            str(selection["top1"]["run"])
            if view == "top-1"
            else ", ".join(str(value) for value in selection["top3"]["runs"])
        )
        lines.append(f"| {row['outputId']} | {selected} |")
    return lines


def render(
    view: str,
    rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> str:
    version = str(evaluation["scoringProtocol"]).rsplit("-", 1)[-1]
    scope = str(evaluation["scope"])
    lines = [
        f"# Experiment 6 Binding candidates {version} × {scope}：{VIEW_LABELS[view]}",
        "",
        "> **DIAGNOSTIC ONLY — OFFICIAL=false — CLAIM-ELIGIBLE=false**",
        "",
        f"- 評估時間：{evaluation.get('time')}。",
        f"- 範圍：{evaluation.get('completedCases')} cases × 10 runs；experimentMatrixComplete=false。",
        f"- 評估 protocol：{evaluation.get('protocol')}；base scoring protocol：{evaluation.get('scoringProtocol')}。",
        f"- Composite method SHA-256：`{evaluation['method']['methodSha256']}`。",
        "- Subject／Trend／Num／Position／DataName 皆為 DataName＋Position hard-anchor 對齊後的 anchored end-to-end field Precision／Recall／F1；既有 anchored end-to-end field F1 標籤保留（0–1）。",
        "- DataName 與 Position 同時也是 binding identity anchor；其欄位分數不是獨立於對齊條件的 conditional accuracy。",
        "- Text judge 未執行，故為 NA；不進五欄 TP／FP／FN。",
        "- top-1／top-3 使用每 case 的共同 run 排序，僅供描述，不得作論文方法優劣主張。",
        "",
    ]
    lines.extend(scoring_example(version))
    for metric_name, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ):
        lines.extend([
            f"## {label}：{VIEW_LABELS[view]}",
            "",
            *metric_table_lines(rows, view, metric_name),
            "",
        ])
    lines.extend(selection_lines(rows, view))
    lines.extend([
        "",
        "## 可解讀範圍",
        "",
        "- 新 bindings 來自生成時已保存的非正式 deterministic repair；未用 gold、別名擴增或 LLM 改寫資料。",
        (
            "- 本表只含單一 FLAN no-adaptor 長上下文診斷案例；不可外推至正式矩陣。"
            if scope == "flan-long-context"
            else "- candidate-merged34 中，同名 12 cases 來自 candidate 資料，其餘 22 cases 沿用凍結的歷史正式 predictions；整體仍為診斷結果。"
        ),
        (
            "- 本診斷刻意不納入其他 37 cases；不得宣稱正式矩陣完成。"
            if scope == "flan-long-context"
            else "- 四個 GPT-4.1 cases 未完成且未納入；不得宣稱 38-case matrix 完成。"
        ),
        "- 現有 gold 對五種特殊 Trend 圖形的 support 均為 0，Trend 分數不證成圖形辨識能力。",
    ])
    return "\n".join(lines) + "\n"


def grouped_case_means(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row["case"]["aggregate"]["macro"]["f1"]["mean"]
        if value is not None and math.isfinite(float(value)):
            grouped[str(row[key])].append(float(value))
    return {
        name: {"n": len(values), "macroF1Mean": statistics.fmean(values)}
        for name, values in sorted(grouped.items())
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_path = args.evaluation_report.resolve()
    evaluation_root = args.evaluation_root.resolve()
    evaluation = read_json(evaluation_path)
    registry = read_json(args.source_registry.resolve())
    validate_evaluation(evaluation)
    rows = display_metadata(evaluation, registry)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for view in ("mean", "top-1", "top-3"):
        path = output_dir / f"experiment_6_v6_欄位分數_{view}.md"
        path.write_text(render(view, rows, evaluation), encoding="utf-8")
        outputs[view] = logical_path(path)

    scoring_configs = {
        "experiment6-reference-aligned-v6.0.1": "config/experiment6_narrative2_evaluation_v6.json",
        "experiment6-reference-aligned-v6.0.2": "config/experiment6_narrative2_evaluation_v6_0_2.json",
        "experiment6-reference-aligned-v6.1.0": "config/experiment6_narrative2_evaluation_v6_1.json",
    }
    scoring_protocol = str(evaluation["scoringProtocol"])
    require(scoring_protocol in scoring_configs, f"unknown scoring protocol: {scoring_protocol}")
    scoring_config_path = (DIST.parent / scoring_configs[scoring_protocol]).resolve()
    scoring_config = read_json(scoring_config_path)
    gold_path = (DIST.parent / scoring_config["goldPath"]).resolve()
    relationship = formal_tables.relationship_summary(rows, evaluation_root, gold_path)
    relationship.update({
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "claimEligible": False,
        "scope": evaluation["scope"],
        "scoringProtocol": evaluation["scoringProtocol"],
        "overall": evaluation["overall"],
        "coverage": evaluation["coverage"],
        "candidateValidation": evaluation["candidateValidation"],
        "byInputOrigin": grouped_case_means(rows, "origin"),
        "sourceEvaluationReport": logical_path(evaluation_path),
        "sourceEvaluationReportSha256": sha256_file(evaluation_path),
    })
    relationship_path = output_dir / "experiment_6_v6_relationship_summary.json"
    relationship_path.write_text(
        json.dumps(relationship, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "claimEligible": False,
        "scope": evaluation["scope"],
        "scoringProtocol": evaluation["scoringProtocol"],
        "outputs": outputs,
        "relationshipSummaryPath": logical_path(relationship_path),
        "relationshipSummarySha256": sha256_file(relationship_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build(args)
    except (CandidateTableError, formal_tables.TableError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
