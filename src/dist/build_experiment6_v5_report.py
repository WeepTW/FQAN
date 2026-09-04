#!/usr/bin/env python3
"""Build the canonical portable-artifact input for Experiment 6 v5.1."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TITLE = "Experiment 6：七組零分重跑與 reference-aligned hybrid-v5.1 診斷"
STAGES = (
    ("frozen_hybrid_v4", "Frozen hybrid-v4"),
    ("hard_anchor_local_exact", "硬 anchor＋局部處罰"),
    ("deterministic_normalized", "＋確定式正規化"),
    ("semantic_gpt55_medium", "＋GPT-5.5 語意判定"),
)
FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def mean_at(value: dict[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def case_rows(report: dict[str, Any], route_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        primary = case["primary"]
        pooled = case["ablations"]["semantic_gpt55_medium"]["pooled"]["micro"]
        conditional = case["conditionalContent"]["f1"]
        rows.append({
            "case": case["outputId"],
            "route": route_label,
            "schema_validity": case["strictSchemaValidity"]["mean"],
            "predicted_bindings": case["coverage"]["predictedBindings"],
            "matched_bindings": case["coverage"]["matchedBindings"],
            "anchor_precision": case["coverage"]["anchorPrecision"],
            "anchor_recall": case["coverage"]["anchorRecall"],
            "macro_precision": primary["macro"]["precision"]["mean"],
            "macro_recall": primary["macro"]["recall"]["mean"],
            "macro_f1": primary["macro"]["f1"]["mean"],
            "pooled_micro_precision": pooled["precision"],
            "pooled_micro_recall": pooled["recall"],
            "pooled_micro_f1": pooled["f1"],
            "conditional_f1": conditional["mean"],
            "conditional_eligible_runs": conditional["eligibleRuns"],
        })
    return rows


def ablation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        if case["outputId"] not in {"6_mistral_base_d", "6_mistral_base_m"}:
            continue
        for stage, label in STAGES:
            value = case["ablations"][stage]["macro"]["f1"]["mean"]
            rows.append({"case": case["outputId"], "stage": label, "macro_f1": value})
    return rows


def field_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        if case["outputId"] not in {"6_mistral_base_d", "6_mistral_base_m"}:
            continue
        pooled = case["ablations"]["semantic_gpt55_medium"]["pooled"]["fields"]
        for field in FIELDS:
            rows.append({"case": case["outputId"], "field": field, **pooled[field]})
    return rows


def root_cause_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"case": case["outputId"], **case["rootCauses"]}
        for case in report["cases"]
    ]


def low_precision_rows(
    old_scorecard: Path, low_report: dict[str, Any]
) -> list[dict[str, Any]]:
    old = {row["output_id"]: row for row in read_tsv(old_scorecard)}
    rows: list[dict[str, Any]] = []
    for case in low_report["cases"]:
        output_id = case["outputId"]
        before = old[output_id]
        primary = case["primary"]["macro"]
        rows.append({
            "case": output_id,
            "v4_precision": float(before["overall_mean_precision"]),
            "v4_recall": float(before["overall_mean_recall"]),
            "v4_f1": float(before["overall_mean_f1"]),
            "v5_precision": primary["precision"]["mean"],
            "v5_recall": primary["recall"]["mean"],
            "v5_f1": primary["f1"]["mean"],
            "predicted_bindings": case["coverage"]["predictedBindings"],
            "matched_bindings": case["coverage"]["matchedBindings"],
            "anchor_recall": case["coverage"]["anchorRecall"],
        })
    return sorted(rows, key=lambda row: row["v4_precision"])


def copy_evidence(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.as_posix()


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def write_values_sql(path: Path, table_name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot build SQL source for empty dataset {table_name}")
    columns = list(rows[0])
    values = ",\n  ".join(
        "(" + ", ".join(sql_literal(row.get(column)) for column in columns) + ")"
        for row in rows
    )
    query = (
        f"SELECT * FROM (VALUES\n  {values}\n) "
        f"AS {table_name} ({', '.join(columns)});\n"
    )
    path.write_text(query, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-evaluation", type=Path, required=True)
    parser.add_argument("--direct-evaluation", type=Path, required=True)
    parser.add_argument("--low18-evaluation", type=Path, required=True)
    parser.add_argument("--old-v4-scorecard", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--evaluation-advice", type=Path, required=True)
    parser.add_argument("--evaluator-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    evidence_dir = output_dir / "evidence"
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_path = args.historical_evaluation / "evaluation_report.json"
    direct_path = args.direct_evaluation / "evaluation_report.json"
    low18_path = args.low18_evaluation / "evaluation_report.json"
    historical = read_json(historical_path)
    direct = read_json(direct_path)
    low18 = read_json(low18_path)
    integrity = read_json(args.integrity_audit)
    if integrity.get("status") != "passed" or integrity.get("actualFreshPredictions") != 11050:
        raise ValueError("integrity audit must pass with exactly 11,050 fresh predictions")
    if low18.get("judge", {}).get("model") != "gpt-5.5" or low18.get("judge", {}).get("reasoningEffort") != "medium":
        raise ValueError("low-precision evaluation did not use gpt-5.5 / medium")

    historical_rows = case_rows(historical, "historical retriever→converter")
    direct_rows = case_rows(direct, "direct diagnostic native")
    ablations = ablation_rows(historical)
    fields = field_rows(historical)
    causes = root_cause_rows(historical)
    low_rows = low_precision_rows(args.old_v4_scorecard, low18)
    low_causes = root_cause_rows(low18)

    audit_totals = {"sampled": 0, "agreements": 0, "disagreements": 0, "thirdAdjudications": 0}
    for audit_path in args.low18_evaluation.glob("cases/*/run_*/semantic_audit.json"):
        summary = read_json(audit_path)["summary"]
        for key in audit_totals:
            audit_totals[key] += int(summary[key])
    audit_rate = audit_totals["agreements"] / audit_totals["sampled"]
    numeric_changed_cases = 0
    threshold_changed_cases = 0
    for case in low18["cases"]:
        strict_num = case["primary"]["fields"]["Num"]["f1"]["mean"]
        sensitivities = [item["f1"]["mean"] for item in case["numericSensitivity"].values()]
        if any(abs(value - strict_num) > 1e-15 for value in sensitivities):
            numeric_changed_cases += 1
        thresholds = [case["thresholdSensitivity"][key]["f1"]["mean"] for key in ("0.7", "0.8", "0.9")]
        if len(set(thresholds)) > 1:
            threshold_changed_cases += 1

    evidence_sources = {
        "integrity": (args.integrity_audit, "integrity_audit.json", "11,050 筆完整性稽核"),
        "historical_v5": (historical_path, "historical_v5_report.json", "八組 historical v5 報告"),
        "direct_v5": (direct_path, "direct_v5_report.json", "五組 direct ablation v5 報告"),
        "low18_v5": (low18_path, "low18_v5_report.json", "18 組低 precision v5 稽核"),
        "frozen_v4": (args.old_v4_scorecard, "frozen_v4_scorecard.tsv", "既有 frozen hybrid-v4 scorecard"),
        "advice": (args.evaluation_advice, "評估建議.docx", "narrative 評估建議"),
        "evaluator": (args.evaluator_code, "reference_aligned_v5.py", "reference-aligned hybrid-v5.1 evaluator"),
    }
    manifest_sources: list[dict[str, Any]] = []
    canonical_sources: list[dict[str, Any]] = []
    for source_id, (source_path, filename, label) in evidence_sources.items():
        relative = Path("evidence") / filename
        copy_evidence(source_path.resolve(), output_dir / relative)
        manifest_sources.append({"id": source_id, "label": label, "path": relative.as_posix()})
        canonical_sources.append({"id": source_id, "label": label, "path": relative.as_posix()})

    comparison_relative = Path("evidence") / "low18_comparison.json"
    (output_dir / comparison_relative).write_text(
        json.dumps(low_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison_source = {"id": "low18_comparison", "label": "frozen v4 與 v5 衍生比較表", "path": comparison_relative.as_posix()}
    manifest_sources.append(comparison_source)
    canonical_sources.append(comparison_source)

    dataset_sources = (
        ("historical_cases_sql", "historical_cases.sql", "八組 historical 衍生表", "historical_cases", historical_rows),
        ("ablations_sql", "ablations.sql", "Mistral 消融圖資料", "ablations", ablations),
        ("field_detail_sql", "field_detail.sql", "Mistral 欄位 TP/FP/FN", "field_detail", fields),
        ("low18_sql", "low18.sql", "18 組 v4/v5 比較", "low18", low_rows),
        ("low18_causes_sql", "low18_causes.sql", "18 組根因分類", "low18_causes", low_causes),
        ("direct_cases_sql", "direct_cases.sql", "五組 direct 消融表", "direct_cases", direct_rows),
    )
    for source_id, filename, label, table_name, rows in dataset_sources:
        relative = Path("evidence") / filename
        write_values_sql(output_dir / relative, table_name, rows)
        source = {"id": source_id, "label": label, "path": relative.as_posix()}
        manifest_sources.append(source)
        canonical_sources.append(source)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    direct_validity = sum(row["schema_validity"] for row in direct_rows) / len(direct_rows)
    mistral_d = next(row for row in historical_rows if row["case"] == "6_mistral_base_d")
    mistral_m = next(row for row in historical_rows if row["case"] == "6_mistral_base_m")

    cards: list[dict[str, Any]] = []

    charts = [{
        "id": "mistral_ablation",
        "title": "兩組 Mistral 控制案例的 mean macro-F1 消融梯",
        "subtitle": "評估放寬只修正可比性；未解決極低 anchor coverage。",
        "type": "bar",
        "dataset": "ablations",
        "sourceId": "ablations_sql",
        "valueFormat": "percent",
        "legend": {"position": "bottom"},
        "encodings": {
            "x": {"field": "stage", "type": "nominal", "label": "評估階段"},
            "y": {"field": "macro_f1", "type": "quantitative", "label": "Mean macro-F1"},
            "color": {"field": "case", "type": "nominal", "label": "Case"},
            "tooltip": [{"field": "macro_f1", "type": "quantitative", "label": "Macro-F1", "format": "percent"}],
        },
    }]

    tables = [
        {"id": "historical_cases", "title": "八組 fresh historical rerun", "subtitle": "十輪平均與 pooled 指標分列；conditional 僅描述成功 anchor。", "dataset": "historical_cases", "sourceId": "historical_cases_sql", "defaultSort": {"field": "macro_f1", "direction": "desc"}, "columns": [
            {"field": "case", "label": "Case", "type": "text"},
            {"field": "predicted_bindings", "label": "Predicted", "format": "number"},
            {"field": "matched_bindings", "label": "Matched", "format": "number"},
            {"field": "anchor_recall", "label": "Anchor recall", "format": "percent"},
            {"field": "macro_precision", "label": "Mean macro-P", "format": "percent"},
            {"field": "macro_recall", "label": "Mean macro-R", "format": "percent"},
            {"field": "macro_f1", "label": "Mean macro-F1", "format": "percent"},
            {"field": "conditional_f1", "label": "Conditional F1", "format": "percent"},
        ]},
        {"id": "field_detail", "title": "Mistral pooled 欄位 TP／FP／FN", "subtitle": "低欄位分數主要由 1,700+ FN 造成，而非 precision 聚合錯誤。", "dataset": "field_detail", "sourceId": "field_detail_sql", "defaultSort": {"field": "case", "direction": "asc"}, "columns": [
            {"field": "case", "label": "Case", "type": "text"}, {"field": "field", "label": "Field", "type": "text"},
            {"field": "tp", "label": "TP", "format": "number"}, {"field": "fp", "label": "FP", "format": "number"}, {"field": "fn", "label": "FN", "format": "number"},
            {"field": "precision", "label": "P", "format": "percent"}, {"field": "recall", "label": "R", "format": "percent"}, {"field": "f1", "label": "F1", "format": "percent"},
        ]},
        {"id": "low18", "title": "18 組 precision < 0.3：frozen v4 與 v5 診斷", "subtitle": "v5 不覆寫正式排名；分數升降須連同 coverage 與錯誤類型解讀。", "dataset": "low18", "sourceId": "low18_sql", "defaultSort": {"field": "v4_precision", "direction": "asc"}, "columns": [
            {"field": "case", "label": "Case", "type": "text"},
            {"field": "v4_precision", "label": "v4 P", "format": "percent"}, {"field": "v4_f1", "label": "v4 F1", "format": "percent"},
            {"field": "v5_precision", "label": "v5 P", "format": "percent"}, {"field": "v5_recall", "label": "v5 R", "format": "percent"}, {"field": "v5_f1", "label": "v5 F1", "format": "percent"},
            {"field": "predicted_bindings", "label": "Predicted", "format": "number"}, {"field": "matched_bindings", "label": "Matched", "format": "number"},
        ]},
        {"id": "low18_causes", "title": "18 組失分根因", "subtitle": "生成漏失、舊整列 gate 低估、真正語意錯誤與未配對項分列。", "dataset": "low18_causes", "sourceId": "low18_causes_sql", "defaultSort": {"field": "generator_no_binding_rows", "direction": "desc"}, "columns": [
            {"field": "case", "label": "Case", "type": "text"},
            {"field": "generator_no_binding_rows", "label": "No-binding rows", "format": "number"},
            {"field": "whole_row_gate_undercount_rows", "label": "Gate undercount", "format": "number"},
            {"field": "matched_binding_semantic_errors", "label": "Semantic errors", "format": "number"},
            {"field": "unmatched_gold_bindings", "label": "Unmatched gold", "format": "number"},
            {"field": "unmatched_prediction_bindings", "label": "Unmatched pred", "format": "number"},
        ]},
        {"id": "direct_cases", "title": "五組 direct-diagnostic 消融", "subtitle": "不載 adapter、不經 converter、structured output 關閉。", "dataset": "direct_cases", "sourceId": "direct_cases_sql", "defaultSort": {"field": "case", "direction": "asc"}, "columns": [
            {"field": "case", "label": "Case", "type": "text"}, {"field": "schema_validity", "label": "Schema validity", "format": "percent"},
            {"field": "predicted_bindings", "label": "Predicted", "format": "number"}, {"field": "matched_bindings", "label": "Matched", "format": "number"},
            {"field": "macro_precision", "label": "Mean macro-P", "format": "percent"}, {"field": "macro_recall", "label": "Mean macro-R", "format": "percent"}, {"field": "macro_f1", "label": "Mean macro-F1", "format": "percent"},
        ]},
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}\n\n技術稽核報告；產生時間：{generated_at}。"},
        {"id": "summary", "type": "markdown", "body": "## 結論：瓶頸在生成 coverage，而非單一 evaluator 門檻\n\nReference-aligned hybrid-v5.1 保留已命中 anchor 的正確欄位，能揭露 frozen hybrid-v4 的整列 gate 低估；然而六組 FLAN historical case 仍無 matched binding，故合理放寬語意比較也不能創造 TP。`mistral_base_d`／`mistral_base_m` 呈現高 precision、極低 recall：模型只提交極少數可接受 binding。v5.1 是獨立診斷，不覆寫既有正式排名。"},
        {"id": "key_findings", "type": "markdown", "body": "## 分數上升只證明舊 gate 曾連坐，未證明生成品質提高\n\n消融從 frozen v4 到局部欄位處罰，再到確定式與 GPT-5.5 語意判定。若增益只發生在已成功硬 anchor 的內容欄，應解讀為測量偏誤修正；coverage 不變表示生成器仍漏失相同 binding。"},
        {"id": "ablation_chart", "type": "chart", "chartId": "mistral_ablation"},
        {"id": "scope", "type": "markdown", "body": "## 範圍、母體與分母\n\nGold 為 85 Sources、173 bindings；每 case 跑 10 次，故每欄 gold 分母為 1,730。端到端指標納入未配對與 schema-invalid 輸出。條件式內容分數(conditional content score)只限成功硬 anchor 的 binding，零 matched 時記為 N/A，且不得排名。"},
        {"id": "historical_table", "type": "table", "tableId": "historical_cases"},
        {"id": "precision_explanation", "type": "markdown", "body": f"## 為何 precision 正常、各欄 F1 卻極低\n\nFresh `mistral_base_d` 只 matched {mistral_d['matched_bindings']}／1,730 bindings，mean macro-precision 為 {mistral_d['macro_precision']:.3f}，但 mean macro-recall 僅 {mistral_d['macro_recall']:.6f}。`mistral_base_m` matched {mistral_m['matched_bindings']}／1,730，mean macro-precision {mistral_m['macro_precision']:.3f}、recall {mistral_m['macro_recall']:.6f}。precision 的分母只含模型實際提交的正例；FN 則壓低 recall 與 F1。舊官方 `mistral_base_d` 的 0.8 是 8 個 run 各 1 TP、2 個空 run 的 run-mean precision `(8×1+2×0)/10`，不可與 pooled field F1 互比。"},
        {"id": "field_table", "type": "table", "tableId": "field_detail"},
        {"id": "method", "type": "markdown", "body": "## v5 方法：先硬參照對齊，再作欄位局部判分\n\n`DataName` 以 trim＋lowercase 比對；`Position` 保留 JSON 型別、陣列長度、順序和值，僅忽略 object key 順序。兩者組成同一 Source 內的一對一 anchor；重複 prediction 只取第一筆。`ObjectName` 先 NFKC／大小寫／空白正規化，再由 GPT-5.5 判同一實體或明確共指，且禁止用 DataName 代替。`Trend` 先映射固定方向與圖形類別，未決項再審方向、期間、基準與範圍。`Num` 主分數維持有限 JSON number array 與 `isclose(1e-9)`；單位與 0.1%／0.5%／1% 容差只作 sensitivity。`Text` 語意判定要求完整命題的主體、趨勢、數值、時間、範圍、基準及否定極性；finance-safe token F1 只作補充。"},
        {"id": "low18_heading", "type": "markdown", "body": "## 18 組低 precision：逐案看 anchor、局部語意錯誤與 route\n\n比較表同列 frozen v4 與 v5；不得只憑 v5 分數變高宣稱 evaluator 改善。有效證據是 TP/FP/FN 可重算、被舊 gate 抹除的正確欄位可定位、judge 稽核一致，且新方法不把漏生成偽裝成語意等價。"},
        {"id": "low18_table", "type": "table", "tableId": "low18"},
        {"id": "low18_causes_table", "type": "table", "tableId": "low18_causes"},
        {"id": "route_heading", "type": "markdown", "body": "## 直出 route 檢驗 converter 的角色\n\n真正直出由 base model 直接讀 binding prompt；不載 adapter、不經 converter、`structured_output=off`，原始 completion 只接受 strict JSON parser。此消融用來區分 route 影響與 evaluator 影響，不混入正式結果。"},
        {"id": "direct_table", "type": "table", "tableId": "direct_cases"},
        {"id": "robustness", "type": "markdown", "body": f"## 不確定性與穩健性\n\n每案報十輪 mean、sample SD、min/max；top-1/top-3 僅補充。Judge 主門檻為 0.8，另列 0.7／0.9。A/B swap 實際抽樣 {audit_totals['sampled']} 筆，{audit_totals['agreements']} 筆一致（{audit_rate:.1%}）；{audit_totals['disagreements']} 筆歧異均進第三裁決。Num 的 tolerance／單位 sensitivity 實際改變 {numeric_changed_cases}／18 cases；judge 門檻實際改變 {threshold_changed_cases}／18 cases。這些 sensitivity 與 token F1 均不進六欄主要 macro-F1。Fresh converter 結果可能受服務期狀態影響：`mistral_base_d` 從舊 8 次命中變為 fresh 9 次命中，候選 SHA 相同而 converter raw 不同，因此是 converter sufficiency 決策變異，不是 coverage 大幅改善。"},
        {"id": "next_steps", "type": "markdown", "body": "## 建議下一步\n\n1. 正式排名續用 frozen hybrid-v4，v5 僅作診斷與敏感度附錄。\n2. 生成器優先改善 non-empty binding coverage 與 hard anchor 正確率，再談語意 scorer。\n3. 將 ObjectName 的共指錯誤與 DataName substitution 分開訓練／提示。\n4. 對 converter sufficiency 決策固定服務版本並保存 request／response identity。\n5. 新資料先跑 schema、anchor、TP/FP/FN 恆等式與 A/B audit，再解讀總分。"},
        {"id": "questions", "type": "markdown", "body": "## 待驗問題\n\n- Base model 為何長篇複誦 prompt，而非結束於 JSON root？\n- converter 的接受準則能否校準 coverage，而不犧牲目前的高 precision？\n- ObjectName 共指可否以人工雙盲樣本估計 judge 的真實錯誤率？\n- 新 prompt 或 fine-tuning 是否能提高 anchor recall，而不靠 evaluator 放寬？"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": TITLE,
            "description": "Experiment 6 fresh rerun、route 消融與 reference-aligned hybrid-v5.1 技術稽核。",
            "generatedAt": generated_at, "cards": cards, "charts": charts,
            "tables": tables, "sources": manifest_sources, "blocks": blocks,
        },
        "snapshot": {
            "version": 1, "generatedAt": generated_at, "status": "ready",
            "datasets": {
                "headline": [{"fresh_predictions": integrity["actualFreshPredictions"], "historical_predictions": integrity["historical"]["predictionCount"], "direct_validity": direct_validity, "low_precision_cases": len(low_rows)}],
                "historical_cases": historical_rows, "direct_cases": direct_rows,
                "ablations": ablations, "field_detail": fields,
                "root_causes": causes, "low18": low_rows, "low18_causes": low_causes,
            },
        },
        "sources": canonical_sources,
        "package_info": {"originUrl": "artifact://experiment6-reference-aligned-v5.1"},
    }
    (output_dir / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_dir / "artifact.json")


if __name__ == "__main__":
    main()
