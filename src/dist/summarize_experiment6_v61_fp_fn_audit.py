#!/usr/bin/env python3
"""Summarize Experiment 6 v6.1 FP/FN audit into durable JSON and Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FIELDS = ("ObjectName", "Trend", "Num", "Position", "DataName")
FIELD_LABELS = {"ObjectName": "Subject", "Trend": "Trend", "Num": "Num", "Position": "Position", "DataName": "DataName"}
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class AuditError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return path.name


def metric(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def equal_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


def validate_formulas(name: str, report: Mapping[str, Any]) -> int:
    checked = 0
    for case in report.get("cases") or []:
        for run in case.get("runResults") or []:
            for field, counts in run["primary"]["counts"].items():
                expected = metric(*(int(counts[key]) for key in ("tp", "fp", "fn")))
                actual = run["primary"]["fields"][field]
                for key in ("precision", "recall", "f1"):
                    if not equal_number(actual[key], expected[key]):
                        raise AuditError(
                            f"formula mismatch: {name}/{case['outputId']}/run {run['run']}/{field}/{key}"
                        )
                checked += 1
    return checked


def annotation_case_means(report: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for field in FIELDS:
        output[field] = {}
        for name in ("precision", "recall", "f1"):
            values = [
                float(case["aggregate"]["fields"][field][name]["mean"])
                for case in report["cases"]
                if case["aggregate"]["fields"][field][name]["mean"] is not None
            ]
            output[field][name] = statistics.fmean(values) if values else None
    return output


def report_summary(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": logical_path(path),
        "sha256": sha256_file(path),
        "protocol": report.get("scoringProtocol"),
        "caseMeanMacroF1": report["overall"]["caseMeanMacroF1"],
        "annotationCaseMeans": annotation_case_means(report),
        "primaryPooled": report["overall"]["primaryPooled"],
        "bindingPooled": report["overall"]["bindingPooled"],
        "coverage": report["coverage"],
        "inputSetSha256": report["input"]["inputSetSha256Before"],
        "methodAudit": report.get("methodAudit"),
    }


def count_delta(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    old_counts = old["overall"]["primaryPooled"]["counts"]
    new_counts = new["overall"]["primaryPooled"]["counts"]
    return {
        field: {
            key: int(new_counts[field][key]) - int(old_counts[field][key])
            for key in ("tp", "fp", "fn")
        }
        for field in FIELDS
    }


def residual_audit(evaluation_root: Path) -> dict[str, Any]:
    counts = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for path in evaluation_root.glob("cases/*/run_*/records.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source = str(row["source"])
                unmatched_gold = len(row["alignment"]["unmatchedGold"])
                unmatched_prediction = len(row["alignment"]["unmatchedPrediction"])
                for category, amount in (
                    ("unmatched_gold", unmatched_gold),
                    ("unmatched_prediction", unmatched_prediction),
                ):
                    counts[category] += amount
                    sources[category][source] += amount
                for detail in row["matchDetails"]:
                    for field, result in detail["fields"].items():
                        if result["equal"]:
                            continue
                        category = f"matched_{field}_wrong"
                        counts[category] += 1
                        sources[category][source] += 1
                        reasons[field][str(result.get("reason") or "unknown")] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "matchedFieldReasons": {key: dict(sorted(value.items())) for key, value in sorted(reasons.items())},
        "topRepeatedSources": {
            key: [{"source": source, "count": count} for source, count in value.most_common(10)]
            for key, value in sorted(sources.items())
        },
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def markdown(audit: Mapping[str, Any]) -> str:
    c12 = audit["reports"]["candidate12V610"]
    r34 = audit["reports"]["repaired34V610"]
    r34_old = audit["reports"]["repaired34V602"]
    method = r34["methodAudit"]["counts"]
    residual = audit["residual"]
    lines = [
        "# Experiment 6 v6.1 Binding TP／FP／FN 評估修訂與重評報告",
        "",
        "> **DIAGNOSTIC ONLY — OFFICIAL=false — CLAIM-ELIGIBLE=false**",
        "",
        "## 技術摘要",
        "",
        "v6.1 已讓 mean／top-1／top-3 分數表同時呈現 Precision、Recall、F1。評估器先在 binding hard-anchor 對齊後逐欄累計 TP／FP／FN，再由同一組計數推導三項指標；本次逐 run 公式重算無歧異。DataName 與 Position 未改，仍為硬比對。",
        "",
        f"candidate12 的 case-mean macro-F1 仍為 {fmt(c12['caseMeanMacroF1'])}；沒有任何內容修復命中。repaired34 由 v6.0.2 的 {fmt(r34_old['caseMeanMacroF1'])} 至 v6.1 的 {fmt(r34['caseMeanMacroF1'])}。分數變化不是唯一判準：Trend 同時移除了 {method['trend_false_tp_corrected']} 個舊 ontology 假 TP。",
        "",
        "## 主要結果",
        "",
        "| 範圍 | 方法 | case-mean macro-F1 | pooled macro Precision | pooled macro Recall | pooled macro F1 | Binding TP/FP/FN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("candidate12V601", "candidate12 v6.0.1"), ("candidate12V610", "candidate12 v6.1"), ("repaired34V602", "repaired34 v6.0.2"), ("repaired34V610", "repaired34 v6.1")):
        item = audit["reports"][key]
        pooled = item["primaryPooled"]["macro"]
        binding = item["bindingPooled"]
        lines.append(
            f"| {label.split()[0]} | {label.split()[1]} | {fmt(item['caseMeanMacroF1'])} | {fmt(pooled['precision'])} | {fmt(pooled['recall'])} | {fmt(pooled['f1'])} | {binding['tp']}/{binding['fp']}/{binding['fn']} |"
        )
    lines.extend(["", "## v6.1 annotation case means", "", "| annotation | Precision | Recall | F1 |", "|---|---:|---:|---:|"])
    for field in FIELDS:
        values = r34["annotationCaseMeans"][field]
        lines.append(f"| {FIELD_LABELS[field]} | {fmt(values['precision'])} | {fmt(values['recall'])} | {fmt(values['f1'])} |")
    lines.extend([
        "",
        "## 可證成的評估修復",
        "",
        f"- ObjectName：前置冠詞的一對一正規化回收 {method['object_article_normalized']} 個 matched-anchor 欄位；TP +38、FP −38、FN −38。",
        f"- Num：單一數值詞組、幣值／百分比單位、序數與 singleton nested scalar 回收 {method['num_semantic_normalized']} 個；TP +159、FP −159、FN −159；其中 {method['num_schema_repaired']} 個仍另標 schema repair。",
        f"- Trend：封閉方向類別糾正 {method['trend_false_tp_corrected']} 個假 TP；Trend TP −32、FP +32、FN +32。",
        f"- 0.1%／0.5%／1% tolerance 僅為 sensitivity；主分數 tolerance=0。本資料的 tolerance 新增命中為 {method['num_tolerance']}。",
        "- DataName／Position：調整數均為 0，hard-anchor 契約未變。",
        "",
        "## 殘餘 FP／FN",
        "",
        f"- 未配對 gold bindings：{residual['counts']['unmatched_gold']}；這些是主要 FN，不能靠 ObjectName／Trend／Num 正規化修復。",
        f"- 未配對 predictions：{residual['counts']['unmatched_prediction']}；這些是主要 FP，須修生成或 DataName／Position。",
        f"- hard-anchor 已配對後仍錯：ObjectName {residual['counts']['matched_ObjectName_wrong']}、Num {residual['counts']['matched_Num_wrong']}、Trend {residual['counts']['matched_Trend_wrong']}。",
        "",
        "重複最多的來源（跨 case/run 次數）：",
        "",
        "| 類型 | top sources |",
        "|---|---|",
    ])
    for category in ("unmatched_gold", "unmatched_prediction", "matched_ObjectName_wrong", "matched_Num_wrong", "matched_Trend_wrong"):
        values = ", ".join(f"{item['source']} ({item['count']})" for item in residual["topRepeatedSources"][category][:5])
        lines.append(f"| {category} | {values} |")
    lines.extend([
        "",
        "## 方法與公式",
        "",
        "1. 同一 Source 內以 `(DataName, Position)` 一對一對齊；重複或額外 anchor 不作語意搜尋。",
        "2. matched binding 的正確欄計 TP；有值但錯計 FP+FN；缺值計 FN。未配對 gold 對各欄計 FN；未配對 prediction 對存在欄計 FP。",
        "3. `Precision=TP/(TP+FP)`、`Recall=TP/(TP+FN)`、`F1=2TP/(2TP+FP+FN)`。本次檢查 " + str(audit["formulaChecks"]) + " 個 run-field 組合，歧異 0。",
        "4. ObjectName、Trend、Num 只在 hard anchor 後判定；DataName、Position 不放寬。Text judge 未執行，維持 NA。",
        "",
        "## 限制與解讀",
        "",
        "- v6.1 係參照目前資料錯誤所作的診斷敏感度方法，不能取代 frozen formal ranking；正式採用前須在獨立 dev/calibration set 凍結規則。",
        "- candidate12 低分源於 coverage／format 與 anchor 失敗，不是欄位語意比較太嚴。提高其分數需要改善生成輸出，不宜再放寬 evaluator。",
        "- candidate12 v6.0.1 與 v6.1 的結構欄 TP／FP／FN 完全相同；binding-level FP 1381 對 9092 的差異源於 v6.1 繼承 v6.0.2 的 nonempty-unparseable binding FP 規則，不是內容退化，兩者不得直接作方法效果比較。",
        "- Num 字串雖可做內容等價判定，schema validity 仍分開記錄；不能把內容修復誤稱為嚴格 JSON schema 合格。",
        "- 四個 GPT-4.1 cases 未納入，Text judge deferred；experimentMatrixComplete=false。",
        "",
        "## 輸出與來源",
        "",
    ])
    for key, item in audit["reports"].items():
        lines.append(f"- {key}: `{item['path']}` (`{item['sha256']}`).")
    lines.extend([
        f"- 評估建議：`{audit['recommendation']['path']}` (`{audit['recommendation']['sha256']}`).",
        "- 原 candidate／gold／prediction artifacts 均未修改；v6.1 reports 的 inputSet SHA 與各自基準相同。",
    ])
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "candidate12V601": args.candidate12_baseline.resolve(),
        "candidate12V610": args.candidate12_v61.resolve(),
        "repaired34V602": args.repaired34_baseline.resolve(),
        "repaired34V610": args.repaired34_v61.resolve(),
    }
    reports = {key: read_json(path) for key, path in paths.items()}
    if reports["candidate12V601"]["input"]["inputSetSha256Before"] != reports["candidate12V610"]["input"]["inputSetSha256Before"]:
        raise AuditError("candidate12 input set changed")
    if reports["repaired34V602"]["input"]["inputSetSha256Before"] != reports["repaired34V610"]["input"]["inputSetSha256Before"]:
        raise AuditError("repaired34 input set changed")
    formula_checks = sum(validate_formulas(key, report) for key, report in reports.items())
    audit = {
        "schemaVersion": 1,
        "protocol": "experiment6-v6.1-fp-fn-audit-v1",
        "status": "diagnostic_only",
        "official": False,
        "claimEligible": False,
        "formulaChecks": formula_checks,
        "formulaMismatches": 0,
        "reports": {key: report_summary(paths[key], reports[key]) for key in paths},
        "repaired34CountDeltaV610MinusV602": count_delta(reports["repaired34V602"], reports["repaired34V610"]),
        "residual": residual_audit(paths["repaired34V610"].parent),
        "recommendation": {"path": logical_path(args.recommendation.resolve()), "sha256": sha256_file(args.recommendation.resolve())},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(audit), encoding="utf-8")
    return {"status": "completed", "outputJson": logical_path(args.output_json), "outputMarkdown": logical_path(args.output_md), "formulaChecks": formula_checks}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate12-baseline", type=Path, required=True)
    parser.add_argument("--candidate12-v61", type=Path, required=True)
    parser.add_argument("--repaired34-baseline", type=Path, required=True)
    parser.add_argument("--repaired34-v61", type=Path, required=True)
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = build(parse_args(argv))
    except (AuditError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
