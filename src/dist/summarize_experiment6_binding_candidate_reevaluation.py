#!/usr/bin/env python3
"""Summarize four claim-ineligible Experiment 6 candidate evaluations."""

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

from evaluate_experiment6_binding_candidates_v1 import PROTOCOL, logical_path, sha256_file


REPORT_KEYS = (
    "candidate_v601",
    "candidate_v602",
    "merged_v601",
    "merged_v602",
)


class SummaryError(RuntimeError):
    """Raised when candidate evaluation reports are not comparable."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def mean_defined(values: Sequence[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(numbers) if numbers else None


def field_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(case["outputId"]): {
            "runs": [
                {
                    "run": int(run["run"]),
                    "primary": run["primary"],
                    "withoutTrendAblation": run["withoutTrendAblation"],
                }
                for run in case["runResults"]
            ],
            "aggregate": {
                "fields": case["aggregate"]["fields"],
                "macro": case["aggregate"]["macro"],
                "micro": case["aggregate"]["micro"],
                "pooled": case["aggregate"]["pooled"],
            },
            "selection": case["selection"],
        }
        for case in report["cases"]
    }


def binding_counts(report: Mapping[str, Any]) -> dict[str, int]:
    values = report["overall"]["bindingPooled"]
    return {key: int(values[key]) for key in ("tp", "fp", "fn")}


def validate_report(report: Mapping[str, Any], scope: str, version: str, cases: int) -> None:
    require(report.get("protocol") == PROTOCOL, "candidate protocol mismatch")
    require(report.get("scope") == scope, f"scope mismatch: expected {scope}")
    require(str(report.get("scoringProtocol")).endswith(version), f"scoring version mismatch: {version}")
    require(report.get("status") == "diagnostic_only", "candidate status mismatch")
    require(report.get("official") is False and report.get("claimEligible") is False, "claim boundary mismatch")
    require(report.get("selectionRole") == "diagnostic-descriptive-only", "selection role mismatch")
    require(report.get("scopeComplete") is True, "scope incomplete")
    require(report.get("experimentMatrixComplete") is False, "matrix completion incorrectly claimed")
    require(int(report.get("completedCases", 0)) == cases, "case count mismatch")
    require(int(report.get("completedCaseRuns", 0)) == cases * 10, "case-run count mismatch")
    require(report.get("input", {}).get("inputSetSha256Before") == report.get("input", {}).get("inputSetSha256After"), "input set changed")
    require(report.get("runtime", {}).get("textJudge") == "disabled", "Text judge unexpectedly used")
    require(report.get("runtime", {}).get("chatMockUsed") is False, "ChatMock unexpectedly used")
    require(report.get("runtime", {}).get("newGpuProcess") is False, "new GPU process detected")


def model_diagnostics(report: Mapping[str, Any], registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = {str(item["outputId"]): item for item in report["caseMetadata"]}
    stats = report["candidateStatsByCase"]
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    scores: dict[str, list[float]] = defaultdict(list)
    for case in report["cases"]:
        output_id = str(case["outputId"])
        source_id = str(metadata[output_id]["sourceId"])
        model = str((registry.get("sources") or {}).get(source_id, {}).get("baseModel") or source_id)
        target = grouped[model]
        case_stats = stats[output_id]
        target.update({key: int(value) for key, value in case_stats.items() if not key.startswith("status:")})
        for run in case["runResults"]:
            target["goldBindings"] += int(run["coverage"].get("gold_bindings", 0))
            target["matchedBindings"] += int(run["coverage"].get("matched_bindings", 0))
            target["formatInvalidRows"] += int(run["coverage"].get("format_invalid_rows", 0))
            target["emptyOutputRows"] += int(run["coverage"].get("empty_output_rows", 0))
        score = case["aggregate"]["macro"]["f1"]["mean"]
        if score is not None:
            scores[model].append(float(score))
    rows = []
    for model, values in sorted(grouped.items()):
        rows.append({
            "retrieverModel": model,
            **dict(values),
            "acceptedRate": values.get("acceptedRows", 0) / values["rows"] if values["rows"] else None,
            "anchorMatchRate": values["matchedBindings"] / values["goldBindings"] if values["goldBindings"] else None,
            "formatInvalidRate": values["formatInvalidRows"] / values["rows"] if values["rows"] else None,
            "caseMeanMacroF1": mean_defined(scores[model]),
        })
    return rows


def aggregate_row(name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    counts = binding_counts(report)
    return {
        "name": name,
        "scope": report["scope"],
        "scoringProtocol": report["scoringProtocol"],
        "cases": report["completedCases"],
        "caseRuns": report["completedCaseRuns"],
        "caseMeanMacroF1": report["overall"]["caseMeanMacroF1"],
        "withoutTrendCaseMeanMacroF1": report["overall"]["withoutTrendCaseMeanMacroF1"],
        "bindingCounts": counts,
        "bindingMetrics": {
            key: report["overall"]["bindingPooled"][key]
            for key in ("precision", "recall", "f1")
        },
        "formatInvalidRate": report["coverage"]["formatInvalidRate"],
        "emptyOutputRate": report["coverage"]["emptyOutputRate"],
        "anchorMatchRate": report["coverage"]["anchorMatchRate"],
        "formalCaseMeanMacroF1": report["formalComparison"]["formalCaseMeanMacroF1"],
        "inputSetSha256": report["input"]["inputSetSha256Before"],
        "methodSha256": report["method"]["methodSha256"],
    }


def table_files(table_roots: Mapping[str, Path]) -> list[dict[str, Any]]:
    artifacts = []
    for key, root in sorted(table_roots.items()):
        for path in sorted(root.glob("experiment_6_v6_*")):
            if path.is_file():
                artifacts.append({"group": key, "path": logical_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return artifacts


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_report(audit: Mapping[str, Any]) -> str:
    aggregates = audit["evaluations"]
    lines = [
        "# Experiment 6 整理後 Binding 雙軌診斷重評",
        "",
        "## 技術摘要",
        "",
        f"- 整理後 candidate12 的五欄 case-mean macro-F1 為 **{fmt(audit['thresholdDecision']['observed'])}**，低於 0.1；此結果為診斷性敏感度證據，不取代正式 corrected12 的 0 分。",
        f"- candidate-merged34 的 v6.0.2 五欄 case-mean macro-F1 為 **{fmt(aggregates['merged_v602']['caseMeanMacroF1'])}**；其中 12 cases 使用整理後 bindings，22 cases 沿用歷史 predictions，故仍不可作正式主張。",
        "- v6.0.1 與 v6.0.2 的五欄 TP／FP／FN、F1 與共同 run 排名完全一致；差別只在不可解析非空回應的 binding-level FP。",
        "- Text judge、ChatMock 與 GPU 推論均未使用；四個 GPT-4.1 cases 仍缺，38-case matrix 未完成。",
        "",
        "## 四組結果與正式基準",
        "",
        "| scope | scoring | cases/runs | candidate macro-F1 | formal macro-F1 | binding TP/FP/FN | format-invalid | anchor match |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in REPORT_KEYS:
        row = aggregates[key]
        counts = row["bindingCounts"]
        lines.append(
            f"| {row['scope']} | {row['scoringProtocol']} | {row['cases']}/{row['caseRuns']} | "
            f"{fmt(row['caseMeanMacroF1'])} | {fmt(row['formalCaseMeanMacroF1'])} | "
            f"{counts['tp']}/{counts['fp']}/{counts['fn']} | {fmt(row['formatInvalidRate'])} | {fmt(row['anchorMatchRate'])} |"
        )
    lines.extend([
        "",
        "此表以 case 為等權單位：先算每 case 的十次 run mean，再跨 cases 平均。top-1／top-3 僅在各自表格中依同一份 shared run order 呈現，不參與低分判定。",
        "",
        "## Candidate 資料完整，但可用 coverage 很低",
        "",
        "| retriever model | rows | schema-valid | accepted rate | anchor match | format-invalid | case-mean macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in audit["candidate12ByRetrieverModel"]:
        lines.append(
            f"| {row['retrieverModel']} | {row['rows']} | {row.get('acceptedRows', 0)} | {fmt(row['acceptedRate'])} | "
            f"{fmt(row['anchorMatchRate'])} | {fmt(row['formatInvalidRate'])} | {fmt(row['caseMeanMacroF1'])} |"
        )
    validation = audit["candidateValidation"]["counts"]
    lines.extend([
        "",
        f"資料的 composite keys、完整六欄 schema、來源雜湊與 inventory 均通過驗證；然而 10,200 rows 中僅 {validation['acceptedRows']} rows 通過 schema，且只有 {validation['acceptedRowsWithBindings']} rows 含至少一個 binding。故低分主要來自 coverage 與 hard-anchor 對齊失敗，而非 ObjectName alias 或 Sentence-BERT 門檻。",
        "",
        "## 計分契約與版本差異",
        "",
        "- 五個主欄為 ObjectName（表中 Subject）、Trend、Num、Position、DataName；Text 為 NA，未進 TP／FP／FN。",
        "- 已輸出但錯誤或型別無效的欄位計 FP+FN；缺欄計 FN；未配對 gold／prediction 分別保留 FN／FP。",
        "- v6.0.2 對非空但不可解析的 rawResponse 額外計一次 binding-level FP，不增添無法定位的 field-level FP。",
        f"- candidate12 v6.0.2 與既有 deterministic-repair sensitivity 的逐 run score projection：**{audit['sensitivityReproduction']['status']}**。",
        "",
        "## 限制、穩健性與可解讀範圍",
        "",
        "- 整理後 bindings 是生成時已保存之 deterministic repair 的 materialization；因事後檢視測試 predictions，永久為 diagnosticOnly／claimEligible=false。",
        "- candidate-merged34 混合兩種 provenance，只能說明整理後 12 cases 對既有 34-case 視圖的敏感度，不能解讀為新模型或正式方法改善。",
        "- 五種特殊 Trend 類別在 gold 的正樣本 support 皆為 0；即使 Trend F1 偏高，仍不能宣稱圖形分類能力。",
        "- 本報告不用圖表：僅四個 scope/version 聚合與三個模型群，精確 audit table 比圖形更可核查。",
        "",
        "## 後續建議",
        "",
        "- 不依本次低分調整別名字典或 Sentence-BERT 門檻；它們尚非主要瓶頸，且以測試分數調參會造成 test-set leakage。",
        "- v6.1 僅能在獨立校準集凍結純語法 wrapper normalization：固定 prefix、code fence 與外部空白；須恰有一個 JSON payload，禁止補 key、改值或用 gold 推測欄位。",
        "- 後續本機生成宜採 schema／grammar-constrained decoding；retriever-to-binding 校正須為純程序、chart-grounded，歧義時拒收，不得用 ChatMock 整理。",
        "",
        "## 尚待回答",
        "",
        "- 欲主張六類 Trend 效能，仍須另建含五類特殊圖形正樣本的凍結測試集。",
        "- 四個 GPT-4.1 cases 完成後，須以同一凍結 protocol 另跑全 38-case 正式分析；不得把本次 diagnostic candidates 直接補入。",
    ])
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "candidate_v601": args.candidate_v601.resolve(),
        "candidate_v602": args.candidate_v602.resolve(),
        "merged_v601": args.merged_v601.resolve(),
        "merged_v602": args.merged_v602.resolve(),
    }
    reports = {key: read_json(path) for key, path in paths.items()}
    validate_report(reports["candidate_v601"], "candidate12", "v6.0.1", 12)
    validate_report(reports["candidate_v602"], "candidate12", "v6.0.2", 12)
    validate_report(reports["merged_v601"], "candidate-merged34", "v6.0.1", 34)
    validate_report(reports["merged_v602"], "candidate-merged34", "v6.0.2", 34)
    require(field_signature(reports["candidate_v601"]) == field_signature(reports["candidate_v602"]), "candidate12 field scores or rankings differ across v6.0.1/v6.0.2")
    require(field_signature(reports["merged_v601"]) == field_signature(reports["merged_v602"]), "candidate-merged34 field scores or rankings differ across v6.0.1/v6.0.2")
    require(reports["candidate_v601"]["input"]["inputSetSha256Before"] == reports["candidate_v602"]["input"]["inputSetSha256Before"], "candidate12 input hashes differ")
    require(reports["merged_v601"]["input"]["inputSetSha256Before"] == reports["merged_v602"]["input"]["inputSetSha256Before"], "merged input hashes differ")
    sensitivity = reports["candidate_v602"].get("sensitivityReproduction")
    require(isinstance(sensitivity, Mapping) and sensitivity.get("status") == "exact-score-projection-match", "sensitivity reproduction did not pass")
    registry = read_json(args.source_registry.resolve())
    observed = reports["candidate_v602"]["overall"]["caseMeanMacroF1"]
    threshold = reports["candidate_v602"]["overall"]["lowScoreThreshold"]
    table_root_map = {}
    for raw in args.table_root:
        key, separator, value = raw.partition("=")
        require(separator == "=" and key in REPORT_KEYS, f"invalid --table-root: {raw}")
        table_root_map[key] = Path(value).resolve()
    require(set(table_root_map) == set(REPORT_KEYS), "four table roots are required")

    audit = {
        "schemaVersion": 1,
        "kind": "experiment6_binding_candidate_dual_track_reevaluation",
        "status": "complete_diagnostic_scoped_34_case_dual_version",
        "time": reports["merged_v602"]["time"],
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "scopeComplete": True,
        "experimentMatrixComplete": False,
        "evaluations": {key: aggregate_row(key, reports[key]) for key in REPORT_KEYS},
        "candidateValidation": reports["candidate_v602"]["candidateValidation"],
        "candidate12ByRetrieverModel": model_diagnostics(reports["candidate_v602"], registry),
        "thresholdDecision": {
            "scope": "candidate12",
            "metric": "unweighted mean across 12 case-level ten-run mean five-field macro-F1",
            "threshold": threshold,
            "observed": observed,
            "triggered": observed is not None and float(observed) < float(threshold),
            "parameterTuningAllowed": False,
            "rootCauseFocus": ["schema-valid coverage", "empty bindings", "hard-anchor match rate"],
        },
        "versionDifference": {
            "candidate12": {
                "fieldScoresAndSharedRankingsEqual": True,
                "v601BindingCounts": binding_counts(reports["candidate_v601"]),
                "v602BindingCounts": binding_counts(reports["candidate_v602"]),
            },
            "candidateMerged34": {
                "fieldScoresAndSharedRankingsEqual": True,
                "v601BindingCounts": binding_counts(reports["merged_v601"]),
                "v602BindingCounts": binding_counts(reports["merged_v602"]),
            },
        },
        "sensitivityReproduction": sensitivity,
        "resourceGuard": {
            "cpuOnly": True,
            "cpuThreads": 4,
            "textJudge": "disabled",
            "chatMockUsed": False,
            "newGpuProcesses": False,
            "perEvaluationGpuSnapshots": {
                key: reports[key]["runtime"] for key in REPORT_KEYS
            },
        },
        "missingMatrixCaseIds": reports["candidate_v602"]["missingMatrixCaseIds"],
        "artifacts": {
            "evaluationReports": [
                {"group": key, "path": logical_path(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for key, path in paths.items()
            ],
            "scoreTables": table_files(table_root_map),
        },
        "prospectiveV61": {
            "status": "planned_only",
            "requiresIndependentCalibration": True,
            "allowedNormalization": ["fixed prefix removal", "code-fence removal", "outer whitespace removal", "exactly one JSON payload"],
            "forbidden": ["gold-guided repair", "key completion", "value changes", "test-derived alias expansion", "test-derived Sentence-BERT tuning", "ChatMock binding rewrite"],
        },
        "limitations": [
            "All candidate scores are diagnostic and claim-ineligible.",
            "The merged scope combines 12 materialized candidate cases and 22 historical official cases.",
            "Text is NA and four GPT-4.1 cases remain missing.",
            "Five special Trend classes have zero positive support in the current gold data.",
        ],
    }
    markdown = markdown_report(audit)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    audit["artifacts"]["technicalReport"] = {
        "path": logical_path(args.output_md.resolve()),
        "sha256": sha256_file(args.output_md.resolve()),
        "bytes": args.output_md.stat().st_size,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-v601", type=Path, required=True)
    parser.add_argument("--candidate-v602", type=Path, required=True)
    parser.add_argument("--merged-v601", type=Path, required=True)
    parser.add_argument("--merged-v602", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--table-root", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build(args)
    except (SummaryError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "status": report["status"],
        "claimEligible": report["claimEligible"],
        "candidate12MacroF1": report["thresholdDecision"]["observed"],
        "lowScoreTriggered": report["thresholdDecision"]["triggered"],
        "outputJson": logical_path(args.output_json),
        "outputMarkdown": logical_path(args.output_md),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
