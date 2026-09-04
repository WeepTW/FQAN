#!/usr/bin/env python3
"""Build the three audit-ready Experiment 6 v6 field-score tables."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_COLUMNS = (
    "fine-tuned method (prompt-type or no-adaptor)",
    "retriever model",
    "input prompt-type",
    "**Subject**",
    "**Trend**",
    "**Num**",
    "**Position**",
    "**DataName**",
    "**Text**",
)
FIELD_MAP = (
    ("Subject", "ObjectName"),
    ("Trend", "Trend"),
    ("Num", "Num"),
    ("Position", "Position"),
    ("DataName", "DataName"),
)
VIEW_LABELS = {
    "mean": "10-run mean",
    "top-1": "共同排序 top-1",
    "top-3": "共同排序 top-3 mean",
}
METHOD_ORDER = {
    "zero-shot": 0,
    "many-shot": 1,
    "dynamic-shot": 2,
    "no-adaptor": 3,
}
MODEL_ORDER = {
    "google/flan-t5-large": 0,
    "mistralai/Mistral-7B-Instruct-v0.3": 1,
    "google/t5gemma-2-1b-1b": 2,
    "gpt-5.5": 3,
}
PROMPT_ORDER = {
    "zero-shot": 0,
    "many-shot": 1,
    "dynamic-shot": 2,
    "original": 3,
}


class TableError(RuntimeError):
    """Raised when a source cannot support a formal score table."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TableError(f"expected JSON object: {path}")
    return value


def format_score(value: Any) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.6f}"


def safe_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def generation_finished_at(rows: Sequence[Mapping[str, Any]]) -> str:
    values = [str(row.get("finishedAt") or "") for row in rows]
    return max(values) if values else "NA"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def score_for(case: Mapping[str, Any], view: str, field: str) -> Any:
    if view == "mean":
        return case["aggregate"]["fields"][field]["f1"]["mean"]
    if view == "top-1":
        return case["selection"]["top1"]["fields"][field]["f1"]
    if view == "top-3":
        return case["selection"]["top3"]["fields"][field]["f1"]
    raise TableError(f"unknown score view: {view}")


def consistent_value(items: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {json.dumps(item.get(key), ensure_ascii=False, sort_keys=True) for item in items}
    if len(values) != 1:
        raise TableError(f"inconsistent {key}: {sorted(values)}")
    return items[0].get(key)


def build_metadata(
    generation_report: Mapping[str, Any],
    sources: Mapping[str, Any],
    expected_runs: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for manifest in generation_report.get("manifests", []):
        if manifest.get("official"):
            grouped[str(manifest["outputId"])].append(manifest)
    metadata: dict[str, dict[str, Any]] = {}
    for output_id, items in grouped.items():
        runs = sorted(int(item["run"]) for item in items)
        if runs != list(range(1, expected_runs + 1)):
            continue
        source_id = str(consistent_value(items, "sourceId"))
        if source_id not in sources:
            raise TableError(f"source registry entry missing: {source_id}")
        source = sources[source_id]
        kind = str(source.get("kind"))
        fine_tuned_method = (
            str(source.get("promptMode"))
            if kind == "adapter"
            else "no-adaptor"
        )
        metadata[output_id] = {
            "outputId": output_id,
            "fineTunedMethod": fine_tuned_method,
            "retrieverModel": str(source.get("baseModel") or source_id),
            "inputPromptType": str(consistent_value(items, "promptMode")),
            "sourceId": source_id,
            "kind": kind,
            "finishedAt": max(str(item.get("finishedAt") or "") for item in items),
        }
    return metadata


def validate_evaluation(
    evaluation: Mapping[str, Any],
    metadata: Mapping[str, Any],
    expected_runs: int,
    protocol: str,
) -> list[Mapping[str, Any]]:
    if evaluation.get("protocol") != protocol:
        raise TableError(
            f"protocol mismatch: expected {protocol}, got {evaluation.get('protocol')}"
        )
    if evaluation.get("mode") != "formal" or evaluation.get("status") not in {
        "completed",
        "completed_with_runtime_blocked",
    }:
        raise TableError("evaluation report is not a completed formal report")
    cases = evaluation.get("cases")
    if not isinstance(cases, list):
        raise TableError("evaluation cases missing")
    ids = [str(case.get("outputId")) for case in cases]
    if len(ids) != len(set(ids)):
        raise TableError("duplicate outputId in evaluation report")
    if set(ids) != set(metadata):
        raise TableError(
            "evaluation/generation case mismatch: "
            f"evaluation-only={sorted(set(ids) - set(metadata))}; "
            f"generation-only={sorted(set(metadata) - set(ids))}"
        )
    for case in cases:
        if int(case.get("runs", 0)) != expected_runs:
            raise TableError(f"{case.get('outputId')} does not have {expected_runs} runs")
        selection = case.get("selection")
        if not isinstance(selection, Mapping):
            raise TableError(f"{case.get('outputId')} has no shared formal selection")
        if len(selection["top3"]["runs"]) != 3:
            raise TableError(f"{case.get('outputId')} top-3 is incomplete")
    return cases


def row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        METHOD_ORDER.get(str(row["fineTunedMethod"]), 99),
        MODEL_ORDER.get(str(row["retrieverModel"]), 99),
        PROMPT_ORDER.get(str(row["inputPromptType"]), 99),
        str(row["outputId"]),
    )


def current_status(
    report: Mapping[str, Any],
    expected_runs: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for manifest in report.get("manifests", []):
        if manifest.get("official"):
            grouped[str(manifest["outputId"])].append(manifest)
    rows = []
    for output_id, items in grouped.items():
        statuses = Counter(str(item.get("status")) for item in items)
        run_count = len({int(item["run"]) for item in items})
        rows.append(
            {
                "outputId": output_id,
                "runs": run_count,
                "expectedRuns": expected_runs,
                "state": "complete" if run_count == expected_runs else "in-progress",
                "statuses": dict(sorted(statuses.items())),
                "lastFinishedAt": max(str(item.get("finishedAt") or "") for item in items),
            }
        )
    return sorted(rows, key=lambda item: item["outputId"])


def scoring_example() -> list[str]:
    return [
        "## 一個 binding 的五欄評分範例",
        "",
        "gold 為 Subject=[United States]、DataName=Revenue—USD、"
        "Position=[{begin:[11,1],end:[11,1]}]、Trend=declined、Num=[3,050]；"
        "prediction 為 Subject=[U.S.]、DataName= revenue-usd、"
        "Position=[{END:[11,1],BEGIN:[11,1]}]、Trend=decrease、Num=[3050]。",
        "",
        "| 報告欄位 | v6 實際欄位 | 判定 | 本例計數 |",
        "|---|---|---|---|",
        "| Subject | ObjectName | U.S. 依凍結 alias 與 United States 一對一命中 | TP |",
        "| Trend | Trend | declined 與 decrease 都是一般方向詞，六類前處理後皆為 none | TP |",
        "| Num | Num | 只讀 Num；去千分位後同為 3050，且無單位衝突 | TP |",
        "| Position | Position | key 大小寫／順序可不同，座標與區間次序相同 | TP |",
        "| DataName | DataName | NFKC、trim、casefold 與等價 dash 後嚴格相等 | TP |",
        "| Text | Text | 不進 TP/FP/FN；須由盲化 LLM judge 評事實一致性與流暢度 | 本批 deferred／NA |",
        "",
        "若已輸出欄位錯誤或型別無效，該欄同時計 FP+FN；欄位缺失只計 FN。"
        "Subject 是論文表名，程式欄位名為 ObjectName。五個結構欄均報 F1（0–1）；"
        "Text 若完成則是獨立 0–100 分，不能與結構欄 F1 直接平均。",
        "",
    ]


def table_lines(rows: Sequence[Mapping[str, Any]], view: str) -> list[str]:
    output = [
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "|" + "|".join("---" if index < 3 else "---:" for index in range(len(REPORT_COLUMNS))) + "|",
    ]
    for row in rows:
        values = [
            row["fineTunedMethod"],
            row["retrieverModel"],
            row["inputPromptType"],
        ]
        values.extend(format_score(score_for(row["case"], view, field)) for _, field in FIELD_MAP)
        values.append("NA (judge deferred)")
        output.append("| " + " | ".join(markdown_escape(value) for value in values) + " |")
    return output


def selection_lines(rows: Sequence[Mapping[str, Any]], view: str) -> list[str]:
    if view == "mean":
        return []
    output = [
        "",
        "## 共同 run 選擇稽核",
        "",
        "| outputId | selected run(s) |",
        "|---|---|",
    ]
    for row in rows:
        selection = row["case"]["selection"]
        selected = (
            str(selection["top1"]["run"])
            if view == "top-1"
            else ", ".join(str(item) for item in selection["top3"]["runs"])
        )
        output.append(f"| {row['outputId']} | {selected} |")
    return output


def in_progress_lines(
    current: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> list[str]:
    output = [
        "",
        "## 目前生成狀態快照",
        "",
        f"- snapshot updatedAt：{report.get('updatedAt')}",
        "- complete 表示該 case 已具有 runs 1–10；部分 runs 不輸出 mean/top-1/top-3。",
        "",
        "| outputId | runs | state | last finished |",
        "|---|---:|---|---|",
    ]
    for item in current:
        output.append(
            f"| {item['outputId']} | {item['runs']}/{item['expectedRuns']} | "
            f"{item['state']} | {item['lastFinishedAt']} |"
        )
    return output


def relationship_summary(
    rows: Sequence[Mapping[str, Any]],
    evaluation_root: Path,
    gold_path: Path,
) -> dict[str, Any]:
    f1_values: list[float] = []
    matched_rates: list[float] = []
    empty_rates: list[float] = []
    by_method: dict[str, list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)
    by_input: dict[str, list[float]] = defaultdict(list)
    field_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        case = row["case"]
        macro_f1 = case["aggregate"]["macro"]["f1"]["mean"]
        if macro_f1 is None:
            continue
        value = float(macro_f1)
        f1_values.append(value)
        gold_bindings = sum(
            int(run["coverage"]["gold_bindings"]) for run in case["runResults"]
        )
        matched = sum(
            int(run["coverage"]["matched_bindings"]) for run in case["runResults"]
        )
        empty_rows = sum(
            int(run["coverage"].get("empty_output_rows", 0)) for run in case["runResults"]
        )
        record_paths = sorted(
            (evaluation_root / "cases" / str(row["outputId"])).glob(
                "run_*/records.jsonl"
            )
        )
        if len(record_paths) != len(case["runResults"]):
            raise TableError(
                f"{row['outputId']} records/run count mismatch: "
                f"{len(record_paths)} != {len(case['runResults'])}"
            )
        row_count = sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in record_paths
        )
        if empty_rows > row_count:
            raise TableError(
                f"{row['outputId']} empty rows exceed evaluated rows: "
                f"{empty_rows} > {row_count}"
            )
        matched_rates.append(matched / gold_bindings if gold_bindings else 0.0)
        empty_rates.append(empty_rows / row_count if row_count else 0.0)
        by_method[str(row["fineTunedMethod"])].append(value)
        by_model[str(row["retrieverModel"])].append(value)
        by_input[str(row["inputPromptType"])].append(value)
        for label, field in FIELD_MAP:
            field_score = case["aggregate"]["fields"][field]["f1"]["mean"]
            if field_score is not None:
                field_values[label].append(float(field_score))

    gold_rows = {
        str(item["source"]): item
        for item in read_json(gold_path).get("rows", [])
    }
    aligned_pairs = 0
    object_mismatches = 0
    object_multi_mention_mismatches = 0
    object_examples: Counter[tuple[str, str]] = Counter()
    trend_methods: Counter[str] = Counter()
    trend_classes: Counter[str] = Counter()
    for row in rows:
        output_id = str(row["outputId"])
        case_root = evaluation_root / "cases" / output_id
        for record_path in sorted(case_root.glob("run_*/records.jsonl")):
            for line in record_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                gold_row = gold_rows.get(str(record.get("source")))
                if gold_row is None:
                    raise TableError(f"gold source missing: {record.get('source')}")
                for detail in record.get("matchDetails", []):
                    aligned_pairs += 1
                    object_result = detail["fields"]["ObjectName"]
                    if not object_result["equal"]:
                        object_mismatches += 1
                        gold_value = gold_row["targetBindings"][int(detail["goldIndex"])]["ObjectName"]
                        prediction_value = object_result.get("predictionNormalized")
                        if (
                            isinstance(gold_value, list)
                            and isinstance(prediction_value, list)
                            and len(gold_value) != len(prediction_value)
                        ):
                            object_multi_mention_mismatches += 1
                        object_examples[
                            (
                                json.dumps(gold_value, ensure_ascii=False),
                                json.dumps(prediction_value, ensure_ascii=False),
                            )
                        ] += 1
                    trend = detail["fields"]["Trend"].get("predictionNormalized") or {}
                    trend_methods[str(trend.get("method") or "missing")] += 1
                    trend_classes[str(trend.get("class") or "invalid")] += 1

    def grouped(values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        return {
            key: {"n": len(items), "macroF1Mean": statistics.fmean(items)}
            for key, items in sorted(values.items())
        }

    macro_summary = {
        "n": len(f1_values),
        "mean": statistics.fmean(f1_values) if f1_values else None,
        "min": min(f1_values) if f1_values else None,
        "max": max(f1_values) if f1_values else None,
    }
    return {
        "cases": len(rows),
        "macroF1AcrossCases": macro_summary,
        "correlations": {
            "matchedBindingRateVsMacroF1": safe_correlation(matched_rates, f1_values),
            "emptyRowRateVsMacroF1": safe_correlation(empty_rates, f1_values),
        },
        "byFineTunedMethod": grouped(by_method),
        "byRetrieverModel": grouped(by_model),
        "byInputPromptType": grouped(by_input),
        "fieldF1AcrossCases": {
            key: {"n": len(values), "mean": statistics.fmean(values)}
            for key, values in field_values.items()
        },
        "objectNameDiagnostics": {
            "alignedPairs": aligned_pairs,
            "mismatches": object_mismatches,
            "mismatchRate": object_mismatches / aligned_pairs if aligned_pairs else None,
            "multiMentionCountMismatch": object_multi_mention_mismatches,
            "testDerivedExamplesNotEligibleForAliasExpansion": [
                {"gold": gold, "prediction": prediction, "count": count}
                for (gold, prediction), count in object_examples.most_common(10)
            ],
        },
        "trendDiagnostics": {
            "alignedPredictionMethods": dict(trend_methods),
            "alignedPredictionClasses": dict(trend_classes),
        },
    }


def render(
    view: str,
    rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    generation: Mapping[str, Any],
    current: Sequence[Mapping[str, Any]],
    current_report: Mapping[str, Any],
) -> str:
    generation_finished = generation_finished_at(rows)
    protocol = str(evaluation.get("protocol"))
    protocol_version = protocol.rsplit("-", 1)[-1]
    lines = [
        f"# Experiment 6 {protocol_version} 欄位 F1：{VIEW_LABELS[view]}",
        "",
        f"- 評估時間：{evaluation.get('time')}。",
        f"- prediction 最晚完成時間：{generation_finished}。",
        f"- 正式範圍：{len(rows)} cases × 10 runs；來源 generation protocol：{generation.get('protocol')}。",
        f"- 評估 protocol：{evaluation.get('protocol')}；method SHA-256：{evaluation['method']['methodSha256']}。",
        "- 分數定義：Subject／Trend／Num／Position／DataName 均為 field-level F1（0–1）。",
        "- Text 為獨立 0–100 LLM judge 指標；ChatMock 未用於本批，故一律 NA，不進結構欄 F1。",
        "- 表內 top-1/top-3 均使用每 case 的同一組 shared runs，未按欄位另選最佳 run。",
        "",
    ]
    lines.extend(scoring_example())
    lines.extend(
        [
            f"## 最終分數：{VIEW_LABELS[view]}",
            "",
            *table_lines(rows, view),
        ]
    )
    lines.extend(selection_lines(rows, view))
    lines.extend(in_progress_lines(current, current_report))
    lines.extend(
        [
            "",
            "## 可解讀範圍",
            "",
            "- 原定 38 cases 中，4 個 GPT-4.1 cases 沒有完成 predictions，因此未列入正式表。",
            "- 若來源 protocol 為 evaluation overlay，本表以 corrected12 覆蓋同名歷史 cases；未被覆蓋者仍是歷史生成。",
            "- completion 與 freshness 以各 manifest 的 finishedAt 為準，不由報告檔名推定。",
            "- 現有 gold 的五種特殊 Trend 圖形 support 均為 0；Trend 分數不能解讀為五類圖形辨識能力。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--current-generation-report", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = read_json(args.evaluation_report)
    generation = read_json(args.generation_report)
    current_report = read_json(args.current_generation_report)
    registry = read_json(args.source_registry)
    config = read_json(args.config)
    expected_runs = int(config["expectedRuns"])
    metadata = build_metadata(
        generation,
        registry["sources"],
        expected_runs,
    )
    cases = validate_evaluation(
        evaluation,
        metadata,
        expected_runs,
        str(config["protocolId"]),
    )
    by_id = {str(case["outputId"]): case for case in cases}
    rows = [
        {**values, "case": by_id[output_id]}
        for output_id, values in metadata.items()
    ]
    rows.sort(key=row_sort_key)
    report_keys = [
        (
            str(row["fineTunedMethod"]),
            str(row["retrieverModel"]),
            str(row["inputPromptType"]),
        )
        for row in rows
    ]
    if len(report_keys) != len(set(report_keys)):
        duplicates = sorted(
            key for key, count in Counter(report_keys).items() if count > 1
        )
        raise TableError(f"duplicate report metadata keys: {duplicates}")
    current = current_status(current_report, expected_runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for view in ("mean", "top-1", "top-3"):
        path = args.output_dir / f"experiment_6_v6_欄位分數_{view}.md"
        path.write_text(
            render(view, rows, evaluation, generation, current, current_report),
            encoding="utf-8",
        )
        outputs[view] = str(path.resolve())
    summary = relationship_summary(
        rows,
        args.evaluation_root.resolve(),
        (
            args.config.resolve().parent.parent / str(config["goldPath"])
        ).resolve(),
    )
    summary_path = args.output_dir / "experiment_6_v6_relationship_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "protocol": evaluation["protocol"],
                "outputs": outputs,
                "relationshipSummaryPath": str(summary_path.resolve()),
                "relationshipSummary": summary,
                "currentGeneration": {
                    "updatedAt": current_report.get("updatedAt"),
                    "complete": current_report.get("complete"),
                    "caseRuns": {item["outputId"]: item["runs"] for item in current},
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
