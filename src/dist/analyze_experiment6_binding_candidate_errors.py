#!/usr/bin/env python3
"""Audit Experiment 6 Binding candidates against frozen gold without rescoring.

The formal v6 score remains the hard-anchor end-to-end score.  This tool reads
the aggregate ``bindings.jsonl`` directly and emits claim-ineligible diagnostic
views that explain whether failures come from coverage, binding identity, or a
particular field.  It never edits candidates, gold, or evaluator settings.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DIST = Path(__file__).resolve().parent
REPO_ROOT = DIST.parent
WORKSPACE_ROOT = REPO_ROOT.parent

import evaluate_narrative2_reference_aligned_v6_0_2 as scorer
from validate_experiment6_binding_candidates import validate_output as validate_candidate_v1
from validate_experiment6_bindings_v2 import validate_output as validate_candidate_v2
from materialize_experiment6_bindings_relaxed_v3 import validate_output as validate_candidate_v3
from materialize_experiment6_bindings_repaired_v4 import validate_output as validate_candidate_v4


PROTOCOL = "experiment6-binding-candidate-error-analysis-v1"
CANDIDATE_VALIDATORS = {
    "experiment6-binding-candidate-materialization-v1": validate_candidate_v1,
    "experiment6-binding-materialization-v2-unified34": validate_candidate_v2,
    "experiment6-binding-materialization-relaxed-v3-unified34": validate_candidate_v3,
    "experiment6-binding-materialization-repaired-v4-unified34": validate_candidate_v4,
}
NON_TREND_FIELDS = tuple(field for field in scorer.PRIMARY_FIELDS if field != "Trend")
CORE_FIELDS = (*scorer.PRIMARY_FIELDS, "Text")


class AnalysisError(RuntimeError):
    """Raised when an audit invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return resolved.name


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["outputId"]), int(row["run"]), str(row["source"])


def binding_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (*row_key(row), int(row["bindingIndex"]))


def core_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in CORE_FIELDS}


def field_equal(
    field: str,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    objects: scorer.ObjectMatcher,
    trends: scorer.TrendClassifier,
) -> bool:
    return bool(scorer.field_comparison(field, gold, prediction, objects, trends)["equal"])


def maximum_weight_pairs(equal_vectors: Sequence[Sequence[Sequence[bool]]]) -> list[tuple[int, int]]:
    """Pair bindings for diagnosis, excluding Trend-only coincidences.

    The score maximizes equal non-Trend fields.  Trend is only a tie-break, so
    the all-``none`` gold support cannot manufacture a near match.
    """

    gold_count = len(equal_vectors)
    prediction_count = len(equal_vectors[0]) if gold_count else 0

    @functools.lru_cache(maxsize=None)
    def visit(gold_index: int, used_predictions: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if gold_index == gold_count:
            return 0, ()
        best = visit(gold_index + 1, used_predictions)
        for prediction_index in range(prediction_count):
            if used_predictions & (1 << prediction_index):
                continue
            vector = equal_vectors[gold_index][prediction_index]
            non_trend_equal = sum(int(vector[index]) for index, field in enumerate(scorer.PRIMARY_FIELDS) if field != "Trend")
            if non_trend_equal == 0:
                continue
            weight = non_trend_equal * 100 + int(vector[scorer.PRIMARY_FIELDS.index("Trend")])
            suffix_score, suffix_pairs = visit(gold_index + 1, used_predictions | (1 << prediction_index))
            candidate = (weight + suffix_score, ((gold_index, prediction_index), *suffix_pairs))
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
        return best

    return list(visit(0, 0)[1])


def empty_counts() -> dict[str, dict[str, int]]:
    return scorer.zero_counts()


def update_official_counts(
    counts: dict[str, dict[str, int]],
    gold: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
    objects: scorer.ObjectMatcher,
    trends: scorer.TrendClassifier,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alignment = scorer.align_bindings(gold, prediction)
    details: list[dict[str, Any]] = []
    for pair in alignment["matches"]:
        gold_binding = gold[pair["goldIndex"]]
        predicted_binding = prediction[pair["predictionIndex"]]
        equal_fields = []
        for field in scorer.PRIMARY_FIELDS:
            if field_equal(field, gold_binding, predicted_binding, objects, trends):
                counts[field]["tp"] += 1
                equal_fields.append(field)
            else:
                counts[field]["fp"] += 1
                counts[field]["fn"] += 1
        details.append({**pair, "equalFields": equal_fields})
    for _ in alignment["unmatchedGold"]:
        for field in scorer.PRIMARY_FIELDS:
            counts[field]["fn"] += 1
    for prediction_index in alignment["unmatchedPrediction"]:
        predicted_binding = prediction[prediction_index]
        for field in scorer.PRIMARY_FIELDS:
            if field in predicted_binding:
                counts[field]["fp"] += 1
    return alignment, details


def update_inventory_counts(
    counts: dict[str, dict[str, int]],
    gold: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
    objects: scorer.ObjectMatcher,
    trends: scorer.TrendClassifier,
) -> None:
    for field in scorer.PRIMARY_FIELDS:
        pairs = scorer.maximum_matching(
            len(gold),
            len(prediction),
            lambda gold_index, prediction_index: field_equal(
                field, gold[gold_index], prediction[prediction_index], objects, trends
            ),
        )
        true_positives = len(pairs)
        counts[field]["tp"] += true_positives
        counts[field]["fp"] += len(prediction) - true_positives
        counts[field]["fn"] += len(gold) - true_positives


def classify_pair(equal_fields: Sequence[str]) -> str:
    equal = set(equal_fields)
    if equal == set(scorer.PRIMARY_FIELDS):
        return "all_five_fields_equal"
    if {"DataName", "Position"} <= equal:
        return "hard_anchor_equal_other_field_error"
    if "DataName" in equal or "Position" in equal:
        return "partial_anchor_near_miss"
    if "ObjectName" in equal or "Num" in equal:
        return "value_only_near_miss"
    return "unrelated"


def compact_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {field: binding.get(field) for field in CORE_FIELDS}


def append_example(examples: dict[str, list[dict[str, Any]]], category: str, value: dict[str, Any], limit: int) -> None:
    if len(examples[category]) < limit:
        examples[category].append(value)


def metrics_for_json(counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    return scorer.metrics_from_counts(counts)


def annotation_case_means(evaluation_report: Mapping[str, Any]) -> dict[str, Any]:
    """Average each annotation metric over cases with a defined case mean."""
    output: dict[str, Any] = {}
    cases = list(evaluation_report.get("cases") or [])
    for field in (*scorer.PRIMARY_FIELDS, "Text"):
        if field == "Text":
            output[field] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "definedCases": 0,
                "totalCases": len(cases),
                "status": "NA",
            }
            continue
        result: dict[str, Any] = {"totalCases": len(cases), "status": "defined"}
        defined_counts = []
        for metric in ("precision", "recall", "f1"):
            values = [
                case["aggregate"]["fields"][field][metric]["mean"]
                for case in cases
                if case["aggregate"]["fields"][field][metric]["mean"] is not None
            ]
            result[metric] = sum(values) / len(values) if values else None
            result[f"{metric}DefinedCases"] = len(values)
            defined_counts.append(len(values))
        result["definedCases"] = min(defined_counts)
        output[field] = result
    return output


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    candidate_root = args.candidate_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_report = read_json(args.evaluation_report.resolve())
    merged_report = read_json(args.merged_report.resolve()) if args.merged_report else None
    dataset = read_json(candidate_root / "dataset_manifest.json")
    candidate_protocol = str(dataset.get("protocol") or "")
    require(candidate_protocol in CANDIDATE_VALIDATORS, f"unsupported candidate protocol: {candidate_protocol}")
    validation = CANDIDATE_VALIDATORS[candidate_protocol](candidate_root)
    require(validation["status"] == "valid", "candidate validator did not return valid")
    require(evaluation_report.get("scoringProtocol") == scorer.PROTOCOL, "expected v6.0.2 evaluation report")
    require(evaluation_report.get("scope") in {"candidate12", "candidate34"}, "expected candidate12/candidate34 evaluation report")
    require(evaluation_report.get("claimEligible") is False, "candidate evaluation unexpectedly claim-eligible")

    config = read_json(args.config.resolve())
    scorer.validate_config(config)
    gold_path = (REPO_ROOT / config["goldPath"]).resolve()
    require(sha256_file(gold_path) == config["goldSha256"], "gold SHA mismatch")
    gold_rows = read_json(gold_path)["rows"]
    gold_by_source = {str(row["source"]): list(row["targetBindings"]) for row in gold_rows}

    aggregate_binding_path = candidate_root / "bindings.jsonl"
    aggregate_row_path = candidate_root / "rows.jsonl"
    aggregate_bindings = read_jsonl(aggregate_binding_path)
    aggregate_rows = read_jsonl(aggregate_row_path)
    require(all(row.get("protocol") == candidate_protocol for row in aggregate_bindings), "aggregate binding protocol mismatch")

    per_run_bindings: list[dict[str, Any]] = []
    for path in sorted((candidate_root / "cases").glob("*/run_*/bindings.jsonl")):
        per_run_bindings.extend(read_jsonl(path))
    aggregate_by_id = {str(row["candidateId"]): row for row in aggregate_bindings}
    per_run_by_id = {str(row["candidateId"]): row for row in per_run_bindings}
    require(len(aggregate_by_id) == len(aggregate_bindings), "duplicate candidateId in aggregate bindings")
    require(aggregate_by_id == per_run_by_id, "aggregate bindings differ from per-run bindings")

    predictions_by_row: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for binding in aggregate_bindings:
        predictions_by_row[row_key(binding)].append(binding)
    for values in predictions_by_row.values():
        values.sort(key=lambda row: int(row["bindingIndex"]))

    for row in aggregate_rows:
        key = row_key(row)
        direct = [core_binding(value) for value in predictions_by_row.get(key, [])]
        require(direct == row["Binding"], f"aggregate binding payload mismatch: {key}")
        require(len(direct) == int(row["bindingCount"]), f"aggregate binding count mismatch: {key}")

    objects = scorer.ObjectMatcher(config["objectName"])
    trends = scorer.TrendClassifier(config["trend"], allow_model=True)
    official_counts = empty_counts()
    inventory_counts = empty_counts()
    inventory_by_case: dict[str, dict[str, dict[str, int]]] = defaultdict(empty_counts)
    conditional = {field: Counter(total=0, equal=0) for field in scorer.PRIMARY_FIELDS}
    category_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    text_pairs_eligible = 0
    num_true_positive_support = Counter()

    for row in aggregate_rows:
        output_id, run, source = row_key(row)
        gold = gold_by_source[source]
        prediction = [core_binding(value) for value in predictions_by_row.get(row_key(row), [])]
        alignment, official_details = update_official_counts(official_counts, gold, prediction, objects, trends)
        update_inventory_counts(inventory_counts, gold, prediction, objects, trends)
        update_inventory_counts(inventory_by_case[output_id], gold, prediction, objects, trends)
        for detail in official_details:
            gold_binding = gold[detail["goldIndex"]]
            predicted_binding = prediction[detail["predictionIndex"]]
            for field in scorer.PRIMARY_FIELDS:
                conditional[field]["total"] += 1
                conditional[field]["equal"] += int(field in detail["equalFields"])
            if isinstance(gold_binding.get("Text"), str) and isinstance(predicted_binding.get("Text"), str):
                text_pairs_eligible += 1
            if "Num" in detail["equalFields"]:
                num_true_positive_support["empty" if gold_binding.get("Num") == [] else "nonempty"] += 1

        equal_vectors = [
            [
                [field_equal(field, gold_binding, predicted_binding, objects, trends) for field in scorer.PRIMARY_FIELDS]
                for predicted_binding in prediction
            ]
            for gold_binding in gold
        ]
        diagnostic_pairs = maximum_weight_pairs(equal_vectors) if gold and prediction else []
        paired_gold = {gold_index for gold_index, _ in diagnostic_pairs}
        paired_prediction = {prediction_index for _, prediction_index in diagnostic_pairs}
        for gold_index, prediction_index in diagnostic_pairs:
            equal_fields = [
                field for field, equal in zip(scorer.PRIMARY_FIELDS, equal_vectors[gold_index][prediction_index]) if equal
            ]
            category = classify_pair(equal_fields)
            category_counts[category] += 1
            append_example(examples, category, {
                "outputId": output_id,
                "run": run,
                "source": source,
                "goldIndex": gold_index,
                "predictionIndex": prediction_index,
                "equalFields": equal_fields,
                "gold": compact_binding(gold[gold_index]),
                "prediction": compact_binding(prediction[prediction_index]),
            }, args.examples_per_category)
        category_counts["diagnostic_unmatched_gold"] += len(gold) - len(paired_gold)
        category_counts["diagnostic_unmatched_prediction"] += len(prediction) - len(paired_prediction)
        for prediction_index in range(len(prediction)):
            if prediction_index in paired_prediction:
                continue
            append_example(examples, "unmatched_prediction", {
                "outputId": output_id,
                "run": run,
                "source": source,
                "predictionIndex": prediction_index,
                "gold": [compact_binding(item) for item in gold],
                "prediction": compact_binding(prediction[prediction_index]),
            }, args.examples_per_category)
        if gold and not prediction:
            append_example(examples, "empty_prediction", {
                "outputId": output_id,
                "run": run,
                "source": source,
                "candidateStatus": row.get("candidateStatus"),
                "schemaValid": row.get("schemaValid"),
                "gold": [compact_binding(item) for item in gold],
                "prediction": [],
            }, args.examples_per_category)

    expected_counts = evaluation_report["overall"]["primaryPooled"]["counts"]
    require(official_counts == expected_counts, "direct aggregate bindings do not reproduce official field counts")

    total_gold_bindings = sum(len(gold_by_source[str(row["source"])]) for row in aggregate_rows)
    total_prediction_bindings = len(aggregate_bindings)
    maximum_true_positives = min(total_gold_bindings, total_prediction_bindings)
    count_ceiling = scorer.metric(
        tp=maximum_true_positives,
        fp=total_prediction_bindings - maximum_true_positives,
        fn=total_gold_bindings - maximum_true_positives,
    )
    conditional_metrics = {
        field: {
            "eligibleHardAnchorPairs": values["total"],
            "correct": values["equal"],
            "accuracy": values["equal"] / values["total"] if values["total"] else None,
        }
        for field, values in conditional.items()
    }
    gpt_cases: list[dict[str, Any]] = []
    if merged_report is not None:
        metadata = {str(item["outputId"]): item for item in merged_report.get("caseMetadata", [])}
        for case in merged_report.get("cases", []):
            output_id = str(case["outputId"])
            source_id = str(metadata.get(output_id, {}).get("sourceId") or "")
            if "gpt5.5" in output_id.casefold() or source_id == "gpt-5.5":
                gpt_cases.append({
                    "outputId": output_id,
                    "inputOrigin": case.get("inputOrigin"),
                    "sourceId": source_id,
                    "macroF1Mean": case["aggregate"]["macro"]["f1"]["mean"],
                })

    report = {
        "schemaVersion": 1,
        "protocol": PROTOCOL,
        "status": "diagnostic_only",
        "official": False,
        "claimEligible": False,
        "selectionRole": "error-analysis-only",
        "time": scorer.utc_now(),
        "input": {
            "candidateRoot": logical_path(candidate_root),
            "aggregateBindingsPath": logical_path(aggregate_binding_path),
            "aggregateBindingsSha256": sha256_file(aggregate_binding_path),
            "aggregateRowsPath": logical_path(aggregate_row_path),
            "aggregateRowsSha256": sha256_file(aggregate_row_path),
            "evaluationReportPath": logical_path(args.evaluation_report.resolve()),
            "evaluationReportSha256": sha256_file(args.evaluation_report.resolve()),
            "goldPath": logical_path(gold_path),
            "goldSha256": sha256_file(gold_path),
        },
        "directAggregateAudit": {
            "aggregateBindingCount": len(aggregate_bindings),
            "perRunBindingCount": len(per_run_bindings),
            "candidateIdsAndRecordsEqual": True,
            "rowEmbeddedPayloadsEqual": True,
            "officialFieldCountsReproduced": True,
        },
        "coverage": {
            **validation["counts"],
            "goldBindingsAcrossCaseRuns": total_gold_bindings,
            "predictionBindings": total_prediction_bindings,
            "maximumRecallFromBindingCount": count_ceiling["recall"],
            "maximumF1FromBindingCount": count_ceiling["f1"],
            "observedHardAnchorMatches": evaluation_report["coverage"]["matched_bindings"],
            "observedHardAnchorMatchRateOverGold": evaluation_report["coverage"]["anchorMatchRate"],
            "observedHardAnchorMatchRateOverPredictions": (
                evaluation_report["coverage"]["matched_bindings"] / total_prediction_bindings
                if total_prediction_bindings else None
            ),
        },
        "formalEndToEnd": {
            "caseMeanMacroF1": evaluation_report["overall"]["caseMeanMacroF1"],
            "fieldMetrics": scorer.metrics_from_counts(official_counts),
            "bindingMetrics": evaluation_report["overall"]["bindingPooled"],
            "interpretation": "Hard-anchor end-to-end extraction; not standalone field accuracy.",
        },
        "annotationCaseMeans": annotation_case_means(evaluation_report),
        "hardAnchorConditionalAccuracy": conditional_metrics,
        "fieldSupport": {
            "num": {
                "goldEmpty": sum(
                    int(binding.get("Num") == [])
                    for row in aggregate_rows
                    for binding in gold_by_source[str(row["source"])]
                ),
                "goldNonempty": sum(
                    int(binding.get("Num") != [])
                    for row in aggregate_rows
                    for binding in gold_by_source[str(row["source"])]
                ),
                "predictionEmpty": sum(int(binding.get("Num") == []) for binding in aggregate_bindings),
                "predictionNonempty": sum(int(binding.get("Num") != []) for binding in aggregate_bindings),
                "hardAnchorTruePositiveEmpty": num_true_positive_support["empty"],
                "hardAnchorTruePositiveNonempty": num_true_positive_support["nonempty"],
            },
            "trendGoldSupport": evaluation_report["trend"]["support"],
        },
        "rowLevelFieldInventoryDiagnostic": {
            "definition": "Per source row, each field is matched independently as a multiset; binding associations are ignored.",
            "claimEligible": False,
            "metrics": metrics_for_json(inventory_counts),
            "byCase": {
                output_id: metrics_for_json(counts)
                for output_id, counts in sorted(inventory_by_case.items())
            },
        },
        "diagnosticPairCategories": dict(sorted(category_counts.items())),
        "text": {
            "status": "NA",
            "tpFpFnDefined": False,
            "judgeUsed": False,
            "eligibleHardAnchorPairs": text_pairs_eligible,
            "reason": "Open-ended Text quality is a separate 0-100 secondary judgment and was not run; it is not a structural TP/FP/FN field.",
        },
        "gpt55Context": {
            "candidateFileContainsGpt55": any("gpt5.5" in str(row["outputId"]).casefold() for row in aggregate_bindings),
            "mergedHistoricalCases": gpt_cases,
            "interpretation": "GPT-5.5 rows in merged34 come from historical22, not from the candidate bindings.jsonl.",
        },
        "materializationQuality": {
            "candidateProtocol": candidate_protocol,
            "gibberishRows": validation["counts"].get("gibberishRows"),
            "gibberishBindings": validation["counts"].get("gibberishBindings"),
            "duplicateBindingsWithinRow": validation["counts"].get("duplicateBindingsWithinRow"),
            "previouslyUnrecoverableRows": validation["counts"].get("previouslyUnrecoverableRows"),
        },
        "examples": dict(examples),
        "methodReview": {
            "retain": [
                "Keep v6.0.2 hard-anchor end-to-end scoring as the frozen primary tuple-extraction result.",
                "Keep wrong present fields as FP+FN, missing fields as FN, and unpaired predictions as FP.",
                "Keep Text outside structural TP/FP/FN and preserve all raw strings.",
            ],
            "changeReporting": [
                "Rename current field F1 to anchored end-to-end field F1; DataName and Position are anchors and therefore not independent field tests.",
                "Always report schema-valid coverage, binding-count ceiling, hard-anchor coverage, and conditional accuracy beside F1.",
                "Report row-level field-inventory F1 only as a diagnostic because it ignores cross-field binding association.",
            ],
            "prospectiveOnly": [
                "If a softer binding identity is needed, freeze it on an independent calibration set under a new protocol before inspecting test gains.",
                "Do not expand aliases or tune Sentence-BERT on these test predictions; neither addresses missing outputs or wrong chart coordinates.",
                "For Text, pair without using Text, blind model/case identity, report raw 0-100 dimensions and judge coverage, and never mix them into structural F1.",
            ],
        },
        "limitations": [
            "Candidate materialization is diagnosticOnly=true and claimEligible=false.",
            "Row-level field inventory can score values that belong to the wrong binding and must not replace end-to-end results.",
            "All current Trend gold labels map to none; five special-pattern classes have zero positive support.",
            "Text remains unevaluated and GPT-5.5 merged scores are structural generation scores, not Text-judge scores.",
        ],
    }
    write_json(output_dir / "binding_error_analysis.json", report)
    write_markdown(output_dir / "binding_error_analysis.md", report)
    return report


def format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    coverage = report["coverage"]
    formal_fields = report["formalEndToEnd"]["fieldMetrics"]["fields"]
    inventory_fields = report["rowLevelFieldInventoryDiagnostic"]["metrics"]["fields"]
    conditional = report["hardAnchorConditionalAccuracy"]
    lines = [
        "# Experiment 6 Binding candidate error analysis",
        "",
        "> DIAGNOSTIC ONLY — OFFICIAL=false — CLAIM-ELIGIBLE=false",
        "",
        "## Direct input audit",
        "",
        f"- Direct aggregate file: `{report['input']['aggregateBindingsPath']}` (`{report['input']['aggregateBindingsSha256']}`).",
        f"- Its {report['directAggregateAudit']['aggregateBindingCount']} records exactly equal the union of all per-run binding records.",
        "- Recomputing from the aggregate file reproduces the evaluator's field TP/FP/FN exactly.",
        "",
        "## Why the score is low",
        "",
        f"- Rows: {coverage['rows']}; schema-valid: {coverage['acceptedRows']}; rejected: {coverage['rejectedRows']}.",
        f"- Candidate bindings: {coverage['predictionBindings']}; gold bindings over {coverage['caseRuns']} case-runs: {coverage['goldBindingsAcrossCaseRuns']}.",
        f"- Even if every candidate binding were correct, count alone caps pooled binding recall at {format_value(coverage['maximumRecallFromBindingCount'])} and pooled binding F1 at {format_value(coverage['maximumF1FromBindingCount'])}.",
        f"- Hard-anchor matches: {coverage['observedHardAnchorMatches']} ({format_value(coverage['observedHardAnchorMatchRateOverPredictions'])} of predictions).",
        f"- Reproduced five-field case-mean macro-F1: {format_value(report['formalEndToEnd']['caseMeanMacroF1'])}.",
        "",
        "## Mean Precision / Recall / F1 by annotation",
        "",
        "Each value is the unweighted mean of the case-level ten-run mean. Undefined precision cases are omitted and their count is reported. Text is NA because this structural evaluator does not invoke a Text judge.",
        "",
        "| Annotation | Mean Precision | Mean Recall | Mean F1 | Precision-defined cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, values in report["annotationCaseMeans"].items():
        lines.append(
            f"| {field} | {format_value(values['precision'])} | {format_value(values['recall'])} | {format_value(values['f1'])} | {values.get('precisionDefinedCases', 0)}/{values['totalCases']} |"
        )
    lines.extend([
        "",
        "## What each field number means",
        "",
        "The formal column is the frozen end-to-end score after `DataName+Position` hard alignment. The inventory column asks only whether an equal field value exists somewhere in the same source row; it ignores binding association and is diagnostic only. Conditional accuracy is limited to exact hard-anchor pairs.",
        "",
        "| Field | Formal end-to-end TP/FP/FN | Formal F1 | Row inventory F1 | Hard-anchor conditional accuracy |",
        "|---|---:|---:|---:|---:|",
    ])
    for field in scorer.PRIMARY_FIELDS:
        formal = formal_fields[field]
        inventory = inventory_fields[field]
        cond = conditional[field]
        lines.append(
            f"| {field} | {formal['tp']}/{formal['fp']}/{formal['fn']} | {format_value(formal['f1'])} | {format_value(inventory['f1'])} | {format_value(cond['accuracy'])} ({cond['correct']}/{cond['eligibleHardAnchorPairs']}) |"
        )
    lines.extend([
        "",
        "`DataName` and `Position` conditional accuracy is necessarily 1.0 because they define the hard anchor. Their current F1 values are therefore end-to-end extraction scores, not independent tests of those fields.",
        "",
        f"Num support check: {report['fieldSupport']['num']['hardAnchorTruePositiveNonempty']} of {report['fieldSupport']['num']['hardAnchorTruePositiveNonempty'] + report['fieldSupport']['num']['hardAnchorTruePositiveEmpty']} Num TP contain actual numbers and {report['fieldSupport']['num']['hardAnchorTruePositiveEmpty']} are correct empty arrays. Trend support is entirely `none`; its apparent accuracy cannot establish five-pattern classification ability.",
        "",
        "## Text",
        "",
        f"Text is `NA`, has no TP/FP/FN, and no judge was called. There are {report['text']['eligibleHardAnchorPairs']} hard-anchor pairs eligible for a later blinded secondary 0–100 judgment. GPT-5.5's high merged34 rows are structural generation scores from historical inputs, not Text-judge scores.",
        "",
        "## Diagnostic pair categories",
        "",
    ])
    for category, count in report["diagnosticPairCategories"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Representative gold comparisons", ""])
    for category, examples in report["examples"].items():
        if not examples:
            continue
        lines.append(f"### {category}")
        lines.append("")
        for example in examples:
            identity = f"{example['outputId']} / run {example['run']} / {example['source']}"
            lines.append(f"- **{identity}**")
            if "equalFields" in example:
                lines.append(f"  - Equal fields: {', '.join(example['equalFields']) or 'none'}")
            lines.append(f"  - Gold: `{json.dumps(example['gold'], ensure_ascii=False, sort_keys=True)}`")
            lines.append(f"  - Prediction: `{json.dumps(example['prediction'], ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    lines.extend([
        "## Method changes",
        "",
        "- Keep v6.0.2 hard-anchor end-to-end scoring as the frozen primary result, but label its field columns as anchored end-to-end field F1.",
        "- Always place format coverage, binding-count ceiling, hard-anchor coverage, and conditional field accuracy beside the primary score.",
        "- Keep row-inventory metrics diagnostic; they deliberately ignore whether fields belong to the same binding.",
        "- Do not tune aliases or Sentence-BERT on these test predictions. Missing outputs and wrong chart coordinates are the dominant failures.",
        "- Any softer identity rule must be calibrated independently and released under a new protocol/hash.",
        "",
        "## Reproducibility",
        "",
        f"- Aggregate bindings SHA-256: `{report['input']['aggregateBindingsSha256']}`.",
        f"- Evaluation report SHA-256: `{report['input']['evaluationReportSha256']}`.",
        f"- Gold SHA-256: `{report['input']['goldSha256']}`.",
        "- The aggregate binding records reproduce the official field counts exactly.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--merged-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "experiment6_narrative2_evaluation_v6_0_2.json",
    )
    parser.add_argument("--examples-per-category", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except (AnalysisError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, scorer.ProtocolError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "status": report["status"],
        "aggregateBindings": report["directAggregateAudit"]["aggregateBindingCount"],
        "caseMeanMacroF1": report["formalEndToEnd"]["caseMeanMacroF1"],
        "hardAnchorMatches": report["coverage"]["observedHardAnchorMatches"],
        "annotationCaseMeans": report["annotationCaseMeans"],
        "output": logical_path(args.output_dir.resolve() / "binding_error_analysis.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
