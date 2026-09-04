#!/usr/bin/env python3
"""Reference-aligned hybrid-v5.1 diagnostic evaluator for Experiment 6."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import evaluate_narrative2_hybrid_v4_no_gpt41 as v4


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
FIELDS = v4.FIELDS
CONTENT_FIELDS = ("ObjectName", "Trend", "Num", "Text")
PROTOCOL = "narrative2-reference-aligned-hybrid-v5.1"
MISTRAL_CHAT_PROJECTION_PROTOCOL = "experiment6-mistral-chat-repaired-projection-v1"
METHOD_REVISION = "evidence-gated-strict-json-schema-percent-dual-20260812"
PRIMARY_THRESHOLD = 0.8
THRESHOLDS = (0.7, 0.8, 0.9)
ABLATIONS = (
    "hard_anchor_local_exact",
    "deterministic_normalized",
    "semantic_gpt55_medium",
)
TREND_CLASS_ALIASES = {
    "expanded": "increase",
    "inched up": "increase",
    "increase": "increase",
    "increased": "increase",
    "increases": "increase",
    "rise": "increase",
    "rises": "increase",
    "rose": "increase",
    "up": "increase",
    "upgrade trend": "increase",
    "uptrend": "increase",
    "decline": "decrease",
    "declined": "decrease",
    "declines": "decrease",
    "decrease": "decrease",
    "decreased": "decrease",
    "down": "decrease",
    "downtrend": "decrease",
    "lower": "decrease",
    "plunge": "decrease",
    "plunged": "decrease",
    "plunges": "decrease",
    "slipped": "decrease",
    "reversal": "reversal",
    "reversing": "reversal",
    "reversed": "reversal",
    "trend reversal": "reversal",
    "trough": "trough",
    "double bottom": "double_bottom",
    "double-bottom": "double_bottom",
    "triple top": "triple_top",
    "triple-top": "triple_top",
    "head and shoulders": "head_and_shoulders",
    "head-and-shoulders": "head_and_shoulders",
}
TREND_ONTOLOGY_VERSION = "trend-class-aliases-v1"
TREND_ONTOLOGY_SHA256 = v4.sha256_text(json.dumps(
    TREND_CLASS_ALIASES,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
))
UNIT_FACTORS = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}
FINANCE_TOKEN_RE = re.compile(
    r"(?:19|20)\d{2}(?:[-/]\d{1,2}){1,2}"
    r"|(?:[$€£¥]\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    r"|[^\W\d_]+",
    re.UNICODE,
)


class ProtocolError(RuntimeError):
    """Raised when diagnostic artifacts violate the v5 contract."""


def method_metadata() -> dict[str, Any]:
    material = {
        "protocol": PROTOCOL,
        "revision": METHOD_REVISION,
        "anchor": "DataName trim-lower plus typed Position exact",
        "alignment": "within Source, one-to-one, first duplicate consumes anchor",
        "localPenalty": True,
        "objectName": (
            "NFKC/case/whitespace exact, then same-entity/coreference judge; "
            "DataName substitution forbidden"
        ),
        "trend": "versioned deterministic class mapping, then contextual judge",
        "trendOntologyVersion": TREND_ONTOLOGY_VERSION,
        "trendOntologySha256": TREND_ONTOLOGY_SHA256,
        "text": (
            "normalized exact, then full-proposition judge covering subject, trend, "
            "number, time, scope, baseline, and negation"
        ),
        "semanticEvidenceGate": (
            "non-empty verbatim evidenceSpan in Source; evidenceValid=true; "
            "no validationError"
        ),
        "primaryThreshold": PRIMARY_THRESHOLD,
        "thresholdSensitivity": list(THRESHOLDS),
        "numericPrimary": "finite JSON number array, one-to-one isclose(1e-9)",
        "percentageSensitivityConventions": [
            "ratio: 1%=0.01",
            "percentage-point: 1%=1",
        ],
        "primaryScoreRole": "diagnostic_only_not_formal_ranking",
        "abSwapAuditUnit": "Source row containing one or more semantic decisions",
        "thirdAdjudicationRole": "audit_only_not_score_mutation",
        "numericSensitivityRankingRole": "diagnostic_only",
        "tokenF1RankingRole": "diagnostic_only",
        "conditionalContentRankingRole": "diagnostic_only",
    }
    compatibility_sha = v4.sha256_text(json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return {
        **material,
        "methodCompatibilitySha256": compatibility_sha,
        "evaluatorSha256": v4.sha256_file(Path(__file__).resolve()),
    }


def validate_evaluation_contract(evaluation: Mapping[str, Any]) -> None:
    errors: list[str] = []
    exact = {
        "fields": list(FIELDS),
        "hardFields": ["DataName", "Position"],
        "semanticFields": ["ObjectName", "Trend", "Text"],
        "numericField": "Num",
        "failurePolicy": "rejected-zero",
    }
    for name, expected in exact.items():
        if evaluation.get(name) != expected:
            errors.append(f"{name}={evaluation.get(name)!r} expected={expected!r}")
    judge = evaluation.get("judge")
    if not isinstance(judge, Mapping):
        errors.append("judge config missing")
    else:
        judge_expected = {
            "model": "gpt-5.5",
            "reasoningEffort": "medium",
            "minimumConfidence": PRIMARY_THRESHOLD,
            "requireEvidenceSpans": True,
            "blindCaseAndModel": True,
            "randomizeAB": True,
            "requestTimeoutSeconds": 300,
            "maxAttempts": 3,
            "retryDelaysSeconds": [5, 15],
        }
        for name, expected in judge_expected.items():
            if judge.get(name) != expected:
                errors.append(
                    f"judge.{name}={judge.get(name)!r} expected={expected!r}"
                )
    audit = evaluation.get("audit")
    if not isinstance(audit, Mapping):
        errors.append("audit config missing")
    else:
        audit_expected = {
            "sampleRate": 0.1,
            "swapAB": True,
            "thirdAdjudicationOnDisagreement": True,
        }
        for name, expected in audit_expected.items():
            if audit.get(name) != expected:
                errors.append(
                    f"audit.{name}={audit.get(name)!r} expected={expected!r}"
                )
    if errors:
        raise ProtocolError("evaluation contract mismatch: " + "; ".join(errors))


def write_json(path: Path, value: Any) -> None:
    v4.write_json(path, value)


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    v4.write_jsonl(path, values)


def read_json(path: Path) -> Any:
    return v4.read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return v4.read_jsonl(path)


def metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return v4.metric(tp, fp, fn)


def zero_counts(fields: Sequence[str] = FIELDS) -> dict[str, dict[str, int]]:
    return {field: {"tp": 0, "fp": 0, "fn": 0} for field in fields}


def add_counts(
    target: dict[str, dict[str, int]],
    source: Mapping[str, Mapping[str, int]],
) -> None:
    for field in target:
        for name in ("tp", "fp", "fn"):
            target[field][name] += int(source[field][name])


def normalized_string(value: Any) -> str:
    return " ".join(v4.base.normalize_unicode(value).strip().casefold().split())


def strict_json_equal(left: Any, right: Any) -> bool:
    return v4.base.fixed_canonical(left) == v4.base.fixed_canonical(right)


def position_valid(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict) or set(item) != {"Begin", "End"}:
            return False
        for name in ("Begin", "End"):
            coordinate = item[name]
            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or not all(
                    isinstance(number, int) and not isinstance(number, bool)
                    for number in coordinate
                )
            ):
                return False
    return True


def object_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, str)
            and bool(item.strip())
            and not v4.is_absent(item)
            for item in value
        )
    )


def strict_numeric_array(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    if not all(v4.is_finite_number(item) for item in value):
        return None
    return [float(item) for item in value]


def strict_numeric_equal(left: Any, right: Any) -> bool:
    gold = strict_numeric_array(left)
    prediction = strict_numeric_array(right)
    if gold is None or prediction is None or len(gold) != len(prediction):
        return False
    remaining = list(prediction)
    for gold_value in gold:
        match = next((
            index
            for index, predicted_value in enumerate(remaining)
            if math.isclose(
                gold_value,
                predicted_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ), None)
        if match is None:
            return False
        remaining.pop(match)
    return not remaining


def field_type_valid(field: str, value: Any) -> bool:
    if field == "ObjectName":
        return object_valid(value)
    if field == "DataName":
        return isinstance(value, str)
    if field == "Position":
        return position_valid(value)
    if field == "Trend":
        return isinstance(value, str)
    if field == "Num":
        return strict_numeric_array(value) is not None
    if field == "Text":
        return isinstance(value, str)
    raise KeyError(field)


def binding_schema_errors(binding: Any) -> list[str]:
    if not isinstance(binding, dict):
        return ["binding_not_object"]
    errors: list[str] = []
    missing = set(FIELDS) - set(binding)
    extra = set(binding) - set(FIELDS)
    if missing:
        errors.append("missing:" + ",".join(sorted(missing)))
    if extra:
        errors.append("extra:" + ",".join(sorted(extra)))
    for field in FIELDS:
        if field in binding and not field_type_valid(field, binding[field]):
            errors.append(f"{field}:invalid_type")
    return errors


def prediction_schema_errors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return [{
            "predictionIndex": None,
            "errors": ["result_not_array"],
        }]
    return [
        {
            "predictionIndex": index,
            "errors": binding_schema_errors(binding),
        }
        for index, binding in enumerate(value)
        if binding_schema_errors(binding)
    ]


def anchor_key(binding: Any) -> tuple[str, str] | None:
    if not isinstance(binding, dict):
        return None
    data_name = binding.get("DataName")
    position = binding.get("Position")
    if not isinstance(data_name, str) or not position_valid(position):
        return None
    return (
        data_name.strip().lower(),
        json.dumps(position, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def align_bindings(
    gold: Sequence[Mapping[str, Any]],
    prediction: Sequence[Any],
) -> dict[str, Any]:
    available: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for gold_index, binding in enumerate(gold):
        key = anchor_key(binding)
        if key is None:
            raise ProtocolError(f"gold binding {gold_index} has an invalid hard anchor")
        available[key].append(gold_index)
    matches: list[dict[str, int]] = []
    unmatched_prediction: list[int] = []
    used_gold: set[int] = set()
    for prediction_index, binding in enumerate(prediction):
        key = anchor_key(binding)
        if key is None or not available.get(key):
            unmatched_prediction.append(prediction_index)
            continue
        gold_index = available[key].popleft()
        used_gold.add(gold_index)
        matches.append({
            "goldIndex": gold_index,
            "predictionIndex": prediction_index,
        })
    unmatched_gold = [
        index for index in range(len(gold)) if index not in used_gold
    ]
    return {
        "matches": matches,
        "unmatchedGold": unmatched_gold,
        "unmatchedPrediction": unmatched_prediction,
    }


def trend_class(value: Any) -> str | None:
    if v4.is_absent(value):
        return None
    if not isinstance(value, str):
        return None
    normalized = normalized_string(value)
    return TREND_CLASS_ALIASES.get(normalized, normalized)


def normalized_object_pairs(
    gold: Any, prediction: Any
) -> list[tuple[int, int]]:
    if not isinstance(gold, list) or not isinstance(prediction, list):
        return []
    return v4.base.maximum_matching(
        len(gold),
        len(prediction),
        lambda left, right: (
            isinstance(gold[left], str)
            and isinstance(prediction[right], str)
            and normalized_string(gold[left])
            == normalized_string(prediction[right])
        ),
    )


def object_normalized_equal(gold: Any, prediction: Any) -> bool:
    if not object_valid(gold) or not object_valid(prediction):
        return False
    pairs = normalized_object_pairs(gold, prediction)
    return len(pairs) == len(gold) == len(prediction)


def object_is_data_name_substitution(
    gold: Mapping[str, Any], prediction: Mapping[str, Any]
) -> bool:
    gold_objects = gold.get("ObjectName")
    predicted_objects = prediction.get("ObjectName")
    data_name = prediction.get("DataName")
    if (
        not object_valid(gold_objects)
        or not object_valid(predicted_objects)
        or not isinstance(data_name, str)
    ):
        return False
    data_normalized = normalized_string(data_name)
    return (
        {normalized_string(item) for item in predicted_objects} == {data_normalized}
        and data_normalized not in {
            normalized_string(item) for item in gold_objects
        }
    )


def normalized_text_equal(gold: Any, prediction: Any) -> bool:
    return (
        isinstance(gold, str)
        and isinstance(prediction, str)
        and normalized_string(gold) == normalized_string(prediction)
    )


def build_semantic_plan(
    input_data: Any,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    binding_index: int,
) -> dict[str, Any]:
    gold_objects = gold.get("ObjectName")
    predicted_objects = prediction.get("ObjectName")
    exact_pairs = normalized_object_pairs(gold_objects, predicted_objects)
    matched_gold = {left for left, _ in exact_pairs}
    matched_prediction = {right for _, right in exact_pairs}
    unmatched_gold = [
        index
        for index in range(len(gold_objects) if isinstance(gold_objects, list) else 0)
        if index not in matched_gold
    ]
    unmatched_prediction = [
        index
        for index in range(
            len(predicted_objects) if isinstance(predicted_objects, list) else 0
        )
        if index not in matched_prediction
    ]
    decisions: list[dict[str, Any]] = []
    object_decision_id = f"binding_{binding_index}_ObjectName"
    substitution = object_is_data_name_substitution(gold, prediction)
    if unmatched_gold and unmatched_prediction and not substitution:
        decisions.append({
            "decisionId": object_decision_id,
            "field": "ObjectName",
            "gold": [gold_objects[index] for index in unmatched_gold],
            "prediction": [
                predicted_objects[index] for index in unmatched_prediction
            ],
            "DataName": gold["DataName"],
            "Position": gold["Position"],
            "dataEvidence": v4.base.build_data_evidence(input_data, gold),
        })

    gold_trend = gold.get("Trend")
    predicted_trend = prediction.get("Trend")
    gold_class = trend_class(gold_trend)
    predicted_class = trend_class(predicted_trend)
    trend_decision_id = f"binding_{binding_index}_Trend"
    if not field_type_valid("Trend", predicted_trend):
        trend_deterministic: bool | None = False
    elif gold_class == predicted_class:
        trend_deterministic = True
    elif (gold_class is None) != (predicted_class is None):
        trend_deterministic = False
    else:
        trend_deterministic = None
        decisions.append({
            "decisionId": trend_decision_id,
            "field": "Trend",
            "gold": gold_trend,
            "prediction": predicted_trend,
            "DataName": gold["DataName"],
            "Position": gold["Position"],
            "dataEvidence": v4.base.build_data_evidence(input_data, gold),
        })

    text_decision_id = f"binding_{binding_index}_Text"
    text_deterministic = normalized_text_equal(
        gold.get("Text"), prediction.get("Text")
    )
    if not text_deterministic and field_type_valid("Text", prediction.get("Text")):
        decisions.append({
            "decisionId": text_decision_id,
            "field": "Text",
            "gold": gold.get("Text"),
            "prediction": prediction.get("Text"),
            "DataName": gold["DataName"],
            "Position": gold["Position"],
            "protectedTokens": {
                "gold": finance_protected_tokens(gold.get("Text")),
                "prediction": finance_protected_tokens(prediction.get("Text")),
            },
        })
    return {
        "decisions": decisions,
        "goldObjects": gold_objects,
        "predictedObjects": predicted_objects,
        "exactObjectPairs": exact_pairs,
        "unmatchedGoldObjects": unmatched_gold,
        "unmatchedPredictedObjects": unmatched_prediction,
        "objectDecisionId": object_decision_id,
        "objectDataNameSubstitution": substitution,
        "trendDecisionId": trend_decision_id,
        "trendDeterministic": trend_deterministic,
        "textDecisionId": text_decision_id,
        "textDeterministic": text_deterministic,
    }


def annotate_judgments(
    judgments: Mapping[str, Mapping[str, Any]],
    source_text: str,
) -> dict[str, dict[str, Any]]:
    return {
        decision_id: {
            **item,
            "evidenceValid": v4.base.evidence_in_source(
                item.get("evidenceSpan"), source_text
            ),
        }
        for decision_id, item in judgments.items()
    }


def decision_accepts(
    judgment: Mapping[str, Any] | None,
    threshold: float,
    field: str,
) -> bool:
    if not judgment:
        return False
    evidence_valid = (
        judgment.get("evidenceValid") is True
        and not judgment.get("validationError")
    )
    object_pairs_valid = field != "ObjectName" or bool(
        judgment.get("matchedPairs")
    )
    return (
        judgment.get("equivalent") is True
        and float(judgment.get("confidence", 0.0)) >= threshold
        and bool(judgment.get("evidenceSpan"))
        and evidence_valid
        and object_pairs_valid
    )


def object_semantic_equal(
    plan: Mapping[str, Any],
    judgments: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> bool:
    if plan["objectDataNameSubstitution"]:
        return False
    pairs = list(plan["exactObjectPairs"])
    judgment = judgments.get(str(plan["objectDecisionId"]))
    if decision_accepts(judgment, threshold, "ObjectName"):
        unmatched_gold = plan["unmatchedGoldObjects"]
        unmatched_prediction = plan["unmatchedPredictedObjects"]
        used_gold: set[int] = set()
        used_prediction: set[int] = set()
        for pair in judgment.get("matchedPairs", []):
            if not isinstance(pair, dict):
                continue
            left = pair.get("goldIndex")
            right = pair.get("predictionIndex")
            if (
                isinstance(left, int)
                and isinstance(right, int)
                and 0 <= left < len(unmatched_gold)
                and 0 <= right < len(unmatched_prediction)
                and left not in used_gold
                and right not in used_prediction
            ):
                used_gold.add(left)
                used_prediction.add(right)
                pairs.append((
                    unmatched_gold[left],
                    unmatched_prediction[right],
                ))
    gold_size = (
        len(plan["goldObjects"])
        if isinstance(plan["goldObjects"], list)
        else 0
    )
    prediction_size = (
        len(plan["predictedObjects"])
        if isinstance(plan["predictedObjects"], list)
        else 0
    )
    return len(pairs) == gold_size == prediction_size


def semantic_passes(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    plan: Mapping[str, Any],
    judgments: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> dict[str, bool]:
    trend_deterministic = plan["trendDeterministic"]
    text_deterministic = bool(plan["textDeterministic"])
    return {
        "ObjectName": object_semantic_equal(plan, judgments, threshold),
        "DataName": True,
        "Position": True,
        "Trend": (
            bool(trend_deterministic)
            if trend_deterministic is not None
            else decision_accepts(
                judgments.get(str(plan["trendDecisionId"])),
                threshold,
                "Trend",
            )
        ),
        "Num": strict_numeric_equal(
            gold.get("Num"), prediction.get("Num")
        ),
        "Text": (
            True
            if text_deterministic
            else decision_accepts(
                judgments.get(str(plan["textDecisionId"])),
                threshold,
                "Text",
            )
        ),
    }


def deterministic_passes(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    normalized: bool,
) -> dict[str, bool]:
    if normalized:
        object_pass = object_normalized_equal(
            gold.get("ObjectName"), prediction.get("ObjectName")
        )
        trend_pass = (
            field_type_valid("Trend", prediction.get("Trend"))
            and trend_class(gold.get("Trend")) == trend_class(
                prediction.get("Trend")
            )
        )
        text_pass = normalized_text_equal(
            gold.get("Text"), prediction.get("Text")
        )
    else:
        object_pass = strict_json_equal(
            gold.get("ObjectName"), prediction.get("ObjectName")
        )
        trend_pass = strict_json_equal(
            gold.get("Trend"), prediction.get("Trend")
        )
        text_pass = strict_json_equal(
            gold.get("Text"), prediction.get("Text")
        )
    return {
        "ObjectName": object_pass,
        "DataName": True,
        "Position": True,
        "Trend": trend_pass,
        "Num": strict_numeric_equal(
            gold.get("Num"), prediction.get("Num")
        ),
        "Text": text_pass,
    }


def add_aligned_pair(
    counts: dict[str, dict[str, int]],
    passes: Mapping[str, bool],
    prediction: Mapping[str, Any],
) -> None:
    for field in counts:
        if passes[field]:
            counts[field]["tp"] += 1
        else:
            counts[field]["fn"] += 1
            if field_type_valid(field, prediction.get(field)):
                counts[field]["fp"] += 1


def add_unmatched_gold(
    counts: dict[str, dict[str, int]],
    amount: int = 1,
) -> None:
    for field in counts:
        counts[field]["fn"] += amount


def add_unmatched_prediction(
    counts: dict[str, dict[str, int]],
    prediction: Any,
) -> None:
    if not isinstance(prediction, dict):
        return
    for field in counts:
        if field in prediction and field_type_valid(field, prediction[field]):
            counts[field]["fp"] += 1


def fields_metrics(
    counts: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    field_results = {
        field: metric(**counts[field]) for field in counts
    }
    macro = {
        name: statistics.mean(
            field_results[field][name] for field in field_results
        )
        for name in ("precision", "recall", "f1")
    }
    pooled = {
        name: sum(int(counts[field][name]) for field in counts)
        for name in ("tp", "fp", "fn")
    }
    return {
        "counts": {
            field: dict(counts[field]) for field in counts
        },
        "fields": field_results,
        "macro": macro,
        "micro": metric(**pooled),
    }


def parse_semantic_number(value: Any) -> tuple[float, float] | None:
    if v4.is_finite_number(value):
        return float(value), 1.0
    if not isinstance(value, str):
        return None
    normalized = normalized_string(value).replace(",", "")
    match = re.fullmatch(
        r"[$€£¥]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        r"(thousand|million|billion|trillion)?\s*(%|percent(?:age)?)?",
        normalized,
    )
    if not match:
        return None
    factor = UNIT_FACTORS.get(match.group(2) or "", 1.0)
    if match.group(3):
        factor *= 0.01
    return float(match.group(1)), factor


def context_unit_factor(binding: Mapping[str, Any]) -> float | None:
    text = normalized_string(binding.get("Text"))
    found = [
        factor for unit, factor in UNIT_FACTORS.items()
        if re.search(rf"\b{re.escape(unit)}\b", text)
    ]
    if "%" in text or re.search(r"\bpercent(?:age)?\b", text):
        found.append(0.01)
    return found[0] if len(set(found)) == 1 else None


def numeric_candidate_arrays(
    binding: Mapping[str, Any],
) -> tuple[list[list[float]], bool]:
    raw = binding.get("Num")
    is_array = isinstance(raw, list)
    items = raw if is_array else [raw]
    schema_invalid = not is_array
    if not items:
        return [[]], False
    if all(v4.is_absent(item) for item in items):
        return [[]], True
    parsed: list[float] = []
    unscaled: list[float] = []
    explicit_factors: list[float] = []
    for item in items:
        value = parse_semantic_number(item)
        if value is None:
            return [], schema_invalid
        unscaled.append(value[0])
        parsed.append(value[0] * value[1])
        explicit_factors.append(value[1])
        if not v4.is_finite_number(item):
            schema_invalid = True
    candidates = [parsed]
    if any(factor == 0.01 for factor in explicit_factors):
        candidates.append(unscaled)
    unit = context_unit_factor(binding)
    if unit is not None and all(factor == 1.0 for factor in explicit_factors):
        candidates.append([value * unit for value in parsed])
    unique: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique, schema_invalid


def numeric_arrays_equal(
    gold: Sequence[float],
    prediction: Sequence[float],
    relative_tolerance: float,
) -> bool:
    if len(gold) != len(prediction):
        return False
    remaining = list(prediction)
    for gold_value in gold:
        match = next((
            index
            for index, predicted_value in enumerate(remaining)
            if math.isclose(
                gold_value,
                predicted_value,
                rel_tol=relative_tolerance,
                abs_tol=1e-9,
            )
        ), None)
        if match is None:
            return False
        remaining.pop(match)
    return not remaining


def numeric_sensitivity_equal(
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    relative_tolerance: float,
    *,
    allow_units: bool,
) -> tuple[bool, bool]:
    if not allow_units:
        left = strict_numeric_array(gold.get("Num"))
        right = strict_numeric_array(prediction.get("Num"))
        if left is None or right is None:
            return False, False
        return numeric_arrays_equal(left, right, relative_tolerance), False
    gold_candidates, _ = numeric_candidate_arrays(gold)
    prediction_candidates, schema_invalid = numeric_candidate_arrays(prediction)
    passed = any(
        numeric_arrays_equal(left, right, relative_tolerance)
        for left in gold_candidates
        for right in prediction_candidates
    )
    return passed, schema_invalid and passed


def finance_tokens(value: Any) -> list[str]:
    normalized = v4.base.normalize_unicode(value).casefold()
    return [
        re.sub(r"\s+", "", match.group(0))
        for match in FINANCE_TOKEN_RE.finditer(normalized)
    ]


def finance_protected_tokens(value: Any) -> list[str]:
    return [
        token for token in finance_tokens(value)
        if any(character.isdigit() for character in token)
        or "%" in token
        or any(symbol in token for symbol in "$€£¥")
    ]


def token_f1(gold: Any, prediction: Any) -> dict[str, Any]:
    gold_tokens = Counter(finance_tokens(gold))
    prediction_tokens = Counter(finance_tokens(prediction))
    overlap = sum((gold_tokens & prediction_tokens).values())
    precision = (
        overlap / sum(prediction_tokens.values())
        if prediction_tokens
        else (1.0 if not gold_tokens else 0.0)
    )
    recall = (
        overlap / sum(gold_tokens.values())
        if gold_tokens
        else (1.0 if not prediction_tokens else 0.0)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    protected_gold = Counter(finance_protected_tokens(gold))
    protected_prediction = Counter(finance_protected_tokens(prediction))
    missing = list((protected_gold - protected_prediction).elements())
    extra = list((protected_prediction - protected_gold).elements())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missingProtectedTokens": missing,
        "extraProtectedTokens": extra,
    }


def decision_signature(
    results: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> list[tuple[Any, ...]]:
    return sorted(
        (
            decision_id,
            decision_accepts(item, threshold, str(decision_id).rsplit("_", 1)[-1]),
            bool(item.get("equivalent")),
            tuple(
                sorted(
                    (
                        pair.get("goldIndex"),
                        pair.get("predictionIndex"),
                    )
                    for pair in item.get("matchedPairs", [])
                    if isinstance(pair, dict)
                )
            ),
        )
        for decision_id, item in results.items()
    )


def resolve_generation_artifact(
    manifest: Mapping[str, Any],
    name: str,
    generation_root: Path | None,
) -> Path:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return Path("")
    declared = Path(str(files.get(name) or ""))
    if declared.is_file() or generation_root is None:
        return declared
    relocated = (
        generation_root
        / "cases"
        / str(manifest.get("outputId"))
        / f"run_{int(manifest.get('run', 0)):02d}"
        / declared.name
    )
    return relocated if relocated.is_file() else declared


def load_prediction_records(
    manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    generation_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    files = manifest.get("files")
    hashes = manifest.get("hashes")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ProtocolError("generation manifest lacks files/hashes")
    prediction_path = resolve_generation_artifact(
        manifest, "predictions", generation_root
    )
    if not prediction_path.is_file():
        if manifest.get("status") == "runtime_blocked":
            return [
                {
                    "source": target["source"],
                    "result": [],
                    "formatValid": False,
                    "inputText": "",
                    "inputData": "",
                    "parserDiagnostic": {"error": "runtime_blocked"},
                }
                for target in targets
            ], {}
        raise ProtocolError(f"prediction file missing: {prediction_path}")
    if v4.sha256_file(prediction_path) != hashes.get("predictions"):
        raise ProtocolError(f"prediction SHA mismatch: {prediction_path}")
    predictions = read_jsonl(prediction_path)
    candidate_by_source: dict[str, dict[str, Any]] = {}
    candidate_path_raw = files.get("retrieverCandidates")
    if candidate_path_raw:
        candidate_path = resolve_generation_artifact(
            manifest, "retrieverCandidates", generation_root
        )
        if (
            not candidate_path.is_file()
            or v4.sha256_file(candidate_path)
            != hashes.get("retrieverCandidates")
        ):
            raise ProtocolError(f"candidate artifact mismatch: {candidate_path}")
        candidate_by_source = {
            str(item.get("source")): item for item in read_jsonl(candidate_path)
        }
    return predictions, candidate_by_source


def validate_run_contract(
    manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    generation_root: Path | None = None,
) -> dict[str, Any]:
    target_sources = [str(item["source"]) for item in targets]
    prediction_sources = [str(item.get("source") or "") for item in predictions]
    errors: list[str] = []
    if prediction_sources != target_sources:
        errors.append("source_order_or_coverage_mismatch")
    if len(set(prediction_sources)) != len(prediction_sources):
        errors.append("duplicate_prediction_source")
    if int(manifest.get("expectedRows", -1)) != len(targets):
        errors.append("manifest_expected_rows_mismatch")
    files = manifest.get("files") or {}
    hashes = manifest.get("hashes") or {}
    for name in ("rawResponse", "prompts", "runtime", "formatReport"):
        raw_path = files.get(name)
        if not raw_path:
            errors.append(f"missing_artifact:{name}")
            continue
        path = resolve_generation_artifact(manifest, name, generation_root)
        if not path.is_file() or v4.sha256_file(path) != hashes.get(name):
            errors.append(f"artifact_hash_mismatch:{name}")
    leakage_markers = (
        '"targetBindings"',
        '"gold_targets"',
        '"Binding_Result"',
    )
    prompts_path = resolve_generation_artifact(
        manifest, "prompts", generation_root
    )
    leakage_hits: list[str] = []
    if prompts_path.is_file():
        prompt_text = prompts_path.read_text(encoding="utf-8")
        leakage_hits = [
            marker for marker in leakage_markers if marker in prompt_text
        ]
        if leakage_hits:
            errors.append("gold_leakage_marker")
    effective = str(manifest.get("effectiveRoute") or "")
    base_mode = str(manifest.get("baseRouteMode") or "historical")
    declared = str(manifest.get("declaredRoute") or manifest.get("route") or "")
    if base_mode == "formal":
        if effective != declared:
            errors.append("formal_declared_effective_route_mismatch")
        if declared == "direct-binding":
            if manifest.get("adapter") is not None:
                errors.append("formal_direct_adapter_loaded")
            if manifest.get("converterModel") is not None:
                errors.append("formal_direct_converter_used")
        elif declared == "adapter-converter":
            if manifest.get("adapter") is None:
                errors.append("formal_adapter_missing")
            if manifest.get("converterModel") != "gpt-5.5":
                errors.append("formal_converter_model_mismatch")
            if manifest.get("reasoningEffort") != "medium":
                errors.append("formal_converter_effort_mismatch")
    elif base_mode == "direct-diagnostic":
        if effective != "direct-diagnostic-native":
            errors.append("direct_effective_route_mismatch")
        if manifest.get("adapter") is not None:
            errors.append("direct_adapter_loaded")
        if manifest.get("converterModel") is not None:
            errors.append("direct_converter_used")
    elif effective == "retriever-converter":
        if manifest.get("converterModel") != "gpt-5.5":
            errors.append("historical_converter_model_mismatch")
        if manifest.get("reasoningEffort") != "medium":
            errors.append("historical_converter_effort_mismatch")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "sources": len(prediction_sources),
        "uniqueSources": len(set(prediction_sources)),
        "promptGoldLeakageMarkers": leakage_hits,
        "declaredRoute": manifest.get("route"),
        "effectiveRoute": effective,
        "baseRouteMode": base_mode,
        "actualModel": manifest.get("actualModel"),
        "converterModel": manifest.get("converterModel"),
        "reasoningEffort": manifest.get("reasoningEffort"),
    }


def update_trend_confusion(
    counts: dict[str, dict[str, int]],
    gold_class: str | None,
    prediction_class: str | None,
) -> None:
    if gold_class is None and prediction_class is None:
        return
    if gold_class == prediction_class:
        assert gold_class is not None
        counts[gold_class]["tp"] += 1
        return
    if gold_class is not None:
        counts[gold_class]["fn"] += 1
    if prediction_class is not None:
        counts[prediction_class]["fp"] += 1


def evaluate_run(
    manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
    case_dir: Path,
    judge: v4.base.SemanticJudge,
    frozen_v4_root: Path | None = None,
) -> dict[str, Any]:
    predictions, candidate_by_source = load_prediction_records(manifest, targets, frozen_v4_root)
    contract = validate_run_contract(
        manifest, predictions, targets, frozen_v4_root
    )
    if contract["status"] != "passed":
        raise ProtocolError(
            f"{manifest.get('outputId')} run {manifest.get('run')} contract errors: "
            + ", ".join(contract["errors"])
        )
    if len(predictions) != len(targets):
        raise ProtocolError(
            f"prediction rows={len(predictions)} expected={len(targets)}"
        )

    stage_counts = {
        stage: zero_counts() for stage in ABLATIONS
    }
    threshold_counts = {
        str(threshold): zero_counts() for threshold in THRESHOLDS
    }
    conditional_counts = zero_counts(CONTENT_FIELDS)
    numeric_sensitivity_counts = {
        "relative_0.1pct": {"tp": 0, "fp": 0, "fn": 0},
        "relative_0.5pct": {"tp": 0, "fp": 0, "fn": 0},
        "relative_1pct": {"tp": 0, "fp": 0, "fn": 0},
        "units_plus_relative_1pct": {"tp": 0, "fp": 0, "fn": 0},
    }
    numeric_schema_invalid_recovered = Counter()
    trend_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    row_records: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    token_records: list[dict[str, Any]] = []
    root_causes = Counter()
    gold_binding_total = 0
    predicted_binding_total = 0
    matched_binding_total = 0
    strict_schema_rows = 0

    for target, prediction_record in zip(targets, predictions):
        source = str(target["source"])
        gold_bindings = target["targetBindings"]
        raw_predicted = prediction_record.get("result")
        predicted_bindings = (
            raw_predicted if isinstance(raw_predicted, list) else []
        )
        schema_errors = prediction_schema_errors(raw_predicted)
        schema_valid = bool(prediction_record.get("formatValid")) and not schema_errors
        if schema_valid:
            strict_schema_rows += 1
        else:
            root_causes["schema_invalid_rows"] += 1
        if not prediction_record.get("formatValid"):
            predicted_bindings = []

        alignment = align_bindings(gold_bindings, predicted_bindings)
        gold_binding_total += len(gold_bindings)
        predicted_binding_total += len(predicted_bindings)
        matched_binding_total += len(alignment["matches"])
        if gold_bindings and not predicted_bindings:
            root_causes["generator_no_binding_rows"] += 1
        if not gold_bindings and not predicted_bindings:
            root_causes["correct_zero_binding_rows"] += 1
        root_causes["unmatched_gold_bindings"] += len(
            alignment["unmatchedGold"]
        )
        root_causes["unmatched_prediction_bindings"] += len(
            alignment["unmatchedPrediction"]
        )

        gate_errors = v4.row_gate(gold_bindings, prediction_record)
        if gate_errors and alignment["matches"]:
            root_causes["whole_row_gate_undercount_rows"] += 1

        plans: list[tuple[dict[str, int], dict[str, Any]]] = []
        decisions: list[dict[str, Any]] = []
        for match_index, match in enumerate(alignment["matches"]):
            gold_binding = gold_bindings[match["goldIndex"]]
            predicted_binding = predicted_bindings[match["predictionIndex"]]
            plan = build_semantic_plan(
                prediction_record.get("inputData") or "",
                gold_binding,
                predicted_binding,
                match_index,
            )
            plans.append((match, plan))
            decisions.extend(plan["decisions"])
        source_text = str(prediction_record.get("inputText") or "")
        judgments = annotate_judgments(
            judge.decide(
                source, source_text, decisions, audit_label="primary-v5.1"
            ),
            source_text,
        )
        if decisions:
            audit_candidates.append({
                "source": source,
                "sourceText": str(prediction_record.get("inputText") or ""),
                "decisions": decisions,
                "primary": {
                    decision_id: judgments[decision_id]
                    for decision_id in [
                        str(item["decisionId"]) for item in decisions
                    ]
                },
            })

        row_stage_counts = {
            stage: zero_counts() for stage in ABLATIONS
        }
        row_threshold_counts = {
            str(threshold): zero_counts() for threshold in THRESHOLDS
        }
        row_conditional = zero_counts(CONTENT_FIELDS)
        match_details: list[dict[str, Any]] = []

        for match, plan in plans:
            gold_binding = gold_bindings[match["goldIndex"]]
            predicted_binding = predicted_bindings[match["predictionIndex"]]
            hard = deterministic_passes(
                gold_binding, predicted_binding, normalized=False
            )
            normalized = deterministic_passes(
                gold_binding, predicted_binding, normalized=True
            )
            semantic_by_threshold = {
                str(threshold): semantic_passes(
                    gold_binding,
                    predicted_binding,
                    plan,
                    judgments,
                    threshold,
                )
                for threshold in THRESHOLDS
            }
            semantic = semantic_by_threshold[str(PRIMARY_THRESHOLD)]
            add_aligned_pair(
                row_stage_counts["hard_anchor_local_exact"],
                hard,
                predicted_binding,
            )
            add_aligned_pair(
                row_stage_counts["deterministic_normalized"],
                normalized,
                predicted_binding,
            )
            add_aligned_pair(
                row_stage_counts["semantic_gpt55_medium"],
                semantic,
                predicted_binding,
            )
            for threshold in THRESHOLDS:
                add_aligned_pair(
                    row_threshold_counts[str(threshold)],
                    semantic_by_threshold[str(threshold)],
                    predicted_binding,
                )
            add_aligned_pair(
                row_conditional,
                {
                    field: semantic[field] for field in CONTENT_FIELDS
                },
                predicted_binding,
            )
            if not all(semantic[field] for field in CONTENT_FIELDS):
                root_causes["matched_binding_semantic_errors"] += 1

            update_trend_confusion(
                trend_counts,
                trend_class(gold_binding.get("Trend")),
                trend_class(predicted_binding.get("Trend")),
            )
            token_detail = token_f1(
                gold_binding.get("Text"), predicted_binding.get("Text")
            )
            token_records.append({
                "source": source,
                **match,
                **token_detail,
            })

            numeric_detail: dict[str, Any] = {}
            for name, tolerance, units in (
                ("relative_0.1pct", 0.001, False),
                ("relative_0.5pct", 0.005, False),
                ("relative_1pct", 0.01, False),
                ("units_plus_relative_1pct", 0.01, True),
            ):
                passed, recovered_invalid = numeric_sensitivity_equal(
                    gold_binding,
                    predicted_binding,
                    tolerance,
                    allow_units=units,
                )
                numeric_detail[name] = {
                    "pass": passed,
                    "schemaInvalidRecovered": recovered_invalid,
                }
                if passed:
                    numeric_sensitivity_counts[name]["tp"] += 1
                else:
                    numeric_sensitivity_counts[name]["fn"] += 1
                    if (
                        strict_numeric_array(predicted_binding.get("Num"))
                        is not None
                        or numeric_candidate_arrays(predicted_binding)[0]
                    ):
                        numeric_sensitivity_counts[name]["fp"] += 1
                if recovered_invalid:
                    numeric_schema_invalid_recovered[name] += 1

            match_judgments = {
                decision_id: judgment
                for decision_id, judgment in judgments.items()
                if decision_id.startswith(
                    f"binding_{alignment['matches'].index(match)}_"
                )
            }
            match_details.append({
                **match,
                "hardExact": hard,
                "deterministicNormalized": normalized,
                "semanticByThreshold": semantic_by_threshold,
                "objectDataNameSubstitution": plan[
                    "objectDataNameSubstitution"
                ],
                "judge": match_judgments,
                "numericSensitivity": numeric_detail,
                "textTokenF1": token_detail,
            })

        for gold_index in alignment["unmatchedGold"]:
            for counts in row_stage_counts.values():
                add_unmatched_gold(counts)
            for counts in row_threshold_counts.values():
                add_unmatched_gold(counts)
            update_trend_confusion(
                trend_counts,
                trend_class(gold_bindings[gold_index].get("Trend")),
                None,
            )
            for item in numeric_sensitivity_counts.values():
                item["fn"] += 1
        for prediction_index in alignment["unmatchedPrediction"]:
            binding = predicted_bindings[prediction_index]
            for counts in row_stage_counts.values():
                add_unmatched_prediction(counts, binding)
            for counts in row_threshold_counts.values():
                add_unmatched_prediction(counts, binding)
            update_trend_confusion(
                trend_counts,
                None,
                trend_class(binding.get("Trend")) if isinstance(binding, dict) else None,
            )
            for name, item in numeric_sensitivity_counts.items():
                if not isinstance(binding, dict):
                    continue
                if name == "units_plus_relative_1pct":
                    valid = bool(numeric_candidate_arrays(binding)[0])
                else:
                    valid = strict_numeric_array(binding.get("Num")) is not None
                if valid:
                    item["fp"] += 1

        for stage in ABLATIONS:
            add_counts(stage_counts[stage], row_stage_counts[stage])
        for threshold in THRESHOLDS:
            add_counts(
                threshold_counts[str(threshold)],
                row_threshold_counts[str(threshold)],
            )
        add_counts(conditional_counts, row_conditional)

        candidate = candidate_by_source.get(source)
        row_records.append({
            "source": source,
            "goldBindings": len(gold_bindings),
            "predictedBindings": len(predicted_bindings),
            "matchedBindings": len(alignment["matches"]),
            "alignment": alignment,
            "strictSchemaValid": schema_valid,
            "schemaErrors": schema_errors,
            "v4RowGateAccepted": not gate_errors,
            "v4RowGateErrors": gate_errors,
            "stageMetrics": {
                stage: fields_metrics(row_stage_counts[stage])
                for stage in ABLATIONS
            },
            "primaryCounts": row_stage_counts[
                "semantic_gpt55_medium"
            ],
            "conditionalContentCounts": row_conditional,
            "matchDetails": match_details,
            "candidate": (
                {
                    "candidate": candidate.get("candidate"),
                    "candidateSha256": candidate.get("candidateSha256"),
                    "raw": candidate.get("raw"),
                }
                if candidate
                else None
            ),
            "declaredRoute": manifest.get("route"),
            "effectiveRoute": manifest.get("effectiveRoute"),
            "parserDiagnostic": prediction_record.get("parserDiagnostic"),
        })

    audit_config = evaluation_config["audit"]
    sample_size = (
        max(
            1,
            math.ceil(
                len(audit_candidates) * float(audit_config["sampleRate"])
            ),
        )
        if audit_candidates
        else 0
    )
    audit_seed = (
        int(audit_config["seed"])
        + int(manifest["run"])
        + int(v4.sha256_text(str(manifest["outputId"]))[:8], 16)
    )
    selected_audit = sorted(
        random.Random(audit_seed).sample(
            range(len(audit_candidates)), sample_size
        )
    )
    audit_records: list[dict[str, Any]] = []
    for candidate_index in selected_audit:
        candidate = audit_candidates[candidate_index]
        swapped = annotate_judgments(
            judge.decide(
                candidate["source"],
                candidate["sourceText"],
                candidate["decisions"],
                force_opposite=True,
                audit_label="swap-audit-v5.1",
            ),
            candidate["sourceText"],
        )
        primary_signature = decision_signature(
            candidate["primary"], PRIMARY_THRESHOLD
        )
        swapped_signature = decision_signature(
            swapped, PRIMARY_THRESHOLD
        )
        agreement = primary_signature == swapped_signature
        third = None
        if (
            not agreement
            and audit_config.get("thirdAdjudicationOnDisagreement")
        ):
            third = annotate_judgments(
                judge.decide(
                    candidate["source"],
                    candidate["sourceText"],
                    candidate["decisions"],
                    audit_label="third-adjudication-v5.1",
                ),
                candidate["sourceText"],
            )
        audit_records.append({
            "candidateIndex": candidate_index,
            "source": candidate["source"],
            "primary": candidate["primary"],
            "swapped": swapped,
            "agreement": agreement,
            "thirdAdjudication": third,
        })
    agreements = sum(item["agreement"] for item in audit_records)
    audit_summary = {
        "population": len(audit_candidates),
        "sampled": len(audit_records),
        "sampleRateConfigured": audit_config["sampleRate"],
        "seed": audit_seed,
        "agreements": agreements,
        "disagreements": len(audit_records) - agreements,
        "agreementRate": (
            agreements / len(audit_records) if audit_records else None
        ),
        "thirdAdjudications": sum(
            item["thirdAdjudication"] is not None
            for item in audit_records
        ),
    }

    trend_metrics = {
        name: metric(**counts)
        for name, counts in sorted(trend_counts.items())
        if name is not None
    }
    trend_macro = (
        statistics.mean(item["f1"] for item in trend_metrics.values())
        if trend_metrics
        else None
    )
    token_summary = {
        "alignedBindings": len(token_records),
        "meanPrecision": (
            statistics.mean(item["precision"] for item in token_records)
            if token_records else None
        ),
        "meanRecall": (
            statistics.mean(item["recall"] for item in token_records)
            if token_records else None
        ),
        "meanF1": (
            statistics.mean(item["f1"] for item in token_records)
            if token_records else None
        ),
        "bindingsMissingProtectedTokens": sum(
            bool(item["missingProtectedTokens"]) for item in token_records
        ),
        "rankingRole": "diagnostic_only",
    }
    frozen_relative = (
        Path("evaluation_v4_no_gpt41")
        / "cases"
        / str(manifest["outputId"])
        / f"run_{int(manifest['run']):02d}"
        / "metrics.json"
    )
    frozen_candidates = []
    if frozen_v4_root is not None:
        frozen_candidates.append(frozen_v4_root / frozen_relative)
    frozen_candidates.append(
        Path(str(manifest["files"]["predictions"])).parent.parent.parent.parent
        / frozen_relative
    )
    frozen_path = next(
        (path for path in frozen_candidates if path.is_file()), None
    )
    frozen = read_json(frozen_path) if frozen_path is not None else None

    result = {
        "protocol": PROTOCOL,
        "outputId": manifest["outputId"],
        "sourceId": manifest["sourceId"],
        "run": int(manifest["run"]),
        "seed": int(manifest["seed"]),
        "declaredRoute": manifest.get("route"),
        "effectiveRoute": manifest.get("effectiveRoute"),
        "baseRouteMode": manifest.get("baseRouteMode", "historical"),
        "actualModel": manifest.get("actualModel"),
        "converterModel": manifest.get("converterModel"),
        "reasoningEffort": manifest.get("reasoningEffort"),
        "rows": len(predictions),
        "strictSchemaValidity": {
            "validRows": strict_schema_rows,
            "invalidRows": len(predictions) - strict_schema_rows,
            "rate": strict_schema_rows / len(predictions) if predictions else 0.0,
        },
        "coverage": {
            "goldBindings": gold_binding_total,
            "predictedBindings": predicted_binding_total,
            "matchedBindings": matched_binding_total,
            "anchorRecall": (
                matched_binding_total / gold_binding_total
                if gold_binding_total else 1.0
            ),
            "anchorPrecision": (
                matched_binding_total / predicted_binding_total
                if predicted_binding_total else (
                    1.0 if not gold_binding_total else 0.0
                )
            ),
        },
        "ablations": {
            stage: fields_metrics(stage_counts[stage])
            for stage in ABLATIONS
        },
        "thresholdSensitivity": {
            threshold: fields_metrics(counts)
            for threshold, counts in threshold_counts.items()
        },
        "conditionalContent": {
            **fields_metrics(conditional_counts),
            "defined": matched_binding_total > 0,
            "matchedBindingsOnly": True,
            "rankingRole": "diagnostic_only",
        },
        "trendClassMetricsExcludingNone": {
            "classes": trend_metrics,
            "macroF1": trend_macro,
        },
        "numericSensitivity": {
            name: {
                **metric(**counts),
                "schemaInvalidRecovered": int(
                    numeric_schema_invalid_recovered[name]
                ),
            }
            for name, counts in numeric_sensitivity_counts.items()
        },
        "textFinanceSafeTokenF1": token_summary,
        "semanticAudit": audit_summary,
        "rootCauses": dict(root_causes),
        "contractValidation": contract,
        "frozenHybridV4": frozen,
    }
    write_jsonl(case_dir / "records.jsonl", row_records)
    write_jsonl(case_dir / "text_token_f1.jsonl", token_records)
    write_json(case_dir / "semantic_audit.json", {
        "summary": audit_summary,
        "records": audit_records,
    })
    write_json(case_dir / "metrics.json", result)
    return result


def statistic(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "sampleSd": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": statistics.mean(values),
        "sampleSd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
    accessor,
) -> dict[str, Any]:
    values = [float(accessor(run)) for run in runs]
    ordered = sorted(
        (
            {
                "run": int(run["run"]),
                "value": float(accessor(run)),
            }
            for run in runs
        ),
        key=lambda item: (item["value"], -item["run"]),
        reverse=True,
    )
    return {
        **statistic(values),
        "top1": ordered[0] if ordered else None,
        "top3": {
            "runs": [item["run"] for item in ordered[:3]],
            "mean": (
                statistics.mean(item["value"] for item in ordered[:3])
                if ordered else None
            ),
        },
    }


def summarize_conditional_runs(
    runs: Sequence[Mapping[str, Any]],
    accessor,
) -> dict[str, Any]:
    eligible = [
        run for run in runs
        if bool(run["conditionalContent"].get("defined"))
    ]
    return {
        **summarize_runs(eligible, accessor),
        "eligibleRuns": len(eligible),
        "totalRuns": len(runs),
    }


def summarize_stage(
    runs: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    pooled = zero_counts()
    for run in runs:
        add_counts(
            pooled,
            run["ablations"][stage]["counts"],
        )
    return {
        "fields": {
            field: {
                name: summarize_runs(
                    runs,
                    lambda run, field=field, name=name: run[
                        "ablations"
                    ][stage]["fields"][field][name],
                )
                for name in ("precision", "recall", "f1")
            }
            for field in FIELDS
        },
        "macro": {
            name: summarize_runs(
                runs,
                lambda run, name=name: run["ablations"][stage]["macro"][name],
            )
            for name in ("precision", "recall", "f1")
        },
        "micro": {
            name: summarize_runs(
                runs,
                lambda run, name=name: run["ablations"][stage]["micro"][name],
            )
            for name in ("precision", "recall", "f1")
        },
        "pooled": fields_metrics(pooled),
    }


def summarize_frozen_v4(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not runs or any(run.get("frozenHybridV4") is None for run in runs):
        return None
    return {
        "source": "immutable evaluation_v4_no_gpt41 metrics.json",
        "macro": {
            name: summarize_runs(
                runs,
                lambda run, name=name: run["frozenHybridV4"]["macro"][name],
            )
            for name in ("precision", "recall", "f1")
        },
        "micro": {
            name: summarize_runs(
                runs,
                lambda run, name=name: run[
                    "frozenHybridV4"
                ]["overallMicro"][name],
            )
            for name in ("precision", "recall", "f1")
        },
        "fields": {
            field: {
                name: summarize_runs(
                    runs,
                    lambda run, field=field, name=name: run[
                        "frozenHybridV4"
                    ]["fieldMetrics"][field][name],
                )
                for name in ("precision", "recall", "f1")
            }
            for field in FIELDS
        },
    }


def aggregate_case(
    output_id: str,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda item: int(item["run"]))
    if [int(item["run"]) for item in ordered] != list(range(1, 11)):
        raise ProtocolError(f"{output_id} does not have runs 1-10")
    root_causes = Counter()
    for run in ordered:
        root_causes.update(run["rootCauses"])
    coverage_totals = {
        name: sum(int(run["coverage"][name]) for run in ordered)
        for name in ("goldBindings", "predictedBindings", "matchedBindings")
    }
    main = summarize_stage(ordered, "semantic_gpt55_medium")
    return {
        "outputId": output_id,
        "sourceId": ordered[0]["sourceId"],
        "declaredRoute": ordered[0]["declaredRoute"],
        "effectiveRoute": ordered[0]["effectiveRoute"],
        "baseRouteMode": ordered[0]["baseRouteMode"],
        "actualModels": sorted({
            str(run["actualModel"]) for run in ordered
        }),
        "runs": len(ordered),
        "ablations": {
            "frozen_hybrid_v4": summarize_frozen_v4(ordered),
            **{
                stage: summarize_stage(ordered, stage)
                for stage in ABLATIONS
            },
        },
        "primary": main,
        "coverage": {
            **coverage_totals,
            "anchorRecall": (
                coverage_totals["matchedBindings"]
                / coverage_totals["goldBindings"]
                if coverage_totals["goldBindings"] else 1.0
            ),
            "anchorPrecision": (
                coverage_totals["matchedBindings"]
                / coverage_totals["predictedBindings"]
                if coverage_totals["predictedBindings"] else (
                    1.0 if not coverage_totals["goldBindings"] else 0.0
                )
            ),
            "runStatistics": {
                name: summarize_runs(
                    ordered,
                    lambda run, name=name: run["coverage"][name],
                )
                for name in (
                    "goldBindings",
                    "predictedBindings",
                    "matchedBindings",
                    "anchorRecall",
                    "anchorPrecision",
                )
            },
        },
        "strictSchemaValidity": summarize_runs(
            ordered,
            lambda run: run["strictSchemaValidity"]["rate"],
        ),
        "conditionalContent": {
            name: summarize_conditional_runs(
                ordered,
                lambda run, name=name: run[
                    "conditionalContent"
                ]["macro"][name],
            )
            for name in ("precision", "recall", "f1")
        },
        "thresholdSensitivity": {
            threshold: {
                name: summarize_runs(
                    ordered,
                    lambda run, threshold=threshold, name=name: run[
                        "thresholdSensitivity"
                    ][threshold]["macro"][name],
                )
                for name in ("precision", "recall", "f1")
            }
            for threshold in ("0.7", "0.8", "0.9")
        },
        "numericSensitivity": {
            method: {
                name: summarize_runs(
                    ordered,
                    lambda run, method=method, name=name: run[
                        "numericSensitivity"
                    ][method][name],
                )
                for name in ("precision", "recall", "f1")
            }
            for method in ordered[0]["numericSensitivity"]
        },
        "textFinanceSafeTokenF1": summarize_runs(
            ordered,
            lambda run: (
                run["textFinanceSafeTokenF1"]["meanF1"]
                if run["textFinanceSafeTokenF1"]["meanF1"] is not None
                else 0.0
            ),
        ),
        "rootCauses": dict(root_causes),
        "runResults": [
            {
                "run": run["run"],
                "seed": run["seed"],
                "schemaValidity": run["strictSchemaValidity"]["rate"],
                "coverage": run["coverage"],
                "macro": run["ablations"][
                    "semantic_gpt55_medium"
                ]["macro"],
                "micro": run["ablations"][
                    "semantic_gpt55_medium"
                ]["micro"],
                "rootCauses": run["rootCauses"],
            }
            for run in ordered
        ],
    }


def selected_from_scorecard(path: Path, threshold: float) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        str(row["output_id"])
        for row in rows
        if float(row["overall_mean_precision"]) < threshold
    }


def write_tables(
    evaluation_root: Path,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    scorecard = evaluation_root / "reference_aligned_case_scorecard.tsv"
    fields = [
        "output_id",
        "source_id",
        "declared_route",
        "effective_route",
        "schema_validity_mean",
        "gold_bindings",
        "predicted_bindings",
        "matched_bindings",
        "anchor_precision",
        "anchor_recall",
        "macro_precision_mean",
        "macro_precision_sample_sd",
        "macro_recall_mean",
        "macro_recall_sample_sd",
        "macro_f1_mean",
        "macro_f1_sample_sd",
        "pooled_micro_precision",
        "pooled_micro_recall",
        "pooled_micro_f1",
        "conditional_content_f1_mean",
    ]
    with scorecard.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for case in cases:
            macro = case["primary"]["macro"]
            pooled = case["primary"]["pooled"]["micro"]
            writer.writerow({
                "output_id": case["outputId"],
                "source_id": case["sourceId"],
                "declared_route": case["declaredRoute"],
                "effective_route": case["effectiveRoute"],
                "schema_validity_mean": case["strictSchemaValidity"]["mean"],
                "gold_bindings": case["coverage"]["goldBindings"],
                "predicted_bindings": case["coverage"]["predictedBindings"],
                "matched_bindings": case["coverage"]["matchedBindings"],
                "anchor_precision": case["coverage"]["anchorPrecision"],
                "anchor_recall": case["coverage"]["anchorRecall"],
                "macro_precision_mean": macro["precision"]["mean"],
                "macro_precision_sample_sd": macro["precision"]["sampleSd"],
                "macro_recall_mean": macro["recall"]["mean"],
                "macro_recall_sample_sd": macro["recall"]["sampleSd"],
                "macro_f1_mean": macro["f1"]["mean"],
                "macro_f1_sample_sd": macro["f1"]["sampleSd"],
                "pooled_micro_precision": pooled["precision"],
                "pooled_micro_recall": pooled["recall"],
                "pooled_micro_f1": pooled["f1"],
                "conditional_content_f1_mean": case[
                    "conditionalContent"
                ]["f1"]["mean"],
            })
    field_scorecard = evaluation_root / "reference_aligned_field_scorecard.tsv"
    field_columns = [
        "output_id", "field",
        "precision_mean", "precision_sample_sd",
        "recall_mean", "recall_sample_sd",
        "f1_mean", "f1_sample_sd",
        "pooled_tp", "pooled_fp", "pooled_fn",
        "pooled_precision", "pooled_recall", "pooled_f1",
    ]
    with field_scorecard.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=field_columns, delimiter="\t"
        )
        writer.writeheader()
        for case in cases:
            for field in FIELDS:
                field_summary = case["primary"]["fields"][field]
                pooled = case["primary"]["pooled"]["fields"][field]
                writer.writerow({
                    "output_id": case["outputId"],
                    "field": field,
                    "precision_mean": field_summary["precision"]["mean"],
                    "precision_sample_sd": field_summary["precision"]["sampleSd"],
                    "recall_mean": field_summary["recall"]["mean"],
                    "recall_sample_sd": field_summary["recall"]["sampleSd"],
                    "f1_mean": field_summary["f1"]["mean"],
                    "f1_sample_sd": field_summary["f1"]["sampleSd"],
                    "pooled_tp": pooled["tp"],
                    "pooled_fp": pooled["fp"],
                    "pooled_fn": pooled["fn"],
                    "pooled_precision": pooled["precision"],
                    "pooled_recall": pooled["recall"],
                    "pooled_f1": pooled["f1"],
                })

    ablation_scorecard = (
        evaluation_root / "reference_aligned_ablation_scorecard.tsv"
    )
    ablation_columns = [
        "output_id", "stage", "available",
        "macro_precision_mean", "macro_precision_sample_sd",
        "macro_recall_mean", "macro_recall_sample_sd",
        "macro_f1_mean", "macro_f1_sample_sd",
        "run_mean_micro_precision", "run_mean_micro_recall",
        "run_mean_micro_f1", "pooled_micro_precision",
        "pooled_micro_recall", "pooled_micro_f1",
    ]
    with ablation_scorecard.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ablation_columns, delimiter="\t"
        )
        writer.writeheader()
        for case in cases:
            for stage_name, stage in case["ablations"].items():
                if stage is None:
                    writer.writerow({
                        "output_id": case["outputId"],
                        "stage": stage_name,
                        "available": False,
                    })
                    continue
                pooled_micro = (stage.get("pooled") or {}).get("micro") or {}
                writer.writerow({
                    "output_id": case["outputId"],
                    "stage": stage_name,
                    "available": True,
                    "macro_precision_mean": stage["macro"]["precision"]["mean"],
                    "macro_precision_sample_sd": stage["macro"]["precision"]["sampleSd"],
                    "macro_recall_mean": stage["macro"]["recall"]["mean"],
                    "macro_recall_sample_sd": stage["macro"]["recall"]["sampleSd"],
                    "macro_f1_mean": stage["macro"]["f1"]["mean"],
                    "macro_f1_sample_sd": stage["macro"]["f1"]["sampleSd"],
                    "run_mean_micro_precision": stage["micro"]["precision"]["mean"],
                    "run_mean_micro_recall": stage["micro"]["recall"]["mean"],
                    "run_mean_micro_f1": stage["micro"]["f1"]["mean"],
                    "pooled_micro_precision": pooled_micro.get("precision", ""),
                    "pooled_micro_recall": pooled_micro.get("recall", ""),
                    "pooled_micro_f1": pooled_micro.get("f1", ""),
                })

    per_run = evaluation_root / "reference_aligned_per_run.tsv"
    run_fields = [
        "output_id", "run", "seed", "schema_validity",
        "gold_bindings", "predicted_bindings", "matched_bindings",
        "macro_precision", "macro_recall", "macro_f1",
        "micro_precision", "micro_recall", "micro_f1",
    ]
    with per_run.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=run_fields, delimiter="\t"
        )
        writer.writeheader()
        for case in cases:
            for run in case["runResults"]:
                writer.writerow({
                    "output_id": case["outputId"],
                    "run": run["run"],
                    "seed": run["seed"],
                    "schema_validity": run["schemaValidity"],
                    "gold_bindings": run["coverage"]["goldBindings"],
                    "predicted_bindings": run["coverage"]["predictedBindings"],
                    "matched_bindings": run["coverage"]["matchedBindings"],
                    "macro_precision": run["macro"]["precision"],
                    "macro_recall": run["macro"]["recall"],
                    "macro_f1": run["macro"]["f1"],
                    "micro_precision": run["micro"]["precision"],
                    "micro_recall": run["micro"]["recall"],
                    "micro_f1": run["micro"]["f1"],
                })
    return {
        "scorecard": str(scorecard),
        "fieldScorecard": str(field_scorecard),
        "ablationScorecard": str(ablation_scorecard),
        "perRun": str(per_run),
    }


def comparison_with(
    cases: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, Any]:
    other = read_json(path)
    current_method = method_metadata()
    other_method = other.get("method")
    if other.get("protocol") != PROTOCOL:
        raise ProtocolError(
            "route comparison protocol mismatch: "
            f"{other.get('protocol')!r} != {PROTOCOL!r}"
        )
    if (
        not isinstance(other_method, Mapping)
        or other_method.get("methodCompatibilitySha256")
        != current_method["methodCompatibilitySha256"]
    ):
        raise ProtocolError("route comparison method fingerprint mismatch")
    other_cases = {
        item["outputId"]: item for item in other.get("cases", [])
    }
    comparisons = []
    for case in cases:
        previous = other_cases.get(case["outputId"])
        if previous is None:
            continue
        current_macro = case["primary"]["macro"]
        previous_macro = previous["primary"]["macro"]
        comparisons.append({
            "outputId": case["outputId"],
            "currentEffectiveRoute": case["effectiveRoute"],
            "referenceEffectiveRoute": previous["effectiveRoute"],
            "macroPrecisionDelta": (
                current_macro["precision"]["mean"]
                - previous_macro["precision"]["mean"]
            ),
            "macroRecallDelta": (
                current_macro["recall"]["mean"]
                - previous_macro["recall"]["mean"]
            ),
            "macroF1Delta": (
                current_macro["f1"]["mean"]
                - previous_macro["f1"]["mean"]
            ),
            "predictedBindingsDelta": (
                case["coverage"]["predictedBindings"]
                - previous["coverage"]["predictedBindings"]
            ),
            "matchedBindingsDelta": (
                case["coverage"]["matchedBindings"]
                - previous["coverage"]["matchedBindings"]
            ),
        })
    return {
        "referenceReport": str(path),
        "overlappingCases": len(comparisons),
        "cases": comparisons,
        "interpretation": (
            "Route deltas diagnose generation-path effects only; score increases "
            "are not by themselves evidence that an evaluator is better."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    method = report["method"]
    lines = [
        f"# Experiment 6 {report['protocol']} diagnostic",
        "",
        f"- status: {report['status']}",
        f"- method revision: `{method['revision']}`",
        f"- evaluator SHA-256: `{method['evaluatorSha256']}`",
        f"- method compatibility SHA-256: `{method['methodCompatibilitySha256']}`",
        (
            f"- Trend ontology: `{method['trendOntologyVersion']}` / "
            f"`{method['trendOntologySha256']}`"
        ),
        f"- primary score role: `{method['primaryScoreRole']}`",
        f"- semantic evidence gate: {method['semanticEvidenceGate']}",
        f"- cases: {report['completedCases']}",
        f"- predictions: {report['formalPredictions']}",
        f"- judge: {report['judge']['model']} / {report['judge']['reasoningEffort']}",
        f"- primary confidence: {PRIMARY_THRESHOLD}",
        f"- strict Num: {method['numericPrimary']}",
        "",
        "| case | effective route | matched / predicted / gold | macro P | macro R | macro F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        macro = case["primary"]["macro"]
        coverage = case["coverage"]
        lines.append(
            f"| {case['outputId']} | {case['effectiveRoute']} | "
            f"{coverage['matchedBindings']} / {coverage['predictedBindings']} / "
            f"{coverage['goldBindings']} | {macro['precision']['mean']:.6f} | "
            f"{macro['recall']['mean']:.6f} | {macro['f1']['mean']:.6f} |"
        )
    mistral = next(
        (
            case for case in report["cases"]
            if case["outputId"] == "6_mistral_base_d"
        ),
        None,
    )
    if mistral:
        macro_precision = mistral["primary"]["macro"]["precision"]["mean"]
        pooled_precision = mistral["primary"]["pooled"]["micro"]["precision"]
        lines.extend([
            "",
            "## mistral_base_d denominator note",
            "",
            (
                "Mean-of-run precision and pooled micro precision answer different "
                f"questions: run mean={macro_precision:.6f}, pooled={pooled_precision:.6f}. "
                "A run with no predicted positive contributes zero under the frozen "
                "run-mean convention; a tiny number of correct, zero-FP bindings can "
                "therefore coexist with very low recall and field F1."
            ),
        ])
    lines.extend([
        "",
        "Conditional content and token-F1 are diagnostic only and must not be used for ranking.",
        "Numeric unit/tolerance results are sensitivity analyses; schema-invalid numeric strings remain invalid.",
    ])
    return "\n".join(lines) + "\n"


def artifact_inventory(
    generation_root: Path,
    evaluation_root: Path,
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for manifest in manifests:
        for name, raw_path in sorted((manifest.get("files") or {}).items()):
            path = Path(str(raw_path))
            if path.is_file():
                entries.append({
                    "scope": "generation",
                    "outputId": manifest.get("outputId"),
                    "run": manifest.get("run"),
                    "name": name,
                    "path": str(path),
                    "sha256": v4.sha256_file(path),
                })
    for path in sorted(evaluation_root.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.json":
            entries.append({
                "scope": "evaluation",
                "path": str(path),
                "sha256": v4.sha256_file(path),
            })
    return {
        "generationRoot": str(generation_root),
        "evaluationRoot": str(evaluation_root),
        "artifacts": entries,
        "count": len(entries),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    config, evaluation = v4.load_effective_config(args.config.resolve())
    validate_evaluation_contract(evaluation)
    output_root = args.output_root.resolve()
    evaluation_root = (
        args.evaluation_root.resolve()
        if args.evaluation_root
        else output_root / "evaluation_reference_aligned_v5"
    )
    generation_snapshot = output_root / "generation_config.snapshot.json"
    if not generation_snapshot.is_file():
        raise ProtocolError(f"generation snapshot missing: {generation_snapshot}")
    generation_config = read_json(generation_snapshot)
    gold_path = v4.workspace_path(evaluation["goldPath"]).resolve()
    if v4.sha256_file(gold_path) != evaluation["goldSha256"]:
        raise ProtocolError("gold SHA-256 mismatch")
    targets = read_json(gold_path).get("rows")
    if not isinstance(targets, list) or len(targets) != int(config["expectedRows"]):
        raise ProtocolError("gold row count mismatch")

    all_manifests = [
        read_json(path)
        for path in sorted((output_root / "manifests").glob("*.json"))
    ]
    if args.mistral_chat_projection:
        invalid = [
            manifest
            for manifest in all_manifests
            if manifest.get("protocol") != MISTRAL_CHAT_PROJECTION_PROTOCOL
            or manifest.get("official") is not False
            or manifest.get("diagnosticOnly") is not True
            or manifest.get("claimEligible") is not False
            or manifest.get("goldAccessed") is not False
        ]
        if invalid:
            raise ProtocolError("invalid Mistral diagnostic projection manifest")
        manifests = all_manifests
    else:
        manifests = [manifest for manifest in all_manifests if manifest.get("official")]
    selected: set[str] = set(args.only_case)
    if args.select_low_precision_from:
        selected.update(selected_from_scorecard(
            args.select_low_precision_from.resolve(),
            args.precision_threshold,
        ))
    if selected:
        manifests = [
            manifest for manifest in manifests
            if str(manifest.get("outputId")) in selected
        ]
        missing = selected - {
            str(manifest.get("outputId")) for manifest in manifests
        }
        if missing:
            raise ProtocolError(f"selected cases missing: {sorted(missing)}")
    if not manifests:
        raise ProtocolError("no official manifests selected")
    keys = [
        (str(item.get("outputId")), int(item.get("run", -1)))
        for item in manifests
    ]
    if len(keys) != len(set(keys)):
        raise ProtocolError("duplicate outputId/run manifest")
    grouped_manifests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        grouped_manifests[str(manifest["outputId"])].append(manifest)
    incomplete = {
        output_id: sorted(int(item["run"]) for item in items)
        for output_id, items in grouped_manifests.items()
        if sorted(int(item["run"]) for item in items) != list(range(1, 11))
    }
    if incomplete and not args.allow_incomplete:
        raise ProtocolError(f"incomplete run coverage: {incomplete}")
    statuses = Counter(str(item.get("status")) for item in manifests)
    disallowed = [
        item for item in manifests
        if item.get("status") not in {
            "completed", "completed_with_format_errors", "runtime_blocked"
        }
    ]
    if disallowed:
        raise ProtocolError("manifest has unsupported status")

    (
        judge_config,
        judge_prompt_report,
        judge_system_prompts,
        judge_validation_versions,
    ) = v4.configure_judge(output_root, config, evaluation_root)
    judge = v4.FieldPromptSemanticJudge(
        judge_config,
        evaluation_root / "judge_checkpoint.jsonl",
        judge_system_prompts,
        judge_validation_versions,
        disabled=args.judge_disabled,
    )
    run_results: list[dict[str, Any]] = []
    for manifest in sorted(
        manifests,
        key=lambda item: (str(item["outputId"]), int(item["run"])),
    ):
        case_dir = (
            evaluation_root
            / "cases"
            / str(manifest["outputId"])
            / f"run_{int(manifest['run']):02d}"
        )
        run_results.append(evaluate_run(
            manifest,
            targets,
            evaluation,
            case_dir,
            judge,
            output_root,
        ))
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in run_results:
        grouped_results[str(result["outputId"])].append(result)
    cases = [
        aggregate_case(output_id, runs)
        for output_id, runs in sorted(grouped_results.items())
    ]
    tables = write_tables(evaluation_root, cases)
    report = {
        "protocol": PROTOCOL,
        "time": v4.utc_now(),
        "status": (
            "runtime_blocked_diagnostic"
            if statuses.get("runtime_blocked")
            else "completed"
        ),
        "generationRoot": str(output_root),
        "evaluationConfig": str(args.config.resolve()),
        "evaluationConfigSha256": v4.sha256_file(args.config.resolve()),
        "sourceEvaluationConfigProtocol": config.get("protocol"),
        "sourceEvaluationConfigSchemaVersion": config.get("schemaVersion"),
        "generationConfig": str(generation_snapshot),
        "generationConfigSha256": v4.sha256_file(generation_snapshot),
        "goldPath": str(gold_path),
        "goldSha256": v4.sha256_file(gold_path),
        "completedCases": len(cases),
        "completedCaseRuns": len(run_results),
        "formalPredictions": len(run_results) * len(targets),
        "manifestStatuses": dict(statuses),
        "judge": {
            "model": judge_config["model"],
            "reasoningEffort": judge_config["reasoningEffort"],
            "minimumConfidence": judge_config["minimumConfidence"],
            "disabled": judge.disabled,
            "liveCalls": judge.live_calls,
            "cacheHits": judge.cache_hits,
            "prompt": judge_prompt_report,
        },
        "method": method_metadata(),
        "tables": tables,
        "cases": cases,
    }
    if args.compare_report:
        report["routeComparison"] = comparison_with(
            cases, args.compare_report.resolve()
        )
    write_json(evaluation_root / "evaluation_report.json", report)
    (evaluation_root / "evaluation_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    inventory = artifact_inventory(output_root, evaluation_root, manifests)
    write_json(evaluation_root / "artifact_inventory.json", inventory)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "config"
            / "experiment6_narrative2_hybrid_v4_no_gpt41.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--select-low-precision-from", type=Path)
    parser.add_argument("--precision-threshold", type=float, default=0.3)
    parser.add_argument("--compare-report", type=Path)
    parser.add_argument("--judge-disabled", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--mistral-chat-projection", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = build(parse_args(argv))
    except (ProtocolError, v4.ProtocolError, v4.base.JudgeError) as error:
        print(json.dumps({
            "time": v4.utc_now(),
            "protocol": PROTOCOL,
            "status": "blocked",
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "time": report["time"],
        "protocol": report["protocol"],
        "status": report["status"],
        "completedCases": report["completedCases"],
        "completedCaseRuns": report["completedCaseRuns"],
        "formalPredictions": report["formalPredictions"],
        "evaluationRoot": str(
            parse_args(argv).evaluation_root
            or Path(report["generationRoot"]) / "evaluation_reference_aligned_v5"
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
