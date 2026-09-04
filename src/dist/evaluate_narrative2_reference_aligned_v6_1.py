#!/usr/bin/env python3
"""Experiment 6 reference-aligned v6.1 deterministic content scorer.

This scorer is diagnostic-only.  DataName and Position remain hard anchors.
It adds evidence-backed ObjectName, Trend, and Num normalization while retaining
TP/FP/FN accounting and an explicit method audit.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_narrative2_reference_aligned_v6_0_2 as base
from evaluate_narrative2_reference_aligned_v6_0_2 import *  # noqa: F401,F403


PROTOCOL = "experiment6-reference-aligned-v6.1.0"
PRIMARY_FIELDS = base.PRIMARY_FIELDS
TREND_CLASSES = (
    "head_and_shoulders",
    "cup_and_handle",
    "rounding_bottom",
    "double_top",
    "triple_top",
    "increase",
    "decrease",
    "stable",
    "reversal",
    "peak",
    "trough",
    "none",
)
FLEXIBLE_NUMBER_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:st|nd|rd|th)?",
    flags=re.IGNORECASE,
)
SCALE_FACTORS = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "bn": 1e9,
    "billion": 1e9,
    "tn": 1e12,
    "trillion": 1e12,
}
_ACTIVE_NUMERIC_CONFIG: Mapping[str, Any] | None = None


def configure(config: Mapping[str, Any]) -> None:
    """Bind the versioned numeric policy before evaluating rows."""
    global _ACTIVE_NUMERIC_CONFIG
    validate_config(config)
    _ACTIVE_NUMERIC_CONFIG = dict(config["numericSemantic"])


class ObjectMatcher(base.ObjectMatcher):
    """Conservative v6.0.2 matcher plus leading-article normalization."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.leading_articles = {
            normalized_string(item)
            for item in config.get("leadingArticles", ["a", "an", "the"])
        }

    def trim_leading_articles(self, value: str) -> tuple[str, ...]:
        tokens = list(object_tokens(value))
        while tokens and tokens[0] in self.leading_articles:
            tokens.pop(0)
        return tuple(tokens)

    def article_equal(self, gold: str, prediction: str) -> bool:
        gold_tokens = self.trim_leading_articles(gold)
        prediction_tokens = self.trim_leading_articles(prediction)
        if not gold_tokens or not prediction_tokens:
            return False
        return self.contains(gold_tokens, prediction_tokens) or self.contains(
            prediction_tokens, gold_tokens
        )

    def equal_with_method(self, gold: Any, prediction: Any) -> tuple[bool, str]:
        if super().equal(gold, prediction):
            return True, "v6.0.2"
        left = self.mentions(gold)
        right = self.mentions(prediction)
        if left is None or right is None or len(left) != len(right):
            return False, "different"
        pairs = maximum_matching(
            len(left), len(right), lambda i, j: self.article_equal(left[i], right[j])
        )
        if len(pairs) == len(left):
            return True, "leading-article-normalized"
        return False, "different"

    def equal(self, gold: Any, prediction: Any) -> bool:
        return self.equal_with_method(gold, prediction)[0]


@dataclass(frozen=True)
class SemanticNumericValue:
    value: float
    scale: float | None
    unit: str | None
    sourceType: str


def _flatten_singleton_numeric(value: Any) -> tuple[list[Any], bool]:
    items = value if isinstance(value, list) else [value]
    flattened = False
    while len(items) == 1 and isinstance(items[0], list):
        items = items[0]
        flattened = True
    return list(items), flattened


def _semantic_numeric_item(value: Any) -> SemanticNumericValue | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return SemanticNumericValue(number, None, None, "json-number")
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = normalized_string(value)
    matches = list(FLEXIBLE_NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return None
    raw_number = re.sub(
        r"(?:st|nd|rd|th)$", "", matches[0].group(0), flags=re.IGNORECASE
    )
    try:
        number = float(raw_number.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    remainder = text[: matches[0].start()] + " " + text[matches[0].end() :]
    words = set(re.findall(r"[^\W\d_]+", remainder, flags=re.UNICODE))
    scales = {SCALE_FACTORS[word] for word in words if word in SCALE_FACTORS}
    if len(scales) > 1:
        return None
    currency_units = {
        base.CURRENCY_SYMBOLS[symbol]
        for symbol in base.CURRENCY_SYMBOLS
        if symbol in remainder
    }
    currency_units.update(
        words & {"usd", "eur", "gbp", "jpy", "sgd", "aud", "cad", "cny", "rmb"}
    )
    percentage_point = (
        "pp" in words
        or {"percentage", "point"} <= words
        or {"percentage", "points"} <= words
    )
    percent = "%" in remainder or "percent" in words or "pct" in words
    units = set(currency_units)
    if percentage_point:
        units.add("percentage_point")
    elif percent:
        units.add("percent")
    if len(units) > 1:
        return None
    return SemanticNumericValue(
        value=number,
        scale=next(iter(scales), None),
        unit=next(iter(units), None),
        sourceType="numeric-phrase",
    )


def _data_name_scale(value: Any) -> float | None:
    text = normalized_string(value)
    for pattern, factor in (
        (r"\btrillion", 1e12),
        (r"\bbillion|\bbn\b", 1e9),
        (r"\bmillion|\bmn\b", 1e6),
        (r"\bthousand|\b000\b", 1e3),
    ):
        if re.search(pattern, text):
            return factor
    return None


def _semantic_numeric_pair(
    gold: SemanticNumericValue,
    prediction: SemanticNumericValue,
    data_scale: float | None,
    relative_tolerance: float,
) -> tuple[bool, str]:
    if gold.unit is not None and prediction.unit is not None and gold.unit != prediction.unit:
        return False, "unit-conflict"
    explicit_scale_conflict = (
        gold.scale is not None
        and prediction.scale is not None
        and gold.scale != prediction.scale
    )
    if not explicit_scale_conflict and math.isclose(
        gold.value, prediction.value, rel_tol=1e-9, abs_tol=1e-9
    ):
        return True, "semantic-normalized"
    gold_factor = gold.scale if gold.scale is not None else (data_scale or 1.0)
    prediction_factor = (
        prediction.scale if prediction.scale is not None else (data_scale or 1.0)
    )
    if math.isclose(
        gold.value * gold_factor,
        prediction.value * prediction_factor,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return True, "scale-converted"
    if relative_tolerance > 0:
        coefficient_close = not explicit_scale_conflict and math.isclose(
            gold.value,
            prediction.value,
            rel_tol=relative_tolerance,
            abs_tol=1e-9,
        )
        scaled_close = math.isclose(
            gold.value * gold_factor,
            prediction.value * prediction_factor,
            rel_tol=relative_tolerance,
            abs_tol=1e-9,
        )
        if coefficient_close or scaled_close:
            return True, "tolerance"
    return False, "different"


def semantic_numeric_equal(
    gold: Any,
    prediction: Any,
    data_name: Any,
    relative_tolerance: float = 0.01,
) -> tuple[bool, str, dict[str, Any]]:
    if base.numeric_equal(gold, prediction):
        return True, "v6.0.2", {"schemaRepair": False}
    gold_items, gold_flattened = _flatten_singleton_numeric(gold)
    prediction_items, prediction_flattened = _flatten_singleton_numeric(prediction)
    left = [_semantic_numeric_item(item) for item in gold_items]
    right = [_semantic_numeric_item(item) for item in prediction_items]
    if (
        any(item is None for item in left + right)
        or len(left) != len(right)
    ):
        return False, "invalid-or-cardinality", {
            "schemaRepair": prediction_flattened,
            "goldFlattened": gold_flattened,
        }
    data_scale = _data_name_scale(data_name)
    pair_methods: dict[tuple[int, int], str] = {}
    for left_index, gold_item in enumerate(left):
        for right_index, prediction_item in enumerate(right):
            equal, method = _semantic_numeric_pair(
                gold_item, prediction_item, data_scale, relative_tolerance
            )
            if equal:
                pair_methods[(left_index, right_index)] = method
    pairs = maximum_matching(
        len(left), len(right), lambda i, j: (i, j) in pair_methods
    )
    if len(pairs) != len(left):
        return False, "different", {
            "schemaRepair": prediction_flattened,
            "goldFlattened": gold_flattened,
        }
    methods = {pair_methods[pair] for pair in pairs}
    method = (
        "tolerance"
        if "tolerance" in methods
        else "scale-converted"
        if "scale-converted" in methods
        else "semantic-normalized"
    )
    schema_valid = (
        isinstance(prediction, list)
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in prediction
        )
    )
    return True, method, {
        "schemaRepair": not schema_valid,
        "predictionFlattened": prediction_flattened,
        "dataNameScale": data_scale,
    }


class TrendClassifier:
    """Closed direction/pattern ontology from the frozen recommendation document."""

    def __init__(self, config: Mapping[str, Any], allow_model: bool = True):
        del allow_model
        self.config = config
        self.rules = [
            (str(item["class"]), [re.compile(pattern) for pattern in item["patterns"]])
            for item in config["rules"]
        ]

    def classify(self, value: Any, threshold: float | None = None) -> dict[str, Any]:
        del threshold
        if value is None:
            return {"class": "none", "method": "null-none", "similarity": None}
        if not isinstance(value, str):
            return {"class": None, "method": "invalid", "similarity": None}
        normalized = normalized_string(value)
        for label, patterns in self.rules:
            if any(pattern.search(normalized) for pattern in patterns):
                return {"class": label, "method": "closed-rule", "similarity": None}
        return {"class": "none", "method": "default-none", "similarity": None}

    def deterministic(self, value: Any) -> str | None:
        return self.classify(value)["class"]


def _legacy_trend_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return "none" if value is None else None
    text = normalized_string(value)
    for label, patterns in (
        ("head_and_shoulders", (r"head(?:\s+|-)and(?:\s+|-)shoulders",)),
        ("cup_and_handle", (r"cup(?:\s+|-)and(?:\s+|-)handle",)),
        ("rounding_bottom", (r"round(?:ing|ed)?(?:\s+|-)bottom",)),
        ("double_top", (r"double(?:\s+|-)top",)),
        ("triple_top", (r"triple(?:\s+|-)top",)),
    ):
        if any(re.search(pattern, text) for pattern in patterns):
            return label
    return "none"


def trend_metrics_from_confusion(
    confusion: Mapping[str, Mapping[str, int]],
    support: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    supported_f1: list[float] = []
    for label in TREND_CLASSES:
        tp = int(confusion.get(label, {}).get(label, 0))
        fn = sum(int(value) for key, value in confusion.get(label, {}).items() if key != label)
        fp = sum(
            int(columns.get(label, 0))
            for gold_label, columns in confusion.items()
            if gold_label != label
        )
        values = metric(tp=tp, fp=fp, fn=fn)
        fields[label] = values
        if (support or {}).get(label, tp + fn) > 0:
            supported_f1.append(float(values["f1"] or 0.0))
    return {
        "perClass": fields,
        "macroF1SupportedClasses": statistics.fmean(supported_f1) if supported_f1 else None,
        "supportedClasses": [label for label in TREND_CLASSES if int((support or {}).get(label, 0)) > 0],
    }


def _append_example(
    examples: dict[str, list[dict[str, Any]]],
    key: str,
    value: dict[str, Any],
    limit: int = 12,
) -> None:
    if len(examples.setdefault(key, [])) < limit:
        examples[key].append(value)


def evaluate_rows(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    objects: ObjectMatcher,
    trends: TrendClassifier,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, records = base.evaluate_rows(targets, predictions, objects, trends)
    counts = summary["primary"]["counts"]
    binding_counts = {
        key: int(summary["bindingLevel"][key]) for key in ("tp", "fp", "fn")
    }
    audit = Counter(
        object_article_normalized=0,
        num_semantic_normalized=0,
        num_scale_converted=0,
        num_tolerance=0,
        num_schema_repaired=0,
        trend_false_tp_corrected=0,
        data_name_adjustments=0,
        position_adjustments=0,
    )
    examples: dict[str, list[dict[str, Any]]] = {}
    trend_support = Counter()
    for target in targets:
        for binding in target["targetBindings"]:
            trend_support[trends.classify(binding.get("Trend"))["class"]] += 1

    for target, prediction_record, record in zip(targets, predictions, records):
        raw_result = prediction_record.get("result")
        predicted_bindings = (
            raw_result
            if prediction_record.get("formatValid") and isinstance(raw_result, list)
            else []
        )
        for detail in record["matchDetails"]:
            gold = target["targetBindings"][int(detail["goldIndex"])]
            prediction = predicted_bindings[int(detail["predictionIndex"])]
            previous_binding_correct = bool(detail["bindingCorrect"])

            strict_object = base.ObjectMatcher.equal(
                objects, gold.get("ObjectName"), prediction.get("ObjectName")
            )
            object_equal, object_method = objects.equal_with_method(
                gold.get("ObjectName"), prediction.get("ObjectName")
            )
            detail["fields"]["ObjectName"]["matchMethod"] = object_method
            if object_equal and not strict_object:
                audit["object_article_normalized"] += 1
                _append_example(
                    examples,
                    "object_article_normalized",
                    {
                        "source": target["source"],
                        "gold": gold.get("ObjectName"),
                        "prediction": prediction.get("ObjectName"),
                    },
                )

            strict_num = base.numeric_equal(gold.get("Num"), prediction.get("Num"))
            num_equal, num_method, num_audit = semantic_numeric_equal(
                gold.get("Num"),
                prediction.get("Num"),
                gold.get("DataName"),
                float(
                    (_ACTIVE_NUMERIC_CONFIG or {}).get("mainRelativeTolerance", -1)
                ),
            )
            detail["fields"]["Num"]["matchMethod"] = num_method
            detail["fields"]["Num"]["semanticAudit"] = num_audit
            if num_equal and not strict_num:
                counts["Num"]["tp"] += 1
                counts["Num"]["fp"] -= 1
                counts["Num"]["fn"] -= 1
                detail["fields"]["Num"].update(
                    {"equal": True, "valid": True, "reason": "equal:" + num_method}
                )
                audit_key = {
                    "semantic-normalized": "num_semantic_normalized",
                    "scale-converted": "num_scale_converted",
                    "tolerance": "num_tolerance",
                }[num_method]
                audit[audit_key] += 1
                if num_audit.get("schemaRepair"):
                    audit["num_schema_repaired"] += 1
                _append_example(
                    examples,
                    audit_key,
                    {
                        "source": target["source"],
                        "DataName": gold.get("DataName"),
                        "gold": gold.get("Num"),
                        "prediction": prediction.get("Num"),
                        "schemaRepair": bool(num_audit.get("schemaRepair")),
                    },
                )

            new_binding_correct = all(
                bool(detail["fields"][field]["equal"]) for field in PRIMARY_FIELDS
            )
            if not previous_binding_correct and new_binding_correct:
                binding_counts["tp"] += 1
                binding_counts["fp"] -= 1
                binding_counts["fn"] -= 1
            detail["bindingCorrect"] = new_binding_correct

            legacy_trend_equal = _legacy_trend_class(gold.get("Trend")) == _legacy_trend_class(
                prediction.get("Trend")
            )
            new_trend_equal = bool(detail["fields"]["Trend"]["equal"])
            if legacy_trend_equal and not new_trend_equal:
                audit["trend_false_tp_corrected"] += 1
                _append_example(
                    examples,
                    "trend_false_tp_corrected",
                    {
                        "source": target["source"],
                        "gold": gold.get("Trend"),
                        "prediction": prediction.get("Trend"),
                        "goldClass": trends.classify(gold.get("Trend"))["class"],
                        "predictionClass": trends.classify(prediction.get("Trend"))["class"],
                    },
                )

    summary["primary"] = metrics_from_counts(counts)
    summary["bindingLevel"] = metric(**binding_counts)
    summary["withoutTrendAblation"] = metrics_from_counts(
        {field: counts[field] for field in PRIMARY_FIELDS if field != "Trend"}
    )
    confusion = summary["trend"]["confusionMatrix"]
    summary["trend"]["support"] = {
        label: int(trend_support.get(label, 0)) for label in TREND_CLASSES
    }
    summary["trend"].update(
        trend_metrics_from_confusion(confusion, summary["trend"]["support"])
    )
    summary["methodAudit"] = {
        "counts": dict(audit),
        "examples": examples,
        "unmatchedGoldBindings": int(summary["coverage"].get("gold_bindings", 0))
        - int(summary["coverage"].get("matched_bindings", 0)),
        "unmatchedPredictionBindings": int(summary["coverage"].get("predicted_bindings", 0))
        - int(summary["coverage"].get("matched_bindings", 0)),
    }
    return summary, records


def validate_config(config: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if config.get("protocolId") != PROTOCOL:
        errors.append("protocolId")
    if tuple(config.get("primaryFields") or []) != PRIMARY_FIELDS:
        errors.append("primaryFields")
    if tuple(config.get("trend", {}).get("classes") or []) != TREND_CLASSES:
        errors.append("trend.classes")
    if not config.get("trend", {}).get("rules"):
        errors.append("trend.rules")
    numeric = config.get("numericSemantic", {})
    tolerance = numeric.get("mainRelativeTolerance")
    if tolerance is None or not math.isclose(float(tolerance), 0.0, abs_tol=1e-15):
        errors.append("numericSemantic.mainRelativeTolerance")
    if tuple(numeric.get("reportedSensitivityTolerances") or ()) != (0.001, 0.005, 0.01):
        errors.append("numericSemantic.reportedSensitivityTolerances")
    if numeric.get("schemaValidityReportedSeparately") is not True:
        errors.append("numericSemantic.schemaValidityReportedSeparately")
    evidence = config.get("methodEvidence", {})
    for path_key, hash_key in (
        ("recommendationDocument", "recommendationDocumentSha256"),
        ("candidate12Baseline", "candidate12BaselineSha256"),
    ):
        evidence_path = (base.WORKSPACE_ROOT / str(evidence.get(path_key) or "")).resolve()
        if (
            not evidence_path.is_file()
            or sha256_file(evidence_path) != evidence.get(hash_key)
        ):
            errors.append(f"methodEvidence.{hash_key}")
    if evidence.get("testPredictionTuningForbidden") is not True:
        errors.append("methodEvidence.testPredictionTuningForbidden")
    judge = config.get("textJudge", {})
    if sha256_text(str(judge.get("systemPrompt") or "")) != judge.get(
        "systemPromptSha256"
    ):
        errors.append("textJudge.systemPromptSha256")
    if errors:
        raise ProtocolError("evaluation config mismatch: " + ", ".join(errors))


def method_metadata(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    material = {
        "protocolId": config["protocolId"],
        "primaryFields": config["primaryFields"],
        "alignment": config["alignment"],
        "normalization": config["normalization"],
        "objectName": config["objectName"],
        "trend": config["trend"],
        "numericSemantic": config["numericSemantic"],
        "textJudge": config["textJudge"],
        "counting": config["counting"],
        "aggregation": config["aggregation"],
    }
    compatibility_sha = sha256_text(
        json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    evaluator_sha = sha256_file(Path(__file__).resolve())
    return {
        "protocolId": PROTOCOL,
        "configPath": str(config_path),
        "configSha256": sha256_file(config_path),
        "evaluatorSha256": evaluator_sha,
        "methodCompatibilitySha256": compatibility_sha,
        "methodSha256": sha256_text(compatibility_sha + ":" + evaluator_sha),
        "primaryScoreRole": "diagnostic-method-sensitivity",
        "frozenV4Role": "strict-confirmatory-baseline",
        "textRole": "secondary-only",
    }
