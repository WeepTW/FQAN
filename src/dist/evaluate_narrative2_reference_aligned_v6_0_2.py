#!/usr/bin/env python3
"""Experiment 6 reference-aligned v6.0.2 evaluator.

The evaluator never rewrites generation artifacts.  ``formal`` requires ten
runs per selected case and emits one shared run ranking.  ``diagnostic`` may
evaluate partial runs, but deliberately omits top-1/top-3 selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
PROTOCOL = "experiment6-reference-aligned-v6.0.2"
PRIMARY_FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num")
TREND_CLASSES = (
    "head_and_shoulders",
    "cup_and_handle",
    "rounding_bottom",
    "double_top",
    "triple_top",
    "none",
)
PUNCTUATION_TRANSLATION = str.maketrans({
    "’": "'", "‘": "'", "＇": "'", "“": '"', "”": '"', "＂": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    "，": ",", "．": ".", "：": ":", "；": ";", "％": "%",
    "（": "(", "）": ")", "［": "[", "］": "]", "＆": "&",
})
NUMBER_RE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])"
)
ALLOWED_NUMERIC_WORDS = {
    "usd", "eur", "gbp", "jpy", "sgd", "aud", "cad", "cny", "rmb",
    "dollar", "dollars", "euro", "euros", "yen", "yuan", "pound", "pounds",
    "thousand", "million", "billion", "trillion", "k", "m", "mn", "bn", "tn",
    "percent", "percentage",
    "point", "points", "pct", "pp", "approximately", "about", "around",
}
CURRENCY_SYMBOLS = {"$": "usd", "€": "eur", "£": "gbp", "¥": "jpy"}
SCALE_ALIASES = {
    "thousand": "thousand", "k": "thousand",
    "million": "million", "m": "million", "mn": "million",
    "billion": "billion", "bn": "billion",
    "trillion": "trillion", "tn": "trillion",
}
TREND_CALIBRATION_FIXTURES = (
    ("head_and_shoulders", "head and shoulders"),
    ("cup_and_handle", "cup and handle"),
    ("rounding_bottom", "rounding bottom"),
    ("double_top", "double top"),
    ("triple_top", "triple top"),
    ("head_and_shoulders", "a central head peak flanked by two shoulder peaks"),
    ("head_and_shoulders", "three peaks where the middle peak is the highest"),
    ("cup_and_handle", "a rounded cup followed by a short pullback handle"),
    ("cup_and_handle", "a bowl-shaped recovery with a small handle consolidation"),
    ("rounding_bottom", "a gradual saucer-shaped base and recovery"),
    ("rounding_bottom", "a long rounded base that turns upward"),
    ("double_top", "two similar highs followed by a bearish reversal"),
    ("double_top", "an M-shaped formation with two resistance peaks"),
    ("triple_top", "three failed tests of the same resistance level"),
    ("triple_top", "a three-peak bearish reversal formation"),
    ("none", "revenue increased during the period"),
    ("none", "the series declined steadily"),
    ("none", "the value was unchanged"),
    ("none", "a trough occurred before recovery"),
)


class ProtocolError(RuntimeError):
    """Raised when an input or protocol invariant is violated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).translate(PUNCTUATION_TRANSLATION)
    return " ".join(value.strip().casefold().split())


def object_tokens(value: str) -> tuple[str, ...]:
    normalized = normalized_string(value)
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return tuple(token for token in normalized.split() if token)


def data_name_equal(gold: Any, prediction: Any) -> bool:
    return (
        isinstance(gold, str)
        and isinstance(prediction, str)
        and normalized_string(gold) == normalized_string(prediction)
    )


def canonical_coordinate(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    output: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            number = item
        elif isinstance(item, str) and re.fullmatch(r"[+-]?\d+", item.strip()):
            number = int(item.strip())
        else:
            return None
        if number < 0:
            return None
        output.append(number)
    return output[0], output[1]


def canonical_position(value: Any) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None:
    if not isinstance(value, list):
        return None
    output: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        lowered = {str(key).casefold(): entry for key, entry in item.items()}
        if set(lowered) != {"begin", "end"}:
            return None
        begin = canonical_coordinate(lowered["begin"])
        end = canonical_coordinate(lowered["end"])
        if begin is None or end is None:
            return None
        output.append((begin, end))
    return tuple(output)


@dataclass(frozen=True)
class NumericValue:
    value: float
    unit: str | None
    scale: str | None


def parse_numeric_item(value: Any) -> NumericValue | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return NumericValue(number, None, None) if math.isfinite(number) else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = normalized_string(value)
    matches = list(NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return None
    number = float(matches[0].group(0).replace(",", ""))
    if not math.isfinite(number):
        return None
    remainder = text[:matches[0].start()] + " " + text[matches[0].end():]
    words = set(re.findall(r"[^\W\d_]+", remainder, flags=re.UNICODE))
    if words - ALLOWED_NUMERIC_WORDS:
        return None
    symbols = {symbol for symbol in CURRENCY_SYMBOLS if symbol in remainder}
    currency_words = words & {"usd", "eur", "gbp", "jpy", "sgd", "aud", "cad", "cny", "rmb"}
    currency_units = {CURRENCY_SYMBOLS[item] for item in symbols} | currency_words
    percentage_point = "pp" in words or ({"percentage", "point"} <= words) or ({"percentage", "points"} <= words)
    percent = "%" in remainder or "percent" in words or "pct" in words
    if percentage_point and percent:
        percent = False
    units = set(currency_units)
    if percentage_point:
        units.add("percentage_point")
    elif percent:
        units.add("percent")
    if len(units) > 1:
        return None
    scales = {SCALE_ALIASES[word] for word in words if word in SCALE_ALIASES}
    if len(scales) > 1:
        return None
    cleaned = NUMBER_RE.sub(" ", text, count=1)
    cleaned = re.sub(r"[\s\[\](){}:;,./%$€£¥+\-]", "", cleaned)
    for word in sorted(ALLOWED_NUMERIC_WORDS, key=lambda item: (-len(item), item)):
        cleaned = cleaned.replace(word, "")
    if cleaned:
        return None
    return NumericValue(
        number,
        next(iter(units), None),
        next(iter(scales), None),
    )


def canonical_numbers(value: Any) -> list[NumericValue] | None:
    items = value if isinstance(value, list) else [value]
    output: list[NumericValue] = []
    for item in items:
        parsed = parse_numeric_item(item)
        if parsed is None:
            return None
        output.append(parsed)
    return output


def numeric_item_equal(gold: NumericValue, prediction: NumericValue) -> bool:
    if not math.isclose(gold.value, prediction.value, rel_tol=1e-9, abs_tol=1e-9):
        return False
    if gold.unit is not None and prediction.unit is not None and gold.unit != prediction.unit:
        return False
    if gold.scale is not None and prediction.scale is not None and gold.scale != prediction.scale:
        return False
    return True


def maximum_matching(left_count: int, right_count: int, predicate: Any) -> list[tuple[int, int]]:
    right_match: dict[int, int] = {}

    def visit(left: int, seen: set[int]) -> bool:
        for right in range(right_count):
            if right in seen or not predicate(left, right):
                continue
            seen.add(right)
            if right not in right_match or visit(right_match[right], seen):
                right_match[right] = left
                return True
        return False

    for left in range(left_count):
        visit(left, set())
    return sorted((left, right) for right, left in right_match.items())


def numeric_equal(gold: Any, prediction: Any) -> bool:
    left = canonical_numbers(gold)
    right = canonical_numbers(prediction)
    if left is None or right is None or len(left) != len(right):
        return False
    return len(maximum_matching(len(left), len(right), lambda i, j: numeric_item_equal(left[i], right[j]))) == len(left)


class ObjectMatcher:
    def __init__(self, config: Mapping[str, Any]):
        self.groups: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        for canonical, aliases in (config.get("aliases") or {}).items():
            canonical_tokens = object_tokens(str(canonical))
            variants = {canonical_tokens}
            variants.update(object_tokens(str(alias)) for alias in aliases)
            self.groups[canonical_tokens] = {item for item in variants if item}

    @staticmethod
    def contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
        if not needle or len(needle) > len(haystack):
            return False
        return any(haystack[index:index + len(needle)] == needle for index in range(len(haystack) - len(needle) + 1))

    def mention_equal(self, gold: str, prediction: str) -> bool:
        gold_tokens = object_tokens(gold)
        prediction_tokens = object_tokens(prediction)
        if not gold_tokens or not prediction_tokens:
            return False
        for canonical, variants in self.groups.items():
            gold_in_group = any(self.contains(gold_tokens, item) for item in variants)
            prediction_in_group = any(self.contains(prediction_tokens, item) for item in variants)
            if gold_in_group and prediction_in_group:
                return True
        return self.contains(prediction_tokens, gold_tokens) or self.contains(gold_tokens, prediction_tokens)

    @staticmethod
    def mentions(value: Any) -> list[str] | None:
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or not items:
            return None
        if not all(isinstance(item, str) and normalized_string(item) for item in items):
            return None
        return items

    def equal(self, gold: Any, prediction: Any) -> bool:
        left = self.mentions(gold)
        right = self.mentions(prediction)
        if left is None or right is None or len(left) != len(right):
            return False
        return len(maximum_matching(len(left), len(right), lambda i, j: self.mention_equal(left[i], right[j]))) == len(left)


class TrendClassifier:
    def __init__(self, config: Mapping[str, Any], allow_model: bool = True):
        self.config = config
        self.threshold = float(config["similarityThreshold"])
        self.alias_to_class: dict[str, str] = {}
        for label, aliases in config["aliases"].items():
            for alias in aliases:
                self.alias_to_class[normalized_string(alias)] = label
        self.none_aliases = {normalized_string(item) for item in config["noneAliases"]}
        self.prototypes = dict(config["prototypes"])
        self.allow_model = allow_model
        self._model: Any = None
        self._prototype_vectors: Any = None

    def deterministic(self, value: Any) -> str | None:
        if value is None:
            return "none"
        if not isinstance(value, str):
            return None
        normalized = normalized_string(value)
        if normalized in self.none_aliases:
            return "none"
        return self.alias_to_class.get(normalized)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.allow_model:
            raise ProtocolError("Sentence-BERT is required for an unknown Trend string")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ProtocolError("PyTorch/Transformers is unavailable") from error
        model_config = self.config["sentenceModel"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["modelId"],
            revision=model_config["revision"],
            local_files_only=bool(model_config["localFilesOnly"]),
        )
        model = AutoModel.from_pretrained(
            model_config["modelId"],
            revision=model_config["revision"],
            local_files_only=bool(model_config["localFilesOnly"]),
        )
        model.to("cpu")
        model.eval()

        def encode(texts: Sequence[str]) -> Any:
            encoded = tokenizer(
                list(texts), padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {name: value.to("cpu") for name, value in encoded.items()}
            with torch.inference_mode():
                token_embeddings = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            vectors = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(vectors, p=2, dim=1)

        self._model = encode
        self._prototype_vectors = encode([self.prototypes[label] for label in self.prototypes])

    def similarity(self, value: str) -> tuple[str, float]:
        self._load_model()
        vector = self._model([value])[0]
        scores = self._prototype_vectors @ vector
        index = max(range(len(scores)), key=lambda item: float(scores[item]))
        return list(self.prototypes)[index], float(scores[index])

    def classify(self, value: Any, threshold: float | None = None) -> dict[str, Any]:
        deterministic = self.deterministic(value)
        if deterministic is not None:
            return {"class": deterministic, "method": "alias", "similarity": None}
        if not isinstance(value, str):
            return {"class": None, "method": "invalid", "similarity": None}
        label, score = self.similarity(value)
        active_threshold = self.threshold if threshold is None else threshold
        return {
            "class": label if score >= active_threshold else "none",
            "nearestClass": label,
            "method": "sentence-bert",
            "similarity": score,
            "threshold": active_threshold,
        }


def calibrate_trend(classifier: TrendClassifier) -> dict[str, Any]:
    cached: list[tuple[str, str, float | None, str]] = []
    for gold, text in TREND_CALIBRATION_FIXTURES:
        deterministic = classifier.deterministic(text)
        if deterministic is not None:
            cached.append((gold, deterministic, None, text))
        else:
            nearest, similarity = classifier.similarity(text)
            cached.append((gold, nearest, similarity, text))
    candidates: list[dict[str, Any]] = []
    for threshold in classifier.config["calibration"]["candidateThresholds"]:
        truth_by_class = {label: {"tp": 0, "fp": 0, "fn": 0} for label in TREND_CLASSES}
        for gold, nearest, similarity, _ in cached:
            predicted = nearest if similarity is None or similarity >= threshold else "none"
            for label in TREND_CLASSES:
                if gold == label and predicted == label:
                    truth_by_class[label]["tp"] += 1
                elif gold == label:
                    truth_by_class[label]["fn"] += 1
                elif predicted == label:
                    truth_by_class[label]["fp"] += 1
        values = [metric(**truth_by_class[label])["f1"] for label in TREND_CLASSES]
        macro_f1 = statistics.fmean(value for value in values if value is not None)
        candidates.append({"threshold": threshold, "macroF1": macro_f1, "counts": truth_by_class})
    selected = max(candidates, key=lambda item: (item["macroF1"], item["threshold"]))
    return {
        "fixtures": len(cached),
        "fixtureSha256": sha256_text(json.dumps(TREND_CALIBRATION_FIXTURES, ensure_ascii=False)),
        "model": classifier.config["sentenceModel"],
        "candidates": candidates,
        "selectedThreshold": selected["threshold"],
        "selectionMetric": "macro-f1-over-six-classes",
        "testPredictionsInspected": False,
    }


def metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def zero_counts(fields: Sequence[str] = PRIMARY_FIELDS) -> dict[str, dict[str, int]]:
    return {field: {"tp": 0, "fp": 0, "fn": 0} for field in fields}


def add_counts(target: dict[str, dict[str, int]], source: Mapping[str, Mapping[str, int]]) -> None:
    for field in target:
        for name in ("tp", "fp", "fn"):
            target[field][name] += int(source[field][name])


def metrics_from_counts(counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    fields = {field: metric(**value) for field, value in counts.items()}
    macro: dict[str, Any] = {}
    for name in ("precision", "recall", "f1"):
        values = [item[name] for item in fields.values() if item[name] is not None]
        macro[name] = statistics.fmean(values) if values else None
    pooled = {name: sum(int(item[name]) for item in counts.values()) for name in ("tp", "fp", "fn")}
    return {"counts": counts, "fields": fields, "macro": macro, "micro": metric(**pooled)}


def anchor_key(binding: Any) -> tuple[str, Any] | None:
    if not isinstance(binding, Mapping):
        return None
    data_name = binding.get("DataName")
    position = canonical_position(binding.get("Position"))
    if not isinstance(data_name, str) or position is None:
        return None
    return normalized_string(data_name), position


def align_bindings(gold: Sequence[Mapping[str, Any]], prediction: Sequence[Any]) -> dict[str, Any]:
    gold_groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    prediction_groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for index, binding in enumerate(gold):
        key = anchor_key(binding)
        if key is None:
            raise ProtocolError(f"gold binding {index} has invalid DataName/Position anchor")
        gold_groups[key].append(index)
    invalid_prediction: list[int] = []
    for index, binding in enumerate(prediction):
        key = anchor_key(binding)
        if key is None:
            invalid_prediction.append(index)
        else:
            prediction_groups[key].append(index)
    matches: list[dict[str, int]] = []
    used_gold: set[int] = set()
    used_prediction: set[int] = set()
    ambiguity: list[dict[str, Any]] = []
    for key in sorted(set(gold_groups) | set(prediction_groups), key=repr):
        gold_indices = sorted(gold_groups.get(key, []))
        prediction_indices = sorted(prediction_groups.get(key, []))
        if len(gold_indices) > 1 or len(prediction_indices) > 1:
            ambiguity.append({
                "anchor": {"DataName": key[0], "Position": key[1]},
                "goldIndices": gold_indices,
                "predictionIndices": prediction_indices,
            })
        for gold_index, prediction_index in zip(gold_indices, prediction_indices):
            matches.append({"goldIndex": gold_index, "predictionIndex": prediction_index})
            used_gold.add(gold_index)
            used_prediction.add(prediction_index)
    return {
        "matches": sorted(matches, key=lambda item: (item["goldIndex"], item["predictionIndex"])),
        "unmatchedGold": [index for index in range(len(gold)) if index not in used_gold],
        "unmatchedPrediction": [index for index in range(len(prediction)) if index not in used_prediction],
        "invalidAnchorPrediction": invalid_prediction,
        "ambiguity": ambiguity,
    }


def field_comparison(
    field: str,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    objects: ObjectMatcher,
    trends: TrendClassifier,
) -> dict[str, Any]:
    if field not in prediction:
        return {"present": False, "valid": False, "equal": False, "reason": "missing"}
    value = prediction[field]
    if field == "DataName":
        valid = isinstance(value, str)
        equal = valid and data_name_equal(gold[field], value)
        normalized = normalized_string(value) if valid else None
    elif field == "Position":
        normalized = canonical_position(value)
        valid = normalized is not None
        equal = valid and normalized == canonical_position(gold[field])
    elif field == "ObjectName":
        normalized = ObjectMatcher.mentions(value)
        valid = normalized is not None
        equal = valid and objects.equal(gold[field], value)
    elif field == "Num":
        parsed = canonical_numbers(value)
        normalized = None if parsed is None else [item.__dict__ for item in parsed]
        valid = parsed is not None
        equal = valid and numeric_equal(gold[field], value)
    elif field == "Trend":
        prediction_class = trends.classify(value)
        gold_class = trends.classify(gold[field])
        normalized = prediction_class
        valid = prediction_class["class"] is not None
        equal = valid and prediction_class["class"] == gold_class["class"]
        return {
            "present": True,
            "valid": valid,
            "equal": equal,
            "goldNormalized": gold_class,
            "predictionNormalized": prediction_class,
            "reason": "equal" if equal else ("invalid" if not valid else "different"),
        }
    else:
        raise KeyError(field)
    return {
        "present": True,
        "valid": valid,
        "equal": equal,
        "predictionNormalized": normalized,
        "reason": "equal" if equal else ("invalid" if not valid else "different"),
    }


def evaluate_rows(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    objects: ObjectMatcher,
    trends: TrendClassifier,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = zero_counts()
    binding_counts = {"tp": 0, "fp": 0, "fn": 0}
    coverage = Counter()
    trend_support = Counter()
    gold_ordinary_direction_count = 0
    confusion = defaultdict(Counter)
    raw_direction_pairs = 0
    raw_direction_mismatches = 0
    records: list[dict[str, Any]] = []
    for target, prediction_record in zip(targets, predictions):
        gold_bindings = target["targetBindings"]
        for binding in gold_bindings:
            gold_trend_class = trends.classify(binding["Trend"])["class"]
            trend_support[gold_trend_class] += 1
            raw_gold_trend = normalized_string(binding.get("Trend"))
            if gold_trend_class == "none" and raw_gold_trend not in {"", "none", "null", "n/a", "na"}:
                gold_ordinary_direction_count += 1
        raw_result = prediction_record.get("result")
        format_valid = bool(prediction_record.get("formatValid"))
        if format_valid and isinstance(raw_result, list):
            predicted_bindings = raw_result
        else:
            predicted_bindings = []
            coverage["format_invalid_rows"] += 1
            raw_response = prediction_record.get("rawResponse")
            raw_response_nonempty = (
                bool(raw_response.strip())
                if isinstance(raw_response, str)
                else raw_response not in (None, "", [], {})
            )
            if raw_result not in (None, "", [], {}) or raw_response_nonempty:
                binding_counts["fp"] += 1
                coverage["unparseable_nonempty_binding_fp"] += 1
        if prediction_record.get("parserDiagnostic", {}).get("error") == "runtime_blocked":
            coverage["runtime_blocked_rows"] += 1
        if not predicted_bindings:
            coverage["empty_output_rows"] += 1
        else:
            coverage["nonempty_output_rows"] += 1
        alignment = align_bindings(gold_bindings, predicted_bindings)
        coverage["gold_bindings"] += len(gold_bindings)
        coverage["predicted_bindings"] += len(predicted_bindings)
        coverage["matched_bindings"] += len(alignment["matches"])
        coverage["ambiguous_anchors"] += len(alignment["ambiguity"])
        details: list[dict[str, Any]] = []
        for pair in alignment["matches"]:
            gold = gold_bindings[pair["goldIndex"]]
            prediction = predicted_bindings[pair["predictionIndex"]]
            if not isinstance(prediction, Mapping):
                continue
            field_results: dict[str, Any] = {}
            all_correct = True
            for field in PRIMARY_FIELDS:
                result = field_comparison(field, gold, prediction, objects, trends)
                field_results[field] = result
                if result["equal"]:
                    counts[field]["tp"] += 1
                elif result["present"]:
                    counts[field]["fp"] += 1
                    counts[field]["fn"] += 1
                    all_correct = False
                else:
                    counts[field]["fn"] += 1
                    all_correct = False
            binding_counts["tp" if all_correct else "fp"] += 1
            if not all_correct:
                binding_counts["fn"] += 1
            gold_trend = field_results["Trend"]["goldNormalized"]["class"]
            prediction_trend = field_results["Trend"]["predictionNormalized"]["class"]
            confusion[gold_trend][prediction_trend or "__invalid__"] += 1
            raw_gold = normalized_string(gold.get("Trend"))
            raw_prediction = normalized_string(prediction.get("Trend"))
            if gold_trend == "none" and raw_gold not in {"", "none", "null", "n/a", "na"}:
                raw_direction_pairs += 1
                if raw_gold != raw_prediction:
                    raw_direction_mismatches += 1
            details.append({**pair, "fields": field_results, "bindingCorrect": all_correct})
        for gold_index in alignment["unmatchedGold"]:
            for field in PRIMARY_FIELDS:
                counts[field]["fn"] += 1
            binding_counts["fn"] += 1
            gold_class = trends.classify(gold_bindings[gold_index]["Trend"])["class"]
            confusion[gold_class]["__missing__"] += 1
        for prediction_index in alignment["unmatchedPrediction"]:
            prediction = predicted_bindings[prediction_index]
            binding_counts["fp"] += 1
            if isinstance(prediction, Mapping):
                for field in PRIMARY_FIELDS:
                    if field in prediction:
                        counts[field]["fp"] += 1
                if "Trend" in prediction:
                    prediction_class = trends.classify(prediction["Trend"])["class"]
                    confusion["__spurious__"][prediction_class or "__invalid__"] += 1
            elif prediction not in (None, ""):
                coverage["unparseable_nonempty_binding_fp"] += 1
        for detail in details:
            gold_binding = gold_bindings[detail["goldIndex"]]
            predicted_binding = predicted_bindings[detail["predictionIndex"]]
            detail["textCandidate"] = {
                "gold": gold_binding.get("Text"),
                "prediction": predicted_binding.get("Text") if isinstance(predicted_binding, Mapping) else None,
                "evidence": {
                    "inputData": prediction_record.get("inputData"),
                    "inputText": prediction_record.get("inputText"),
                },
            }
        records.append({
            "source": target["source"],
            "formatValid": format_valid,
            "goldBindings": len(gold_bindings),
            "predictedBindings": len(predicted_bindings),
            "alignment": alignment,
            "matchDetails": details,
        })
    primary = metrics_from_counts(counts)
    without_trend = metrics_from_counts({field: counts[field] for field in PRIMARY_FIELDS if field != "Trend"})
    summary = {
        "primary": primary,
        "bindingLevel": metric(**binding_counts),
        "withoutTrendAblation": without_trend,
        "coverage": dict(coverage),
        "trend": {
            "support": {label: trend_support.get(label, 0) for label in TREND_CLASSES},
            "confusionMatrix": {row: dict(columns) for row, columns in confusion.items()},
            "goldOrdinaryDirectionCount": gold_ordinary_direction_count,
            "goldOrdinaryDirectionRate": (
                gold_ordinary_direction_count / sum(trend_support.values())
                if trend_support else None
            ),
            "alignedRawDirectionPairs": raw_direction_pairs,
            "alignedRawDirectionMismatchRate": (
                raw_direction_mismatches / raw_direction_pairs if raw_direction_pairs else None
            ),
        },
    }
    return summary, records


def resolve_artifact(manifest: Mapping[str, Any], name: str, generation_root: Path) -> Path:
    raw = str((manifest.get("files") or {}).get(name) or "")
    declared = Path(raw)
    if declared.is_file():
        return declared
    relocated = generation_root / "cases" / str(manifest["outputId"]) / f"run_{int(manifest['run']):02d}" / declared.name
    return relocated


def load_predictions(
    manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    generation_root: Path,
) -> tuple[list[dict[str, Any]], Path]:
    path = resolve_artifact(manifest, "predictions", generation_root)
    if not path.is_file():
        if manifest.get("status") == "runtime_blocked":
            return [
                {
                    "source": target["source"], "result": [], "formatValid": False,
                    "parserDiagnostic": {"error": "runtime_blocked"},
                }
                for target in targets
            ], path
        raise ProtocolError(f"prediction file missing: {path}")
    expected_hash = (manifest.get("hashes") or {}).get("predictions")
    if sha256_file(path) != expected_hash:
        raise ProtocolError(f"prediction SHA-256 mismatch: {path}")
    predictions = read_jsonl(path)
    expected_sources = [str(item["source"]) for item in targets]
    actual_sources = [str(item.get("source") or "") for item in predictions]
    if actual_sources != expected_sources:
        raise ProtocolError(f"source order or coverage mismatch: {manifest['outputId']} run {manifest['run']}")
    return predictions, path


def aggregate_values(values: Sequence[float | None], t_critical: float | None = None) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not defined:
        return {"n": 0, "mean": None, "sampleSd": None, "ci95": None, "min": None, "max": None}
    mean = statistics.fmean(defined)
    sample_sd = statistics.stdev(defined) if len(defined) > 1 else None
    ci = None
    if sample_sd is not None and t_critical is not None:
        half = t_critical * sample_sd / math.sqrt(len(defined))
        ci = [mean - half, mean + half]
    return {"n": len(defined), "mean": mean, "sampleSd": sample_sd, "ci95": ci, "min": min(defined), "max": max(defined)}


def complete_mean(values: Sequence[float | None]) -> float | None:
    materialized = list(values)
    if not materialized or any(
        value is None or not math.isfinite(float(value))
        for value in materialized
    ):
        return None
    return statistics.fmean(float(value) for value in materialized)


def format_metric_value(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.6f}"


def common_run_order(run_results: Sequence[Mapping[str, Any]]) -> list[int]:
    def score(result: Mapping[str, Any]) -> tuple[float, float, float, int]:
        macro = result["primary"]["macro"]
        return (
            -(macro["f1"] if macro["f1"] is not None else -math.inf),
            -(macro["precision"] if macro["precision"] is not None else -math.inf),
            -(macro["recall"] if macro["recall"] is not None else -math.inf),
            int(result["run"]),
        )
    return [int(item["run"]) for item in sorted(run_results, key=score)]


def aggregate_case(
    output_id: str,
    runs: Sequence[Mapping[str, Any]],
    mode: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    t_critical = float(config["aggregation"]["studentTCriticalForTenRuns"]) if mode == "formal" else None
    ranking = common_run_order(runs) if mode == "formal" else None
    selected_top3 = ranking[: int(config["aggregation"]["topK"])] if ranking else None
    pooled_counts = zero_counts()
    for run in runs:
        add_counts(pooled_counts, run["primary"]["counts"])
    fields: dict[str, Any] = {}
    for field in PRIMARY_FIELDS:
        fields[field] = {
            name: aggregate_values([run["primary"]["fields"][field][name] for run in runs], t_critical)
            for name in ("precision", "recall", "f1")
        }
    macro = {
        name: aggregate_values([run["primary"]["macro"][name] for run in runs], t_critical)
        for name in ("precision", "recall", "f1")
    }
    micro = {
        name: aggregate_values([run["primary"]["micro"][name] for run in runs], t_critical)
        for name in ("precision", "recall", "f1")
    }
    by_id = {int(run["run"]): run for run in runs}
    descriptive = None
    if ranking:
        top1 = by_id[ranking[0]]
        top3 = [by_id[run_id] for run_id in selected_top3]
        descriptive = {
            "sharedRunOrder": ranking,
            "top1": {
                "run": ranking[0],
                "macro": top1["primary"]["macro"],
                "fields": top1["primary"]["fields"],
            },
            "top3": {
                "runs": selected_top3,
                "macro": {
                    name: complete_mean(
                        [item["primary"]["macro"][name] for item in top3]
                    )
                    for name in ("precision", "recall", "f1")
                },
                "fields": {
                    field: {
                        name: complete_mean(
                            [
                                item["primary"]["fields"][field][name]
                                for item in top3
                            ]
                        )
                        for name in ("precision", "recall", "f1")
                    }
                    for field in PRIMARY_FIELDS
                },
            },
            "role": "descriptive-only",
        }
    return {
        "outputId": output_id,
        "mode": mode,
        "runs": len(runs),
        "runResults": list(sorted(runs, key=lambda item: int(item["run"]))),
        "aggregate": {"fields": fields, "macro": macro, "micro": micro, "pooled": metrics_from_counts(pooled_counts)},
        "selection": descriptive,
    }


def paired_bootstrap(
    current: Mapping[int, float],
    baseline: Mapping[int, float],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    shared = sorted(set(current) & set(baseline))
    if not shared:
        return {"status": "NA", "reason": "no-shared-runs"}
    differences = [current[run] - baseline[run] for run in shared]
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(replicates)
    )
    lower = samples[int(0.025 * (replicates - 1))]
    upper = samples[int(0.975 * (replicates - 1))]
    return {
        "status": "completed",
        "sharedRuns": shared,
        "meanDifference": statistics.fmean(differences),
        "percentile95CI": [lower, upper],
        "replicates": replicates,
        "seed": seed,
    }


def load_baseline_metrics(path: Path, output_id: str) -> tuple[str, dict[int, dict[str, float]]]:
    if path.suffix.casefold() == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("output_id") == output_id]
        return "v4-six-field-macro", {
            int(row["run"]): {name: float(row[f"macro_{name}"]) for name in ("precision", "recall", "f1")}
            for row in rows
        }
    report = read_json(path)
    case = next((item for item in report.get("cases", []) if item.get("outputId") == output_id), None)
    if case is None:
        return str(report.get("protocol") or "unknown"), {}
    return str(report.get("protocol") or "v5-six-field-macro"), {
        int(item["run"]): {name: float(item["macro"][name]) for name in ("precision", "recall", "f1")}
        for item in case.get("runResults", [])
    }


def compare_case(
    case: Mapping[str, Any],
    specifications: Sequence[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    current = {
        int(item["run"]): {name: item["primary"]["macro"][name] for name in ("precision", "recall", "f1")}
        for item in case["runResults"]
    }
    output: list[dict[str, Any]] = []
    for specification in specifications:
        label, raw_path = specification.split("=", 1)
        protocol, baseline = load_baseline_metrics(Path(raw_path).resolve(), str(case["outputId"]))
        comparisons = {}
        for name in ("precision", "recall", "f1"):
            comparisons[name] = paired_bootstrap(
                {run: values[name] for run, values in current.items() if values[name] is not None},
                {run: values[name] for run, values in baseline.items()},
                int(config["aggregation"]["bootstrapReplicates"]),
                int(config["aggregation"]["bootstrapSeed"]),
            )
        output.append({
            "label": label,
            "baselineProtocol": protocol,
            "comparability": "descriptive-only-estimand-changed",
            "reason": "v4/v5 use six fields and different error policies; v6 uses five fields and symmetric FP+FN penalties",
            "pairedBootstrap": comparisons,
        })
    return output


def normalize_text_score(prediction: Any, gold: Any, anchor: Any) -> float | None:
    try:
        values = [float(prediction), float(gold), float(anchor)]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values) or values[1] <= values[2]:
        return None
    return min(100.0, max(0.0, 100.0 * (values[0] - values[2]) / (values[1] - values[2])))


class TextQualityJudge:
    """Optional ChatMock judge; disabled by default and never affects TP/FP/FN."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.live_calls = 0

    def _request(self, messages: list[dict[str, str]]) -> Mapping[str, Any]:
        url = self.config["baseUrl"].rstrip("/") + "/chat/completions"
        key = os.environ.get(self.config["apiKeyEnvironment"], self.config["defaultApiKey"])
        payload = json.dumps({
            "model": self.config["model"],
            "reasoning_effort": self.config["reasoningEffort"],
            "messages": messages,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        delays = [0, 5, 15]
        last_error: Exception | None = None
        for attempt in range(int(self.config["maxAttempts"])):
            if delays[attempt]:
                time.sleep(delays[attempt])
            request = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=int(self.config["requestTimeoutSeconds"])) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self.live_calls += 1
                return json.loads(body["choices"][0]["message"]["content"])
            except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as error:
                last_error = error
        raise ProtocolError(f"Text judge unavailable or invalid: {last_error}")

    def score(self, evidence: str, items_by_role: Mapping[str, str], seed_material: str) -> dict[str, Any]:
        roles = list(items_by_role)
        random.Random(int(sha256_text(seed_material)[:16], 16)).shuffle(roles)
        labels = [chr(ord("A") + index) for index in range(len(roles))]
        role_by_label = dict(zip(labels, roles))
        prompt = "Source evidence:\n" + evidence + "\n\n" + "\n\n".join(
            f"Text {label}:\n{items_by_role[role_by_label[label]]}" for label in labels
        )
        raw = self._request([
            {"role": "system", "content": self.config["systemPrompt"]},
            {"role": "user", "content": prompt},
        ])
        returned = {str(item.get("label")): item for item in raw.get("items", []) if isinstance(item, Mapping)}
        if set(returned) != set(labels):
            raise ProtocolError("Text judge did not return exactly the randomized labels")
        scores: dict[str, Any] = {}
        for label, role in role_by_label.items():
            item = returned[label]
            factual = float(item["factual_consistency"])
            fluency = float(item["natural_fluency"])
            if not all(math.isfinite(value) and 0 <= value <= 100 for value in (factual, fluency)):
                raise ProtocolError("Text judge score outside 0..100")
            scores[role] = {"factualConsistency": factual, "naturalFluency": fluency, "reason": str(item.get("reason") or "")}
        return {"presentation": role_by_label, "scores": scores, "raw": raw}


def probe_text_judge(config: Mapping[str, Any]) -> tuple[bool, str]:
    url = config["baseUrl"].rstrip("/") + "/models"
    key = os.environ.get(config["apiKeyEnvironment"], config["defaultApiKey"])
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, Mapping):
            return False, "models endpoint returned non-object JSON"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        return False, str(error)
    return True, "resource endpoint reachable"


def anchor_text_map(
    generation_root: Path,
    all_manifests: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    anchor_case: str,
    run_id: int,
) -> dict[tuple[str, int], str]:
    manifest = next((
        item for item in all_manifests
        if str(item.get("outputId")) == anchor_case and int(item.get("run", -1)) == run_id
    ), None)
    if manifest is None:
        return {}
    predictions, _ = load_predictions(manifest, targets, generation_root)
    output: dict[tuple[str, int], str] = {}
    for target, prediction_record in zip(targets, predictions):
        raw = prediction_record.get("result")
        predicted = raw if prediction_record.get("formatValid") and isinstance(raw, list) else []
        alignment = align_bindings(target["targetBindings"], predicted)
        for pair in alignment["matches"]:
            binding = predicted[pair["predictionIndex"]]
            if isinstance(binding, Mapping) and isinstance(binding.get("Text"), str):
                output[(str(target["source"]), int(pair["goldIndex"]))] = binding["Text"]
    return output


def evaluate_text_secondary(
    output_id: str,
    run_id: int,
    records: Sequence[Mapping[str, Any]],
    anchors: Mapping[tuple[str, int], str],
    config: Mapping[str, Any],
    limit: int,
) -> dict[str, Any]:
    judge = TextQualityJudge(config)
    candidates: list[dict[str, Any]] = []
    for record in records:
        for detail in record.get("matchDetails", []):
            candidate = detail.get("textCandidate") or {}
            if isinstance(candidate.get("gold"), str) and isinstance(candidate.get("prediction"), str):
                candidates.append({
                    "source": record["source"],
                    "goldIndex": int(detail["goldIndex"]),
                    **candidate,
                })
    candidates.sort(key=lambda item: (item["source"], item["goldIndex"]))
    available_candidates = len(candidates)
    if limit > 0:
        candidates = candidates[:limit]
    judgments: list[dict[str, Any]] = []
    anchor_case = str(config["anchorCase"])
    for candidate in candidates:
        key = (str(candidate["source"]), int(candidate["goldIndex"]))
        anchor_text = anchors.get(key)
        items = {"gold": candidate["gold"], "prediction": candidate["prediction"]}
        same_as_anchor = output_id == anchor_case
        if anchor_text is not None and not same_as_anchor:
            items["flan_raw"] = anchor_text
        evidence = "Input data:\n" + str(candidate["evidence"].get("inputData") or "")
        evidence += "\n\nSource narrative:\n" + str(candidate["evidence"].get("inputText") or "")
        decision = judge.score(evidence, items, f"{candidate['source']}:{candidate['goldIndex']}:{run_id}")
        scores = decision["scores"]
        anchor_scores = scores["prediction"] if same_as_anchor else scores.get("flan_raw")
        normalized: dict[str, Any] = {}
        for output_name, internal_name in (
            ("factualConsistency", "factualConsistency"),
            ("naturalFluency", "naturalFluency"),
        ):
            normalized[output_name] = normalize_text_score(
                scores["prediction"][internal_name],
                scores["gold"][internal_name],
                anchor_scores[internal_name] if anchor_scores is not None else None,
            )
        judgments.append({
            "source": candidate["source"],
            "goldIndex": candidate["goldIndex"],
            "model": config["model"],
            "reasoningEffort": config["reasoningEffort"],
            "anchorCase": anchor_case,
            "anchorAvailable": anchor_scores is not None,
            "presentation": decision["presentation"],
            "rawScores": scores,
            "normalized": normalized,
            "rawDecision": decision["raw"],
        })
    dimensions = ("factualConsistency", "naturalFluency")
    return {
        "status": "completed" if judgments else "NA",
        "sampledBindings": len(judgments),
        "availableCandidates": available_candidates,
        "liveCalls": judge.live_calls,
        "rawPredictionMean": {
            name: aggregate_values([
                item["rawScores"]["prediction"][name] for item in judgments
            ])["mean"] for name in dimensions
        },
        "normalizedMean": {
            name: aggregate_values([item["normalized"][name] for item in judgments])["mean"]
            for name in dimensions
        },
        "judgments": judgments,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if config.get("protocolId") != PROTOCOL:
        errors.append("protocolId")
    if tuple(config.get("primaryFields") or []) != PRIMARY_FIELDS:
        errors.append("primaryFields")
    if tuple(config.get("trend", {}).get("classes") or []) != TREND_CLASSES:
        errors.append("trend.classes")
    model = config.get("trend", {}).get("sentenceModel", {})
    if model.get("modelId") != "sentence-transformers/all-MiniLM-L6-v2":
        errors.append("trend.sentenceModel.modelId")
    if model.get("revision") != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41":
        errors.append("trend.sentenceModel.revision")
    judge = config.get("textJudge", {})
    if sha256_text(str(judge.get("systemPrompt") or "")) != judge.get("systemPromptSha256"):
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
        "textJudge": config["textJudge"],
        "counting": config["counting"],
        "aggregation": config["aggregation"],
    }
    compatibility_sha = sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    evaluator_sha = sha256_file(Path(__file__).resolve())
    return {
        "protocolId": PROTOCOL,
        "configPath": str(config_path),
        "configSha256": sha256_file(config_path),
        "evaluatorSha256": evaluator_sha,
        "methodCompatibilitySha256": compatibility_sha,
        "methodSha256": sha256_text(compatibility_sha + ":" + evaluator_sha),
        "primaryScoreRole": "unassigned-version-comparison",
        "frozenV4Role": "strict-confirmatory-baseline",
        "textRole": "secondary-only",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    if args.calibrate_trend:
        result = calibrate_trend(TrendClassifier(config["trend"], allow_model=True))
        if args.calibration_output:
            write_json(args.calibration_output.resolve(), result)
        return {"protocol": PROTOCOL, "status": "calibrated", "trendCalibration": result, "time": utc_now()}
    generation_root = args.output_root.resolve()
    evaluation_root = (
        args.evaluation_root.resolve()
        if args.evaluation_root
        else generation_root / f"evaluation_reference_aligned_v6_{args.mode}"
    )
    gold_path = (REPO_ROOT / config["goldPath"]).resolve()
    if not gold_path.is_file() or sha256_file(gold_path) != config["goldSha256"]:
        raise ProtocolError(f"gold file missing or SHA mismatch: {gold_path}")
    targets = read_json(gold_path).get("rows")
    if not isinstance(targets, list) or len(targets) != int(config["expectedRows"]):
        raise ProtocolError("gold row count mismatch")
    manifest_paths = sorted((generation_root / "manifests").glob("*.json"))
    all_manifests = [read_json(path) for path in manifest_paths]
    all_manifests = [item for item in all_manifests if item.get("official")]
    manifests = list(all_manifests)
    if args.only_case:
        selected = set(args.only_case)
        manifests = [item for item in manifests if str(item.get("outputId")) in selected]
        missing = selected - {str(item.get("outputId")) for item in manifests}
        if missing:
            raise ProtocolError(f"selected cases missing: {sorted(missing)}")
    if not manifests:
        raise ProtocolError("no official manifests selected")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        grouped[str(manifest["outputId"])].append(manifest)
    expected_runs = list(range(1, int(config["expectedRuns"]) + 1))
    coverage = {
        output_id: sorted(int(item["run"]) for item in items)
        for output_id, items in grouped.items()
    }
    if args.mode == "formal":
        incomplete = {key: runs for key, runs in coverage.items() if runs != expected_runs}
        if incomplete:
            raise ProtocolError(f"formal mode requires runs 1..10 per case: {incomplete}")
    objects = ObjectMatcher(config["objectName"])
    trends = TrendClassifier(config["trend"], allow_model=not args.no_sentence_model)
    run_results: list[dict[str, Any]] = []
    records_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    input_artifacts: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda item: (str(item["outputId"]), int(item["run"]))):
        predictions, prediction_path = load_predictions(manifest, targets, generation_root)
        summary, records = evaluate_rows(targets, predictions, objects, trends)
        run_dir = evaluation_root / "cases" / str(manifest["outputId"]) / f"run_{int(manifest['run']):02d}"
        result = {
            "outputId": manifest["outputId"],
            "run": int(manifest["run"]),
            "seed": manifest.get("seed"),
            "generationStatus": manifest.get("status"),
            "predictionPath": str(prediction_path),
            "predictionSha256": sha256_file(prediction_path) if prediction_path.is_file() else None,
            **summary,
        }
        write_jsonl(run_dir / "records.jsonl", records)
        write_json(run_dir / "metrics.json", result)
        run_results.append(result)
        records_by_run[(str(manifest["outputId"]), int(manifest["run"]))] = records
        manifest_path = next(
            path for path in manifest_paths
            if read_json(path).get("outputId") == manifest["outputId"] and int(read_json(path).get("run", -1)) == int(manifest["run"])
        )
        input_artifacts.extend([
            {"kind": "manifest", "path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            {"kind": "predictions", "path": str(prediction_path), "sha256": result["predictionSha256"]},
        ])
    text_status: dict[str, Any]
    if args.text_judge == "enabled":
        available, probe_reason = probe_text_judge(config["textJudge"])
        if available:
            text_runs: list[dict[str, Any]] = []
            for result in run_results:
                output_id = str(result["outputId"])
                run_id = int(result["run"])
                anchors = anchor_text_map(
                    generation_root,
                    all_manifests,
                    targets,
                    str(config["textJudge"]["anchorCase"]),
                    run_id,
                )
                text_result = evaluate_text_secondary(
                    output_id,
                    run_id,
                    records_by_run[(output_id, run_id)],
                    anchors,
                    config["textJudge"],
                    args.text_judge_limit,
                )
                text_result.update({"outputId": output_id, "run": run_id})
                text_runs.append(text_result)
                text_path = evaluation_root / "cases" / output_id / f"run_{run_id:02d}" / "text_quality.jsonl"
                write_jsonl(text_path, text_result.pop("judgments"))
                result["textSecondary"] = text_result
            text_status = {
                "status": "completed",
                "probe": probe_reason,
                "model": config["textJudge"]["model"],
                "reasoningEffort": config["textJudge"]["reasoningEffort"],
                "runSummaries": text_runs,
            }
        else:
            text_status = {
                "status": "deferred",
                "reason": f"ChatMock resource guard failed: {probe_reason}",
                "placeholderScoresEmitted": False,
            }
    else:
        text_status = {
            "status": "deferred",
            "reason": "ChatMock judge disabled; no placeholder score emitted",
            "placeholderScoresEmitted": False,
        }
    cases = [aggregate_case(output_id, [item for item in run_results if item["outputId"] == output_id], args.mode, config) for output_id in sorted(grouped)]
    for case in cases:
        if args.compare:
            case["protocolDifferences"] = compare_case(case, args.compare, config)
    status_counts = Counter(str(item.get("status")) for item in manifests)
    report = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "status": "completed" if not status_counts.get("runtime_blocked") else "completed_with_runtime_blocked",
        "time": utc_now(),
        "generationRoot": str(generation_root),
        "evaluationRoot": str(evaluation_root),
        "goldPath": str(gold_path),
        "goldSha256": sha256_file(gold_path),
        "completedCases": len(cases),
        "completedCaseRuns": len(run_results),
        "manifestStatuses": dict(status_counts),
        "method": method_metadata(config_path, config),
        "text": {
            **text_status,
            "anchorCase": config["textJudge"]["anchorCase"],
            "normalization": config["textJudge"]["normalization"],
        },
        "inputArtifacts": input_artifacts,
        "inputSetSha256": sha256_text(json.dumps(input_artifacts, sort_keys=True, separators=(",", ":"))),
        "cases": cases,
    }
    write_json(evaluation_root / "evaluation_report.json", report)
    lines = [
        f"# Experiment 6 {PROTOCOL}", "",
        f"- mode: `{args.mode}`", f"- status: `{report['status']}`",
        f"- method hash: `{report['method']['methodSha256']}`",
        f"- Text: `{report['text']['status']}`", "",
        "| case | runs | macro P mean | macro R mean | macro F1 mean | shared top runs |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        macro = case["aggregate"]["macro"]
        selection = case.get("selection")
        top_runs = "NA" if selection is None else ",".join(map(str, selection["top3"]["runs"]))
        lines.append(
            f"| {case['outputId']} | {case['runs']} | "
            f"{format_metric_value(macro['precision']['mean'])} | "
            f"{format_metric_value(macro['recall']['mean'])} | "
            f"{format_metric_value(macro['f1']['mean'])} | {top_runs} |"
        )
    lines.extend(["", "Top-1/top-3 are descriptive only. Diagnostic mode never emits a run ranking."])
    (evaluation_root / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(evaluation_root / "artifact_inventory.json", {
        "inputArtifacts": input_artifacts,
        "outputArtifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(evaluation_root.rglob("*"))
            if path.is_file() and path.name != "artifact_inventory.json"
        ],
    })
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("formal", "diagnostic"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "experiment6_narrative2_evaluation_v6_0_2.json")
    parser.add_argument("--output-root", type=Path, required=False)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--compare", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--text-judge", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--text-judge-limit", type=int, default=0, help="0 evaluates all aligned bindings")
    parser.add_argument("--no-sentence-model", action="store_true")
    parser.add_argument("--calibrate-trend", action="store_true")
    parser.add_argument("--calibration-output", type=Path)
    args = parser.parse_args(argv)
    if not args.calibrate_trend and args.output_root is None:
        parser.error("--output-root is required unless --calibrate-trend is used")
    for item in args.compare:
        if "=" not in item:
            parser.error("--compare must be LABEL=PATH")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = build(args)
    except (ProtocolError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"protocol": PROTOCOL, "status": "blocked", "error": str(error), "time": utc_now()}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "protocol": PROTOCOL,
        "status": report["status"],
        "mode": args.mode,
        "completedCases": report.get("completedCases"),
        "completedCaseRuns": report.get("completedCaseRuns"),
        "evaluationRoot": report.get("evaluationRoot"),
        "time": report["time"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
