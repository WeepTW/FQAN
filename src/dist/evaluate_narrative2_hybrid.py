#!/usr/bin/env python3
"""Hybrid six-field evaluator for the Experiment 6 narrative2 extension."""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
import os
import re
import random
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openai import OpenAI

from experiment6_paths import PATHS
import evaluate_data_binding as legacy_binding


REPO_ROOT = PATHS.repo
WORKSPACE_ROOT = PATHS.workspace
FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
HARD_FIELDS = ("DataName", "Position")
JUDGED_FIELDS = ("ObjectName", "Trend", "Text")
JUDGE_VALIDATION_VERSION = "narrative2-semantic-judge-v3-blinded-ab"

JUDGE_SYSTEM_PROMPT = """You are a blinded annotation-equivalence judge for a financial narrative dataset.
The compared values are labelled A and B; their order is randomized. You are not told which
model, case, run, or annotation role produced either value. Compare only A and B against the
same sourceText and already hard-aligned DataName/Position. Do not repair either value and do
not use outside knowledge.

ObjectName: accept only synonymous or explicit coreferent mentions of the same financial
entity or measure. Reject hypernyms, hyponyms, related companies, related concepts, and any
different entity, population, measure, time scope, or category.

Trend: accept only when direction, period, baseline, and scope are all the same. Opposite
direction or different baselines always fail. A point comparison, sustained trend, reversal,
extremum, and chart pattern are not interchangeable merely because broad direction agrees.

Text: accept only the same complete proposition, preserving subject, trend, measured numbers,
time/condition scope, comparison baseline, and negation polarity. Partial overlap, a missing
qualification, or any different number fails.

Return exactly one strict JSON object with one key, decisions. Return one decision per
decisionId and no extras. Every decision has exactly decisionId, equivalent, matchedPairs,
confidence, evidenceSpan, reasonCode. For ObjectName, matchedPairs contains unique zero-based
A-index/B-index pairs using the historical key names goldIndex/predictionIndex; for other
fields it is []. confidence is from 0 through 1. When equivalent is true, evidenceSpan must be
a verbatim non-empty span of sourceText. Use concise reasonCode values."""



class ProtocolError(ValueError):
    """Raised when evaluation inputs violate the frozen protocol."""


class JudgeError(RuntimeError):
    """Raised when the semantic judge cannot produce a valid decision."""


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict[str, Any]:
        if self.tp == self.fp == self.fn == 0:
            precision = recall = f1 = 1.0
        else:
            precision = (
                self.tp / (self.tp + self.fp)
                if self.tp + self.fp
                else 0.0
            )
            recall = (
                self.tp / (self.tp + self.fn)
                if self.tp + self.fn
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load JSON {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ProtocolError(f"cannot read JSONL {path}: {error}") from error
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ProtocolError(
                f"invalid JSONL {path} line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ProtocolError(
                f"JSONL {path} line {line_number} must be an object"
            )
        records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
            )
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ProtocolError("evaluation config must be an object")
    if config.get("protocol") != "narrative2-hybrid-v2":
        raise ProtocolError("unexpected evaluation protocol")
    if tuple(config.get("hardFields") or ()) != HARD_FIELDS:
        raise ProtocolError("evaluation hardFields must be DataName, Position")
    if config.get("numericField") != "Num":
        raise ProtocolError("evaluation numericField must be Num")
    if config.get("failureScoring") != {
        "formatInvalidPrediction": "empty_prediction_formal_score",
        "runtimeBlockedRow": "empty_prediction_formal_score_with_runtime_block",
        "runtimeBlockedRun": "score_diagnostic_withhold_ranking",
        "partialCaseRequiresAllRuns": True,
    }:
        raise ProtocolError("unexpected evaluation failureScoring policy")
    return config


def workspace_path(raw: str | Mapping[str, Any]) -> Path:
    if isinstance(raw, Mapping):
        return PATHS.resolve_locator(raw)
    path = Path(raw)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "bundle_manifest.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ProtocolError("evaluation bundle manifest has no files[]")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProtocolError("evaluation manifest entry must be an object")
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolError(f"unsafe evaluation manifest path: {relative}")
        target = bundle / relative
        if (
            not target.is_file()
            or target.stat().st_size != entry.get("bytes")
            or sha256_file(target) != entry.get("sha256")
        ):
            raise ProtocolError(f"evaluation manifest mismatch: {relative}")
    return {
        "manifest": str(manifest_path),
        "manifestSha256": sha256_file(manifest_path),
        "filesChecked": len(entries),
        "status": "passed",
    }


def normalize_unicode(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return text


def normalize_hard_string(value: Any) -> str:
    return " ".join(normalize_unicode(value).strip().casefold().split())


def normalize_semantic_exact(value: Any) -> str:
    return " ".join(normalize_unicode(value).strip().split())


def normalize_soft_string(value: Any) -> str:
    text = normalize_hard_string(value)
    text = re.sub(r"[^\w.+%-]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split()).strip(" .")


def is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return normalize_hard_string(value) in {"", "none", "null"}
    if isinstance(value, list):
        return not value or all(is_absent(item) for item in value)
    return False


def fixed_canonical(value: Any) -> Any:
    """Typed JSON identity for hard DataName/Position anchors."""

    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return ("number", "non-finite")
        return (
            ("number", number)
            if math.isfinite(number)
            else ("number", "non-finite")
        )
    if isinstance(value, list):
        return ("array", tuple(fixed_canonical(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                (str(key), fixed_canonical(value[key]))
                for key in sorted(value, key=lambda item: str(item))
            ),
        )
    return (type(value).__name__, repr(value))


def hard_key(binding: Mapping[str, Any]) -> tuple[Any, Any]:
    return (
        fixed_canonical(binding.get("DataName")),
        fixed_canonical(binding.get("Position")),
    )


def stable_binding_key(binding: Mapping[str, Any]) -> str:
    """Order-independent tie-breaker for duplicate hard anchors."""

    return json.dumps(
        binding,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parse_numeric_value(item: Any) -> tuple[float, bool, str] | None:
    if isinstance(item, bool):
        return None
    if isinstance(item, (int, float)):
        number = float(item)
        return (number, False, "number") if math.isfinite(number) else None
    if not isinstance(item, str):
        return None
    text = normalize_unicode(item).strip().lower()
    if not text:
        return None
    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()
    is_percent = bool(re.search(r"(?:%|percent|percentage\s*points?)\s*$", text))
    text = re.sub(r"(?:%|percent|percentage\s*points?)\s*$", "", text).strip()
    multiplier = 1.0
    suffix_match = re.search(r"\s*(thousand|million|billion|bn|[kmb])\s*$", text)
    if suffix_match:
        suffix = suffix_match.group(1)
        multiplier = {
            "k": 1e3, "thousand": 1e3,
            "m": 1e6, "million": 1e6,
            "b": 1e9, "bn": 1e9, "billion": 1e9,
        }[suffix]
        text = text[: suffix_match.start()].strip()
    text = re.sub(r"^[\$€£¥]\s*", "", text)
    text = text.replace(",", "").replace(" ", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return None
    number = float(text) * multiplier
    if negative_parentheses:
        number = -number
    if not math.isfinite(number):
        return None
    return number, is_percent, "percentage_point" if is_percent else "absolute"


def finite_number_list(value: Any) -> list[tuple[float, bool, str]] | None:
    if is_absent(value):
        return []
    if not isinstance(value, list):
        return None
    result: list[tuple[float, bool, str]] = []
    for item in value:
        parsed = parse_numeric_value(item)
        if parsed is None:
            return None
        result.append(parsed)
    return result

def maximum_matching(
    left_size: int,
    right_size: int,
    predicate: Callable[[int, int], bool],
) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []

    def search(left_index: int, used_right: set[int], pairs: list[tuple[int, int]]) -> None:
        nonlocal best
        if left_index >= left_size:
            if len(pairs) > len(best):
                best = list(pairs)
            return
        if len(pairs) + left_size - left_index <= len(best):
            return
        search(left_index + 1, used_right, pairs)
        for right_index in range(right_size):
            if right_index in used_right or not predicate(left_index, right_index):
                continue
            used_right.add(right_index)
            pairs.append((left_index, right_index))
            search(left_index + 1, used_right, pairs)
            pairs.pop()
            used_right.remove(right_index)

    search(0, set(), [])
    return best


def numeric_counts(
    gold: Any, prediction: Any, absolute_tolerance: float, relative_tolerance: float
) -> tuple[Counts, dict[str, Any]]:
    gold_values = finite_number_list(gold)
    prediction_values = finite_number_list(prediction)
    if gold_values is None or prediction_values is None:
        return Counts(
            fp=0 if prediction_values == [] else 1,
            fn=0 if gold_values == [] else 1,
        ), {
            "method": "invalid_numeric_type",
            "gold": gold,
            "prediction": prediction,
        }
    pairs = maximum_matching(
        len(gold_values),
        len(prediction_values),
        lambda left, right: (
            gold_values[left][1] == prediction_values[right][1]
            and math.isclose(
                gold_values[left][0],
                prediction_values[right][0],
                abs_tol=absolute_tolerance,
                rel_tol=relative_tolerance,
            )
        ),
    )
    counts = Counts(
        tp=len(pairs),
        fp=len(prediction_values) - len(pairs),
        fn=len(gold_values) - len(pairs),
    )
    def public(values: Sequence[tuple[float, bool, str]]) -> list[dict[str, Any]]:
        return [
            {"value": value, "isPercentagePoint": percent, "semantic": semantic}
            for value, percent, semantic in values
        ]
    return counts, {
        "method": "numeric_multiset_isclose_percentage_point_sensitive",
        "gold": public(gold_values),
        "prediction": public(prediction_values),
        "matchedPairs": [
            {"goldIndex": left, "predictionIndex": right}
            for left, right in pairs
        ],
        "absoluteTolerance": absolute_tolerance,
        "relativeTolerance": relative_tolerance,
    }

def exact_object_pairs(gold: Any, prediction: Any) -> list[tuple[int, int]]:
    if not isinstance(gold, list) or not isinstance(prediction, list):
        return []
    return maximum_matching(
        len(gold),
        len(prediction),
        lambda left, right: isinstance(gold[left], str)
        and isinstance(prediction[right], str)
        and normalize_semantic_exact(gold[left])
        == normalize_semantic_exact(prediction[right]),
    )


def trend_alias(value: Any, aliases: Mapping[str, str]) -> str | None:
    if is_absent(value):
        return None
    normalized = normalize_soft_string(value)
    return aliases.get(normalized, normalized)


def evidence_in_source(evidence: Any, source_text: str) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    candidate = normalize_unicode(evidence).strip()
    return bool(candidate) and candidate in normalize_unicode(source_text)


def build_data_evidence(
    input_data: Any, binding: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(input_data, str):
        return []
    try:
        rows = json.loads(input_data)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    data_name = binding.get("DataName")
    position = binding.get("Position")
    if not isinstance(data_name, str) or not isinstance(position, list):
        return []
    evidence: list[dict[str, Any]] = []
    for span in position:
        if not isinstance(span, dict):
            continue
        begin = span.get("Begin")
        end = span.get("End")
        if (
            not isinstance(begin, list)
            or not isinstance(end, list)
            or len(begin) != 2
            or len(end) != 2
            or not all(isinstance(value, int) for value in (*begin, *end))
        ):
            continue
        start_row, end_row = sorted((begin[0], end[0]))
        for row_index in range(max(0, start_row), min(len(rows) - 1, end_row) + 1):
            row = rows[row_index]
            if not isinstance(row, dict):
                continue
            first_key = next(iter(row), None)
            item = {"row": row_index, "DataName": data_name, "value": row.get(data_name)}
            if first_key is not None:
                item["axis"] = {first_key: row.get(first_key)}
            evidence.append(item)
    return evidence


class SemanticJudge:
    def __init__(
        self,
        config: Mapping[str, Any],
        checkpoint_path: Path,
        disabled: bool = False,
    ) -> None:
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.disabled = disabled
        self.live_calls = 0
        self.cache_hits = 0
        self.records = (
            {
                str(record["checkpointKey"]): record
                for record in read_jsonl(checkpoint_path)
            }
            if checkpoint_path.is_file()
            else {}
        )
        api_key = os.environ.get(str(config["apiKeyEnvironment"])) or str(config["defaultApiKey"])
        self.client = OpenAI(
            base_url=str(config["baseUrl"]),
            api_key=api_key,
            timeout=float(config["requestTimeoutSeconds"]),
            max_retries=0,
        )

    def _request(self, prompt: str, seed: int) -> tuple[str, str, str]:
        attempts = int(self.config["maxAttempts"])
        delays = [float(value) for value in self.config["retryDelaysSeconds"]]
        last_error: Exception | None = None
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "narrative2_semantic_decisions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["decisions"],
                    "properties": {
                        "decisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "decisionId", "equivalent", "matchedPairs",
                                    "confidence", "evidenceSpan", "reasonCode",
                                ],
                                "properties": {
                                    "decisionId": {"type": "string"},
                                    "equivalent": {"type": "boolean"},
                                    "matchedPairs": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["goldIndex", "predictionIndex"],
                                            "properties": {
                                                "goldIndex": {"type": "integer", "minimum": 0},
                                                "predictionIndex": {"type": "integer", "minimum": 0},
                                            },
                                        },
                                    },
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    "evidenceSpan": {"type": "string"},
                                    "reasonCode": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
        }
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=str(self.config["model"]),
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    reasoning_effort=str(self.config["reasoningEffort"]),
                    response_format=response_format,
                    max_tokens=int(self.config["maxTokens"]),
                    temperature=0,
                    seed=seed,
                )
                content = response.choices[0].message.content if response.choices else ""
                return (
                    str(content or ""),
                    str(getattr(response, "model", "") or ""),
                    str(getattr(response, "id", "") or ""),
                )
            except Exception as error:
                last_error = error
                if attempt >= attempts:
                    break
                delay = delays[min(attempt - 1, len(delays) - 1)] if delays else 0
                if delay:
                    time.sleep(delay)
        assert last_error is not None
        raise JudgeError(str(last_error)) from last_error

    @staticmethod
    def primary_swap(source: str, decision_id: str) -> bool:
        digest = sha256_text(source + "\0" + decision_id + "\0primary")
        return int(digest[-1], 16) % 2 == 1

    def public_decisions(
        self,
        source: str,
        decisions: Sequence[Mapping[str, Any]],
        force_opposite: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, bool]]:
        public: list[dict[str, Any]] = []
        swapped_by_id: dict[str, bool] = {}
        for decision in decisions:
            decision_id = str(decision["decisionId"])
            swapped = self.primary_swap(source, decision_id)
            if force_opposite:
                swapped = not swapped
            swapped_by_id[decision_id] = swapped
            item = {
                key: value
                for key, value in decision.items()
                if key not in {"gold", "prediction"}
            }
            item["A"] = decision.get("prediction") if swapped else decision.get("gold")
            item["B"] = decision.get("gold") if swapped else decision.get("prediction")
            public.append(item)
        return public, swapped_by_id

    def decide(
        self,
        source: str,
        source_text: str,
        decisions: Sequence[Mapping[str, Any]],
        *,
        force_opposite: bool = False,
        audit_label: str = "primary",
    ) -> dict[str, dict[str, Any]]:
        if not decisions:
            return {}
        if self.disabled:
            return {
                str(item["decisionId"]): {
                    "decisionId": item["decisionId"],
                    "equivalent": False,
                    "matchedPairs": [],
                    "confidence": 0.0,
                    "evidenceSpan": "",
                    "reasonCode": "judge_disabled",
                    "accepted": False,
                    "abSwapped": False,
                }
                for item in decisions
            }
        public_decisions, swapped_by_id = self.public_decisions(
            source, decisions, force_opposite
        )
        request_document = {
            "sourceText": source_text,
            "decisions": public_decisions,
        }
        prompt = json.dumps(request_document, ensure_ascii=False, separators=(",", ":"))
        request_sha = sha256_text(prompt)
        policy_key = {
            "source": source,
            "requestSha256": request_sha,
            "model": self.config["model"],
            "reasoningEffort": self.config["reasoningEffort"],
            "minimumConfidence": self.config["minimumConfidence"],
            "validationVersion": JUDGE_VALIDATION_VERSION,
            "auditLabel": audit_label,
        }
        checkpoint_key = sha256_text(
            json.dumps(policy_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        existing = self.records.get(checkpoint_key)
        if existing and existing.get("status") == "completed":
            self.cache_hits += 1
            return {
                str(item["decisionId"]): item
                for item in existing["validatedDecisions"]
            }

        raw_response, response_model, response_id = self._request(
            prompt, int(checkpoint_key[:8], 16)
        )
        self.live_calls += 1
        if response_model != self.config["model"]:
            raise JudgeError(
                f"judge model mismatch: {response_model!r} != {self.config['model']!r}"
            )
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise JudgeError(f"judge strict JSON parse failed: {error}") from error
        if not isinstance(payload, dict) or set(payload) != {"decisions"}:
            raise JudgeError("judge response must contain exactly decisions")
        raw_decisions = payload["decisions"]
        if not isinstance(raw_decisions, list):
            raise JudgeError("judge response has no decisions[]")
        expected_ids = [str(item["decisionId"]) for item in decisions]
        expected_by_id = {str(item["decisionId"]): item for item in decisions}
        public_by_id = {str(item["decisionId"]): item for item in public_decisions}
        by_id: dict[str, dict[str, Any]] = {}
        validated: list[dict[str, Any]] = []
        minimum_confidence = float(self.config["minimumConfidence"])
        required_keys = {
            "decisionId", "equivalent", "matchedPairs",
            "confidence", "evidenceSpan", "reasonCode",
        }
        for raw in raw_decisions:
            if not isinstance(raw, dict) or set(raw) != required_keys:
                raise JudgeError("judge decision has missing or extra fields")
            decision_id = str(raw.get("decisionId") or "")
            if not decision_id or decision_id in by_id or decision_id not in expected_by_id:
                raise JudgeError(f"judge decisionId invalid: {decision_id!r}")
            equivalent = raw.get("equivalent")
            confidence = raw.get("confidence")
            matched_pairs = raw.get("matchedPairs")
            if not isinstance(equivalent, bool):
                raise JudgeError(f"{decision_id} equivalent must be boolean")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
            ):
                raise JudgeError(f"{decision_id} confidence must be in [0,1]")
            if not isinstance(matched_pairs, list):
                raise JudgeError(f"{decision_id} matchedPairs must be an array")
            if not isinstance(raw["evidenceSpan"], str):
                raise JudgeError(f"{decision_id} evidenceSpan must be a string")
            if not isinstance(raw["reasonCode"], str) or not raw["reasonCode"].strip():
                raise JudgeError(f"{decision_id} reasonCode must be non-empty")
            expected = expected_by_id[decision_id]
            converted_pairs: list[dict[str, int]] = []
            if expected.get("field") != "ObjectName" and matched_pairs:
                raise JudgeError(f"{decision_id} matchedPairs is only valid for ObjectName")
            if expected.get("field") == "ObjectName":
                public_item = public_by_id[decision_id]
                a_size = len(public_item.get("A") or [])
                b_size = len(public_item.get("B") or [])
                seen_a: set[int] = set()
                seen_b: set[int] = set()
                pair_validation_error: str | None = None
                for pair in matched_pairs:
                    if not isinstance(pair, dict) or set(pair) != {"goldIndex", "predictionIndex"}:
                        pair_validation_error = "invalid_matched_pair_shape"
                        converted_pairs = []
                        break
                    a_index = pair["goldIndex"]
                    b_index = pair["predictionIndex"]
                    if (
                        isinstance(a_index, bool) or isinstance(b_index, bool)
                        or not isinstance(a_index, int) or not isinstance(b_index, int)
                        or not 0 <= a_index < a_size or not 0 <= b_index < b_size
                        or a_index in seen_a or b_index in seen_b
                    ):
                        pair_validation_error = "invalid_or_non_unique_matched_pair_index"
                        converted_pairs = []
                        break
                    seen_a.add(a_index)
                    seen_b.add(b_index)
                    if swapped_by_id[decision_id]:
                        converted_pairs.append({
                            "goldIndex": b_index,
                            "predictionIndex": a_index,
                        })
                    else:
                        converted_pairs.append({
                            "goldIndex": a_index,
                            "predictionIndex": b_index,
                        })
            accepted = (
                equivalent
                and float(confidence) >= minimum_confidence
                and evidence_in_source(raw.get("evidenceSpan"), source_text)
                and (expected.get("field") != "ObjectName" or bool(converted_pairs))
                and (
                    expected.get("field") != "ObjectName"
                    or pair_validation_error is None
                )
            )
            item = {
                "decisionId": decision_id,
                "equivalent": equivalent,
                "matchedPairs": converted_pairs,
                "confidence": float(confidence),
                "evidenceSpan": raw.get("evidenceSpan"),
                "reasonCode": raw.get("reasonCode"),
                "accepted": accepted,
                "abSwapped": swapped_by_id[decision_id],
                "auditLabel": audit_label,
            }
            if expected.get("field") == "ObjectName" and pair_validation_error:
                item["validationError"] = pair_validation_error
            validated.append(item)
            by_id[decision_id] = item
        if sorted(by_id) != sorted(expected_ids):
            raise JudgeError(
                f"judge decision IDs mismatch: received={sorted(by_id)}, expected={sorted(expected_ids)}"
            )
        checkpoint = {
            "time": utc_now(),
            "checkpointKey": checkpoint_key,
            "source": source,
            "status": "completed",
            **policy_key,
            "responseModel": response_model,
            "responseId": response_id,
            "request": request_document,
            "rawResponse": raw_response,
            "rawResponseSha256": sha256_text(raw_response),
            "validatedDecisions": validated,
        }
        self.records[checkpoint_key] = checkpoint
        write_jsonl(
            self.checkpoint_path,
            [self.records[key] for key in sorted(self.records)],
        )
        return by_id

def align_bindings(
    gold_bindings: Sequence[Mapping[str, Any]],
    predicted_bindings: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    gold_by_key: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    prediction_by_key: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for index, binding in enumerate(gold_bindings):
        gold_by_key[hard_key(binding)].append(index)
    for index, binding in enumerate(predicted_bindings):
        prediction_by_key[hard_key(binding)].append(index)
    pairs: list[tuple[int, int]] = []
    matched_gold: set[int] = set()
    matched_prediction: set[int] = set()
    for key in sorted(set(gold_by_key) & set(prediction_by_key), key=repr):
        gold_indices = sorted(
            gold_by_key[key], key=lambda index: stable_binding_key(gold_bindings[index])
        )
        prediction_indices = sorted(
            prediction_by_key[key],
            key=lambda index: stable_binding_key(predicted_bindings[index]),
        )
        for gold_index, prediction_index in zip(gold_indices, prediction_indices):
            pairs.append((gold_index, prediction_index))
            matched_gold.add(gold_index)
            matched_prediction.add(prediction_index)
    return (
        pairs,
        [
            index
            for index in range(len(gold_bindings))
            if index not in matched_gold
        ],
        [
            index
            for index in range(len(predicted_bindings))
            if index not in matched_prediction
        ],
    )


def scalar_counts(passed: bool) -> Counts:
    return Counts(tp=1) if passed else Counts(fp=1, fn=1)


def unmatched_gold_counts(binding: Mapping[str, Any]) -> dict[str, Counts]:
    return {
        "ObjectName": Counts(fn=len(binding.get("ObjectName") or [])),
        "DataName": Counts(fn=1),
        "Position": Counts(fn=1),
        "Trend": Counts(fn=0 if is_absent(binding.get("Trend")) else 1),
        "Num": Counts(fn=len(finite_number_list(binding.get("Num")) or [])),
        "Text": Counts(fn=1),
    }


def unmatched_prediction_counts(binding: Mapping[str, Any]) -> dict[str, Counts]:
    return {
        "ObjectName": Counts(fp=len(binding.get("ObjectName") or [])),
        "DataName": Counts(fp=1),
        "Position": Counts(fp=1),
        "Trend": Counts(fp=0 if is_absent(binding.get("Trend")) else 1),
        "Num": Counts(fp=len(finite_number_list(binding.get("Num")) or [])),
        "Text": Counts(fp=1),
    }


def build_semantic_plan(
    input_data: Any,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    gold_index: int,
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    gold_objects = gold.get("ObjectName")
    predicted_objects = prediction.get("ObjectName")
    exact_pairs = exact_object_pairs(gold_objects, predicted_objects)
    matched_gold = {left for left, _ in exact_pairs}
    matched_prediction = {right for _, right in exact_pairs}
    unmatched_gold_objects = [
        index
        for index in range(len(gold_objects) if isinstance(gold_objects, list) else 0)
        if index not in matched_gold
    ]
    unmatched_predicted_objects = [
        index
        for index in range(
            len(predicted_objects) if isinstance(predicted_objects, list) else 0
        )
        if index not in matched_prediction
    ]
    object_decision_id = f"binding_{gold_index}_ObjectName"
    decisions: list[dict[str, Any]] = []
    if unmatched_gold_objects and unmatched_predicted_objects:
        decisions.append(
            {
                "decisionId": object_decision_id,
                "field": "ObjectName",
                "gold": [gold_objects[index] for index in unmatched_gold_objects],
                "prediction": [
                    predicted_objects[index]
                    for index in unmatched_predicted_objects
                ],
                "DataName": gold["DataName"],
                "Position": gold["Position"],
                "dataEvidence": build_data_evidence(input_data, gold),
            }
        )

    gold_trend = gold.get("Trend")
    predicted_trend = prediction.get("Trend")
    trend_decision_id = f"binding_{gold_index}_Trend"
    if is_absent(gold_trend) and is_absent(predicted_trend):
        trend_deterministic: bool | None = True
    elif is_absent(gold_trend) != is_absent(predicted_trend):
        trend_deterministic = False
    elif trend_alias(gold_trend, aliases) == trend_alias(
        predicted_trend, aliases
    ):
        trend_deterministic = True
    else:
        trend_deterministic = None
        decisions.append(
            {
                "decisionId": trend_decision_id,
                "field": "Trend",
                "gold": gold_trend,
                "prediction": predicted_trend,
                "DataName": gold["DataName"],
                "Position": gold["Position"],
                "dataEvidence": build_data_evidence(input_data, gold),
            }
        )

    gold_text = gold.get("Text")
    predicted_text = prediction.get("Text")
    text_decision_id = f"binding_{gold_index}_Text"
    text_deterministic = (
        isinstance(gold_text, str)
        and isinstance(predicted_text, str)
        and normalize_semantic_exact(gold_text)
        == normalize_semantic_exact(predicted_text)
    )
    if not text_deterministic:
        decisions.append(
            {
                "decisionId": text_decision_id,
                "field": "Text",
                "gold": gold_text,
                "prediction": predicted_text,
                "DataName": gold["DataName"],
                "Position": gold["Position"],
            }
        )
    return {
        "decisions": decisions,
        "goldObjects": gold_objects,
        "predictedObjects": predicted_objects,
        "exactObjectPairs": exact_pairs,
        "unmatchedGoldObjects": unmatched_gold_objects,
        "unmatchedPredictedObjects": unmatched_predicted_objects,
        "objectDecisionId": object_decision_id,
        "goldTrend": gold_trend,
        "predictedTrend": predicted_trend,
        "trendDecisionId": trend_decision_id,
        "trendDeterministic": trend_deterministic,
        "goldText": gold_text,
        "predictedText": predicted_text,
        "textDecisionId": text_decision_id,
        "textDeterministic": text_deterministic,
    }


def evaluate_aligned_binding(
    source: str,
    source_text: str,
    input_data: Any,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    gold_index: int,
    prediction_index: int,
    aliases: Mapping[str, str],
    numeric_config: Mapping[str, Any],
    judge: SemanticJudge,
    semantic_plan: Mapping[str, Any] | None = None,
    judge_results_override: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Counts], dict[str, Any], int]:
    field_counts = {field: Counts() for field in FIELDS}
    field_counts["DataName"] = Counts(tp=1)
    field_counts["Position"] = Counts(tp=1)
    details: dict[str, Any] = {
        "goldIndex": gold_index,
        "predictionIndex": prediction_index,
        "hardKey": {
            "DataName": gold["DataName"],
            "Position": gold["Position"],
        },
        "fields": {
            "DataName": {"method": "hard_identity", "pass": True},
            "Position": {"method": "hard_identity", "pass": True},
        },
    }
    plan = dict(
        semantic_plan
        or build_semantic_plan(input_data, gold, prediction, gold_index, aliases)
    )
    decisions = plan["decisions"]
    gold_objects = plan["goldObjects"]
    predicted_objects = plan["predictedObjects"]
    exact_pairs = plan["exactObjectPairs"]
    unmatched_gold_objects = plan["unmatchedGoldObjects"]
    unmatched_predicted_objects = plan["unmatchedPredictedObjects"]
    object_decision_id = plan["objectDecisionId"]
    gold_trend = plan["goldTrend"]
    predicted_trend = plan["predictedTrend"]
    trend_decision_id = plan["trendDecisionId"]
    trend_deterministic = plan["trendDeterministic"]
    gold_text = plan["goldText"]
    predicted_text = plan["predictedText"]
    text_decision_id = plan["textDecisionId"]
    text_deterministic = plan["textDeterministic"]
    if judge_results_override is None:
        judge_results = judge.decide(source, source_text, decisions)
        judge_calls = 1 if decisions and not judge.disabled else 0
    else:
        judge_results = dict(judge_results_override)
        judge_calls = 0

    object_pairs = list(exact_pairs)
    object_method = "normalized_exact"
    object_judgment = judge_results.get(object_decision_id)
    if object_judgment and object_judgment["accepted"]:
        used_gold: set[int] = set()
        used_prediction: set[int] = set()
        for raw_pair in object_judgment["matchedPairs"]:
            if not isinstance(raw_pair, dict):
                continue
            left = raw_pair.get("goldIndex")
            right = raw_pair.get("predictionIndex")
            if (
                isinstance(left, int)
                and isinstance(right, int)
                and 0 <= left < len(unmatched_gold_objects)
                and 0 <= right < len(unmatched_predicted_objects)
                and left not in used_gold
                and right not in used_prediction
            ):
                used_gold.add(left)
                used_prediction.add(right)
                object_pairs.append(
                    (
                        unmatched_gold_objects[left],
                        unmatched_predicted_objects[right],
                    )
                )
        object_method = "normalized_exact_plus_gpt55_medium"
    gold_object_count = len(gold_objects) if isinstance(gold_objects, list) else 0
    predicted_object_count = (
        len(predicted_objects) if isinstance(predicted_objects, list) else 0
    )
    field_counts["ObjectName"] = Counts(
        tp=len(object_pairs),
        fp=predicted_object_count - len(object_pairs),
        fn=gold_object_count - len(object_pairs),
    )
    details["fields"]["ObjectName"] = {
        "method": object_method,
        "gold": gold_objects,
        "prediction": predicted_objects,
        "matchedPairs": [
            {"goldIndex": left, "predictionIndex": right}
            for left, right in sorted(object_pairs)
        ],
        "judge": object_judgment,
        **field_counts["ObjectName"].as_dict(),
    }

    if is_absent(gold_trend) and is_absent(predicted_trend):
        field_counts["Trend"] = Counts()
        trend_pass = True
        trend_method = "absent_empty_set"
        trend_judgment = None
    elif trend_deterministic is not None:
        field_counts["Trend"] = scalar_counts(trend_deterministic)
        trend_pass = trend_deterministic
        trend_method = "presence_or_alias"
        trend_judgment = None
    else:
        trend_judgment = judge_results.get(trend_decision_id)
        trend_pass = bool(trend_judgment and trend_judgment["accepted"])
        field_counts["Trend"] = scalar_counts(trend_pass)
        trend_method = "gpt55_medium_semantic"
    details["fields"]["Trend"] = {
        "method": trend_method,
        "gold": gold_trend,
        "prediction": predicted_trend,
        "pass": trend_pass,
        "judge": trend_judgment,
        **field_counts["Trend"].as_dict(),
    }

    num_counts, num_details = numeric_counts(
        gold.get("Num"),
        prediction.get("Num"),
        float(numeric_config["absoluteTolerance"]),
        float(numeric_config["relativeTolerance"]),
    )
    field_counts["Num"] = num_counts
    details["fields"]["Num"] = {**num_details, **num_counts.as_dict()}

    if text_deterministic:
        text_pass = True
        text_method = "unicode_punctuation_normalized_exact"
        text_judgment = None
    else:
        text_judgment = judge_results.get(text_decision_id)
        text_pass = bool(text_judgment and text_judgment["accepted"])
        text_method = "gpt55_medium_semantic"
    field_counts["Text"] = scalar_counts(text_pass)
    details["fields"]["Text"] = {
        "method": text_method,
        "gold": gold_text,
        "prediction": predicted_text,
        "pass": text_pass,
        "judge": text_judgment,
        **field_counts["Text"].as_dict(),
    }
    details["allFieldsExact"] = all(
        counts.fp == 0 and counts.fn == 0 for counts in field_counts.values()
    )
    return field_counts, details, judge_calls


def aggregate_counts(
    target: dict[str, Counts], source: Mapping[str, Counts]
) -> None:
    for field in FIELDS:
        target[field].add(source[field])


def legacy_exact_otn_metrics(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the canonical Experiment 6 vocabulary-set micro F1 for one run."""
    gold_rows = [
        {"case_id": str(target["source"]), "result": target["targetBindings"]}
        for target in targets
    ]
    prediction_rows = [
        {
            "case_id": str(record["source"]),
            "result": record.get("result", []) if record.get("formatValid") else [],
        }
        for record in predictions
    ]
    gold_extracted = legacy_binding.extract_rows(gold_rows, strict=True)
    pred_extracted = legacy_binding.extract_rows(prediction_rows, strict=False)
    blockers = legacy_binding.validate_case_alignment(
        gold_extracted, pred_extracted, require_data=True,
    )
    if blockers:
        raise ProtocolError("legacy O/T/N alignment failed: " + "; ".join(blockers))
    metrics = legacy_binding.metrics_from_extracted(
        gold_extracted, pred_extracted, ["subject", "trend", "numerical"],
    )
    return {
        "protocol": "experiment6-canonical-vocabulary-set-f1",
        "countingUnit": "vocabulary_item_micro_across_rows",
        "overall": {
            key: metrics[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")
        },
        "byField": metrics["by_field"],
    }


def evaluate_case(
    case_manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    case_eval_dir: Path,
    judge_disabled: bool,
    judge_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if case_manifest.get("expectedRows") != len(targets):
        raise ProtocolError(f"{case_manifest.get('outputId')} manifest expectedRows mismatch")
    runtime_blocked_rows = int(case_manifest.get("runtimeBlockedRows") or 0)
    effective_route = str(case_manifest.get("effectiveRoute") or case_manifest.get("route") or "")
    if effective_route in {"retriever-converter", "converter-control"}:
        if (
            case_manifest.get("converterModel") != "gpt-5.5"
            or case_manifest.get("reasoningEffort") != "medium"
        ):
            raise ProtocolError(
                f"{case_manifest.get('outputId')} converter must be exact gpt-5.5 medium"
            )
    if case_manifest.get("sourceId") == "gpt5_5" and case_manifest.get("reasoningEffort") != "medium":
        raise ProtocolError(f"{case_manifest.get('outputId')} direct gpt5_5 must use medium")

    files = case_manifest.get("files")
    hashes = case_manifest.get("hashes")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ProtocolError(f"{case_manifest.get('outputId')} has no files/hashes")
    predictions_path = Path(str(files["predictions"]))
    if sha256_file(predictions_path) != hashes.get("predictions"):
        raise ProtocolError(f"prediction SHA mismatch: {predictions_path}")
    predictions = read_jsonl(predictions_path)
    if len(predictions) != len(targets):
        raise ProtocolError(
            f"{case_manifest['outputId']} prediction rows={len(predictions)} expected={len(targets)}"
        )
    target_sources = [str(target["source"]) for target in targets]
    prediction_sources = [str(record.get("source") or "") for record in predictions]
    if prediction_sources != target_sources or len(set(prediction_sources)) != len(prediction_sources):
        raise ProtocolError(f"{case_manifest['outputId']} prediction Source order/coverage mismatch")
    for record in predictions:
        raw_response = record.get("rawResponse")
        if (
            record.get("run") != case_manifest.get("run")
            or not isinstance(raw_response, str)
            or record.get("rawResponseSha256") != sha256_text(raw_response)
            or not isinstance(record.get("result"), list)
        ):
            raise ProtocolError(f"{case_manifest['outputId']} prediction provenance mismatch")

    aliases = {
        normalize_soft_string(key): normalize_soft_string(value)
        for key, value in config["trendAliases"].items()
    }
    totals = {field: Counts() for field in FIELDS}
    row_outputs: list[dict[str, Any]] = []
    exact_bindings = 0
    binding_denominator = 0
    aligned_bindings = 0
    judge_batches = 0
    judge = SemanticJudge(
        config["judge"],
        judge_checkpoint_path or case_eval_dir / "judge_checkpoint.jsonl",
        disabled=judge_disabled,
    )
    audit_candidates: list[dict[str, Any]] = []

    for target, prediction_record in zip(targets, predictions):
        source = str(target["source"])
        predicted_bindings = prediction_record.get("result", [])
        gold_bindings = target["targetBindings"]
        pairs, unmatched_gold, unmatched_prediction = align_bindings(
            gold_bindings, predicted_bindings
        )
        row_counts = {field: Counts() for field in FIELDS}
        binding_details: list[dict[str, Any]] = []
        source_text = str(prediction_record.get("inputText") or "")
        input_data = prediction_record.get("inputData") or ""
        semantic_items: list[tuple[int, int, dict[str, Any]]] = []
        row_decisions: list[dict[str, Any]] = []
        for alignment_index, (gold_index, prediction_index) in enumerate(pairs):
            plan = build_semantic_plan(
                input_data,
                gold_bindings[gold_index],
                predicted_bindings[prediction_index],
                alignment_index,
                aliases,
            )
            semantic_items.append((gold_index, prediction_index, plan))
            row_decisions.extend(plan["decisions"])
        row_judge_results = judge.decide(source, source_text, row_decisions)
        if row_decisions:
            judge_batches += 1
        for decision in row_decisions:
            audit_candidates.append({
                "source": source,
                "sourceText": source_text,
                "decision": decision,
                "primary": row_judge_results[str(decision["decisionId"])],
            })
        for gold_index, prediction_index, plan in semantic_items:
            counts, details, calls = evaluate_aligned_binding(
                source,
                source_text,
                input_data,
                gold_bindings[gold_index],
                predicted_bindings[prediction_index],
                gold_index,
                prediction_index,
                aliases,
                config["numericComparison"],
                judge,
                semantic_plan=plan,
                judge_results_override=row_judge_results,
            )
            aggregate_counts(row_counts, counts)
            binding_details.append(details)
            aligned_bindings += 1
            exact_bindings += int(details["allFieldsExact"])
            if calls:
                raise AssertionError("batched semantic scoring made a live judge call")
        for gold_index in unmatched_gold:
            counts = unmatched_gold_counts(gold_bindings[gold_index])
            aggregate_counts(row_counts, counts)
            binding_details.append({
                "goldIndex": gold_index,
                "predictionIndex": None,
                "status": "missing_prediction_binding",
                "allFieldsExact": False,
            })
        for prediction_index in unmatched_prediction:
            counts = unmatched_prediction_counts(predicted_bindings[prediction_index])
            aggregate_counts(row_counts, counts)
            binding_details.append({
                "goldIndex": None,
                "predictionIndex": prediction_index,
                "status": "extra_prediction_binding",
                "allFieldsExact": False,
            })
        aggregate_counts(totals, row_counts)
        binding_denominator += max(len(gold_bindings), len(predicted_bindings))
        row_overall = Counts()
        for field in FIELDS:
            row_overall.add(row_counts[field])
        row_outputs.append({
            "source": source,
            "excelRow": target.get("excelRow"),
            "generationStatus": (
                "accepted" if prediction_record.get("formatValid") else "format_rejected_zero_prediction"
            ),
            "generationErrors": (
                [] if prediction_record.get("formatValid")
                else [prediction_record.get("parserDiagnostic")]
            ),
            "goldBindings": len(gold_bindings),
            "predictedBindings": len(predicted_bindings),
            "alignedBindings": len(pairs),
            "unmatchedGoldBindings": unmatched_gold,
            "unmatchedPredictionBindings": unmatched_prediction,
            "fieldMetrics": {field: row_counts[field].as_dict() for field in FIELDS},
            "overall": row_overall.as_dict(),
            "bindings": binding_details,
        })

    field_metrics = {field: totals[field].as_dict() for field in FIELDS}
    overall = Counts()
    for field in FIELDS:
        overall.add(totals[field])
    hybrid6_macro_f1 = sum(field_metrics[field]["f1"] for field in FIELDS) / len(FIELDS)
    macro_row_f1 = (
        sum(row["overall"]["f1"] for row in row_outputs) / len(row_outputs)
        if row_outputs else 1.0
    )

    audit_config = config["audit"]
    sample_size = (
        max(1, math.ceil(len(audit_candidates) * float(audit_config["sampleRate"])))
        if audit_candidates else 0
    )
    audit_seed = int(audit_config["seed"]) + int(case_manifest["run"]) + int(
        sha256_text(str(case_manifest["outputId"]))[:8], 16
    )
    sample_indexes = sorted(
        random.Random(audit_seed).sample(range(len(audit_candidates)), sample_size)
        if sample_size else []
    )
    audit_records: list[dict[str, Any]] = []
    agreements = 0
    for candidate_index in sample_indexes:
        candidate = audit_candidates[candidate_index]
        decision = candidate["decision"]
        decision_id = str(decision["decisionId"])
        swapped_result = judge.decide(
            candidate["source"],
            candidate["sourceText"],
            [decision],
            force_opposite=True,
            audit_label="swap-audit",
        )[decision_id]
        primary_signature = (
            bool(candidate["primary"]["accepted"]),
            tuple(
                sorted(
                    (pair["goldIndex"], pair["predictionIndex"])
                    for pair in candidate["primary"].get("matchedPairs", [])
                )
            ),
        )
        swapped_signature = (
            bool(swapped_result["accepted"]),
            tuple(
                sorted(
                    (pair["goldIndex"], pair["predictionIndex"])
                    for pair in swapped_result.get("matchedPairs", [])
                )
            ),
        )
        agreed = primary_signature == swapped_signature
        agreements += int(agreed)
        adjudication = None
        if not agreed and audit_config.get("thirdAdjudicationOnDisagreement"):
            adjudication = judge.decide(
                candidate["source"],
                candidate["sourceText"],
                [decision],
                force_opposite=False,
                audit_label="third-adjudication",
            )[decision_id]
        audit_records.append({
            "candidateIndex": candidate_index,
            "source": candidate["source"],
            "decision": decision,
            "primary": candidate["primary"],
            "swapped": swapped_result,
            "agreement": agreed,
            "thirdAdjudication": adjudication,
        })
    audit_summary = {
        "population": len(audit_candidates),
        "sampled": len(audit_records),
        "sampleRateConfigured": audit_config["sampleRate"],
        "seed": audit_seed,
        "agreements": agreements,
        "agreementRate": agreements / len(audit_records) if audit_records else None,
        "disagreements": len(audit_records) - agreements,
        "thirdAdjudications": sum(record["thirdAdjudication"] is not None for record in audit_records),
        "lowConfidence": [
            {
                "source": record["source"],
                "decisionId": record["primary"]["decisionId"],
                "confidence": record["primary"]["confidence"],
            }
            for record in audit_records
            if float(record["primary"]["confidence"]) < float(config["judge"]["minimumConfidence"])
        ],
    }

    legacy_exact_otn = legacy_exact_otn_metrics(targets, predictions)
    result = {
        "time": utc_now(),
        "protocol": config["protocol"],
        "outputId": case_manifest["outputId"],
        "sourceId": case_manifest["sourceId"],
        "promptMode": case_manifest["promptMode"],
        "part": case_manifest["part"],
        "official": case_manifest["official"],
        "run": case_manifest["run"],
        "requestedModel": case_manifest["requestedModel"],
        "actualModel": case_manifest["actualModel"],
        "reasoningEffort": case_manifest.get("reasoningEffort"),
        "predictionPath": str(predictions_path),
        "predictionSha256": hashes["predictions"],
        "expectedRows": len(targets),
        "acceptedRows": case_manifest["acceptedRows"],
        "rejectedRows": case_manifest["rejectedRows"],
        "runtimeBlockedRows": runtime_blocked_rows,
        "formatComplianceRate": case_manifest["formatComplianceRate"],
        "fieldMetrics": field_metrics,
        "overall": overall.as_dict(),
        "hybrid6MacroF1": hybrid6_macro_f1,
        "macroRowF1": macro_row_f1,
        "legacyExactOtn": legacy_exact_otn,
        "bindingAlignment": {
            "aligned": aligned_bindings,
            "gold": sum(len(target["targetBindings"]) for target in targets),
            "coverage": (
                aligned_bindings / sum(len(target["targetBindings"]) for target in targets)
                if targets else 1.0
            ),
        },
        "sixFieldBindingExact": {
            "passes": exact_bindings,
            "tested": binding_denominator,
            "rate": exact_bindings / binding_denominator if binding_denominator else 1.0,
        },
        "judgeBatches": judge_batches,
        "judgeCalls": judge.live_calls,
        "judgeCacheHits": judge.cache_hits,
        "judgeCheckpoint": str(judge.checkpoint_path),
        "judgeModel": config["judge"]["model"],
        "judgeReasoningEffort": config["judge"]["reasoningEffort"],
        "semanticAudit": audit_summary,
        "rows": row_outputs,
        "status": (
            "runtime_blocked_scored_no_ranking"
            if runtime_blocked_rows else "completed"
        ),
    }
    case_eval_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(case_eval_dir / "gold_extracted.jsonl", [
        {
            "source": target["source"],
            "excelRow": target.get("excelRow"),
            "targetBindings": target["targetBindings"],
        }
        for target in targets
    ])
    write_jsonl(case_eval_dir / "pred_extracted.jsonl", predictions)
    write_json(case_eval_dir / "extraction_report.json", {
        "rows": len(predictions),
        "formatValidRows": sum(bool(record.get("formatValid")) for record in predictions),
        "formatInvalidRows": sum(not bool(record.get("formatValid")) for record in predictions),
        "predictionSha256": hashes["predictions"],
        "goldSha256": config["goldSha256"],
    })
    write_json(case_eval_dir / "semantic_audit.json", {
        "summary": audit_summary,
        "records": audit_records,
    })
    write_json(case_eval_dir / "metrics.json", result)
    write_json(case_eval_dir / "status.json", {
        "outputId": result["outputId"],
        "run": result["run"],
        "status": result["status"],
        "runtimeBlockedRows": runtime_blocked_rows,
        "hybrid6MacroF1": hybrid6_macro_f1,
        "formatComplianceRate": result["formatComplianceRate"],
    })
    write_json(case_eval_dir / "hybrid_evaluation.json", result)
    return result

def run_fixed_baseline(
    evaluation_bundle: Path,
    case_manifest: Mapping[str, Any],
    case_eval_dir: Path,
) -> dict[str, Any]:
    case_eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_eval_dir / "fixed_strict_evaluation.json"
    stdout_path = case_eval_dir / "fixed_strict_stdout.log"
    command = [
        sys.executable,
        "-B",
        str(evaluation_bundle / "evaluate_narrative2_annotations.py"),
        "compare-batch",
        "--targets",
        str(evaluation_bundle / "gold_targets.json"),
        "--predictions",
        str(case_manifest["files"]["predictions"]),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=case_eval_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(
        "command="
        + " ".join(command)
        + f"\nreturncode={completed.returncode}\n"
        + "\n[stdout]\n"
        + completed.stdout
        + "\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if not output_path.is_file():
        return {
            "status": "blocked",
            "returnCode": completed.returncode,
            "output": str(output_path),
            "stdout": str(stdout_path),
        }
    document = read_json(output_path)
    return {
        "status": "completed" if completed.returncode == 0 else "blocked",
        "returnCode": completed.returncode,
        "reportedValid": document.get("valid"),
        "completedResults": document.get("summary", {}).get("completedResults"),
        "rejectedResults": document.get("summary", {}).get("rejectedResults"),
        "expectedResults": document.get("summary", {}).get("expectedResults"),
        "allFieldPrecision": document.get("summary", {}).get(
            "allFieldPrecision"
        ),
        "optimizationFieldPrecision": document.get("summary", {}).get(
            "optimizationFieldPrecision"
        ),
        "output": str(output_path),
        "outputSha256": sha256_file(output_path),
        "stdout": str(stdout_path),
    }


def score_stats(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "mean": None, "sampleSd": None, "min": None, "max": None}
    return {
        "count": len(numbers),
        "mean": statistics.mean(numbers),
        "sampleSd": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def legacy_field_f1(result: Mapping[str, Any], field: str) -> float:
    legacy = result.get("legacyExactOtn")
    fields = legacy.get("byField") if isinstance(legacy, dict) else None
    metric = fields.get(field) if isinstance(fields, dict) else None
    return float(metric.get("f1") or 0.0) if isinstance(metric, dict) else 0.0


def aggregate_case(
    output_id: str,
    run_results: Sequence[Mapping[str, Any]],
    manifests_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(run_results, key=lambda item: int(item["run"]))
    manifests = [
        manifests_by_key[(output_id, int(result["run"]))]
        for result in ordered
    ]
    runtime_blocked_rows = sum(
        int(manifest.get("runtimeBlockedRows") or 0) for manifest in manifests
    )
    completion_status = (
        "runtime_blocked_scored_no_ranking"
        if runtime_blocked_rows else "completed"
    )
    primary_values = [float(result["hybrid6MacroF1"]) for result in ordered]
    top = sorted(
        ordered,
        key=lambda item: (float(item["hybrid6MacroF1"]), -int(item["run"])),
        reverse=True,
    )[: int(generation_config.get("topK", 3))]
    field_stats = {
        field: score_stats([result["fieldMetrics"][field]["f1"] for result in ordered])
        for field in FIELDS
    }
    legacy = {
        field: score_stats([legacy_field_f1(result, field) for result in ordered])
        for field in ("ObjectName", "Trend", "Num")
    }
    output_files: list[str] = []
    for manifest, result in zip(manifests, ordered):
        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        output_files.extend(str(value) for value in files.values())
        case_eval = Path(str(result["predictionPath"])).parents[0]
        eval_dir = (
            Path(str(result["judgeCheckpoint"])).parent
            / "cases"
            / output_id
            / f"run_{int(result['run']):02d}"
        )
        output_files.extend([
            str(eval_dir / "metrics.json"),
            str(eval_dir / "status.json"),
            str(eval_dir / "gold_extracted.jsonl"),
            str(eval_dir / "pred_extracted.jsonl"),
            str(eval_dir / "extraction_report.json"),
            str(eval_dir / "semantic_audit.json"),
        ])
    unique_output_files = list(dict.fromkeys(output_files))
    first_manifest = manifests[0]
    actual_models = sorted({str(item.get("actualModel") or "") for item in manifests})
    adapters = sorted({str(item.get("adapter")) for item in manifests if item.get("adapter")})
    quantizations = sorted({
        str(item.get("quantization")) for item in manifests if item.get("quantization")
    })
    runtime_profiles = sorted({
        str(item.get("runtimeProfile")) for item in manifests if item.get("runtimeProfile")
    })
    mode = str(first_manifest["promptMode"])
    builder = generation_config["promptBuilder"]
    shot_count = (
        int(builder["manyShotCount"]) if mode == "many-shot"
        else int(builder["dynamicShotCount"]) if mode == "dynamic-shot"
        else 0
    )
    return {
        "model": {
            "output_id": output_id,
            "requested": first_manifest["requestedModel"],
            "actual": actual_models,
            "adapter": adapters,
            "quantization": quantizations,
        },
        "prompt": {
            "mode": mode,
            "part": first_manifest["part"],
            "route": first_manifest["route"],
            "effective_route": first_manifest.get("effectiveRoute"),
            "shot_count": shot_count,
            "prompt_hashes": sorted({str(item["hashes"]["prompts"]) for item in manifests}),
            "data_sha256": generation_config["inputWorkbook"]["sha256"],
            "generation_protocol": generation_config["protocol"],
            "evaluation_protocol": "narrative2-hybrid-v2",
        },
        "runtime": {
            "completion_status": completion_status,
            "runtime_blocked_rows": runtime_blocked_rows,
            "runs": [
                {
                    "run": item["run"],
                    "seed": item["seed"],
                    "runtime_seconds": item["runtimeSeconds"],
                    "status": item["status"],
                    "runtime_blocked_rows": int(item.get("runtimeBlockedRows") or 0),
                    "actual_model": item["actualModel"],
                }
                for item in manifests
            ],
            "runtime_profiles": runtime_profiles,
            "reasoning_effort": first_manifest.get("reasoningEffort"),
            "converter_model": first_manifest.get("converterModel"),
        },
        "output_file": unique_output_files,
        "scores": {
            "runs": [
                {
                    "run": result["run"],
                    "hybrid6_macro_f1": result["hybrid6MacroF1"],
                    "field_f1": {
                        field: result["fieldMetrics"][field]["f1"] for field in FIELDS
                    },
                    "overall_micro_f1": result["overall"]["f1"],
                    "legacy_o_t_n_f1": {
                        field: legacy_field_f1(result, field)
                        for field in ("ObjectName", "Trend", "Num")
                    },
                    "format_compliance_rate": result["formatComplianceRate"],
                    "runtime_blocked_rows": int(result.get("runtimeBlockedRows") or 0),
                    "semantic_audit": result["semanticAudit"],
                }
                for result in ordered
            ],
            "all_10": {
                "hybrid6_macro_f1": score_stats(primary_values),
                "field_f1": field_stats,
                "overall_micro_f1": score_stats([
                    result["overall"]["f1"] for result in ordered
                ]),
                "format_compliance_rate": score_stats([
                    result["formatComplianceRate"] for result in ordered
                ]),
            },
            "top_3": {
                "runs": [int(result["run"]) for result in top],
                "hybrid6_macro_f1_mean": (
                    statistics.mean(float(result["hybrid6MacroF1"]) for result in top)
                    if top else None
                ),
            },
            "legacy_o_t_n": legacy,
            "hybrid6_macro_f1": score_stats(primary_values),
            "format_compliance_rate": score_stats([
                result["formatComplianceRate"] for result in ordered
            ]),
            "completion_status": (
                f"runtime_blocked_{runtime_blocked_rows}_rows_across_{len(ordered)}_runs"
                if runtime_blocked_rows else f"completed_{len(ordered)}_of_{len(ordered)}"
            ),
            "warnings": [],
        },
    }


def canonical_result_hash(record: Mapping[str, Any]) -> str:
    return sha256_text(
        json.dumps(record.get("result", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def converter_diagnostics(
    output_root: Path,
    aggregates: Sequence[dict[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = float(config["diagnostics"]["converterCollapseRowRate"])
    dominance_delta = float(config["diagnostics"]["converterDominanceF1Delta"])
    by_key = {
        (str(manifest["outputId"]), int(manifest["run"])): manifest
        for manifest in manifests
    }
    official_converter_ids = sorted({
        str(manifest["outputId"])
        for manifest in manifests
        if manifest.get("official") and manifest.get("effectiveRoute") == "retriever-converter"
    })
    output_mode = {
        str(manifest["outputId"]): str(manifest["promptMode"])
        for manifest in manifests
    }
    row_hashes: dict[tuple[str, int], list[str]] = {}
    candidate_hashes: dict[tuple[str, int], list[str]] = {}
    for output_id in official_converter_ids:
        for run in range(1, 11):
            manifest = by_key[(output_id, run)]
            predictions = read_jsonl(Path(str(manifest["files"]["predictions"])))
            row_hashes[(output_id, run)] = [canonical_result_hash(record) for record in predictions]
            candidate_path = output_root / "cases" / output_id / f"run_{run:02d}" / "retriever_candidates.jsonl"
            if candidate_path.is_file():
                candidate_hashes[(output_id, run)] = [
                    str(record.get("candidateSha256") or "")
                    for record in read_jsonl(candidate_path)
                ]

    pair_counts: dict[tuple[str, str], list[int]] = {}
    for run in range(1, 11):
        for mode in ("original", "zero-shot", "many-shot", "dynamic-shot"):
            ids = [
                output_id for output_id in official_converter_ids
                if output_mode[output_id] == mode
            ]
            for left_index, left in enumerate(ids):
                for right in ids[left_index + 1:]:
                    left_rows = row_hashes[(left, run)]
                    right_rows = row_hashes[(right, run)]
                    key = (left, right)
                    counts = pair_counts.setdefault(key, [0, 0])
                    counts[0] += sum(a == b for a, b in zip(left_rows, right_rows))
                    counts[1] += min(len(left_rows), len(right_rows))
    collapse_warnings = [
        {
            "type": "converter_collapse_warning",
            "left": left,
            "right": right,
            "identicalRows": counts[0],
            "comparedRows": counts[1],
            "identicalRate": counts[0] / counts[1],
        }
        for (left, right), counts in sorted(pair_counts.items())
        if counts[1] and counts[0] / counts[1] >= threshold
    ]

    aggregate_by_id = {
        item["model"]["output_id"]: item for item in aggregates
    }
    control_by_mode = {
        item["prompt"]["mode"]: item
        for item in aggregates
        if item["model"]["output_id"].startswith("control_converter_")
    }
    dominance_warnings: list[dict[str, Any]] = []
    for output_id in official_converter_ids:
        aggregate = aggregate_by_id[output_id]
        control = control_by_mode.get(aggregate["prompt"]["mode"])
        if not control:
            continue
        formal_mean = aggregate["scores"]["all_10"]["hybrid6_macro_f1"]["mean"]
        control_mean = control["scores"]["all_10"]["hybrid6_macro_f1"]["mean"]
        delta = abs(float(formal_mean) - float(control_mean))
        if delta < dominance_delta:
            dominance_warnings.append({
                "type": "converter_dominance_warning",
                "outputId": output_id,
                "controlId": control["model"]["output_id"],
                "formalF1": formal_mean,
                "controlF1": control_mean,
                "absoluteDelta": delta,
            })

    uniqueness: list[dict[str, Any]] = []
    for item in aggregates:
        output_id = item["model"]["output_id"]
        case_manifests = [
            manifest for manifest in manifests if manifest["outputId"] == output_id
        ]
        prediction_hashes = [str(manifest["hashes"]["predictions"]) for manifest in case_manifests]
        candidate_values = [
            value
            for run in range(1, 11)
            for value in candidate_hashes.get((output_id, run), [])
        ]
        uniqueness.append({
            "outputId": output_id,
            "predictionFileHashes": len(set(prediction_hashes)),
            "predictionRuns": len(prediction_hashes),
            "candidateRowHashes": len(set(candidate_values)) if candidate_values else None,
            "candidateRows": len(candidate_values) if candidate_values else 0,
        })
    return {
        "converterCollapseWarnings": collapse_warnings,
        "converterDominanceWarnings": dominance_warnings,
        "uniqueness": uniqueness,
        "rankingInterpretationStatus": (
            "withheld_converter_dominance"
            if dominance_warnings else "eligible"
        ),
    }


def markdown_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 6 narrative2 評估 v2",
        "",
        f"- 狀態：{report['status']}",
        f"- 正式 case：{report['completedOfficialCases']}/54；每組 10 runs × 85 rows",
        f"- 正式預測：{report['formalPredictions']}；control：{report['controlPredictions']}",
        f"- 排名解讀：{report['diagnostics']['rankingInterpretationStatus']}",
        f"- data SHA-256：{report['dataSha256']}",
        f"- gold SHA-256：{report['goldTargetsSha256']}",
        "",
        "## All-10 exploratory ordering",
        "",
        "| 順序 | output_id | model | prompt | hybrid6 mean | sample SD | min | max | format |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(report["exploratoryOrdering"], start=1):
        stats = item["scores"]["all_10"]["hybrid6_macro_f1"]
        fmt = item["scores"]["all_10"]["format_compliance_rate"]["mean"]
        lines.append(
            f"| {rank} | {item['model']['output_id']} | {item['model']['requested']} | "
            f"{item['prompt']['mode']} | {stats['mean']:.6f} | {stats['sampleSd']:.6f} | "
            f"{stats['min']:.6f} | {stats['max']:.6f} | {fmt:.6f} |"
        )
    lines.extend([
        "",
        "## 評估方法",
        "",
        "1. 先在同一 Source 內以 DataName 與 Position 的 typed JSON identity 對齊；大小寫、型別、陣列順序或值任一不同，即整筆 anchor 形成 FP/FN，且禁止跨列配對。",
        "2. ObjectName 先以 NFKC 與空白規格化做一對一精確配對；剩餘 mention 交由盲化 GPT-5.5 medium。只接受同義或明確共指；上下位詞、關聯公司與相近概念皆拒絕。",
        "3. Trend 先以版本化方向詞表正規化；未決者由 judge 同時核對方向、期間、基準與範圍。反向或基準不同必拒絕。",
        "4. Num 不使用 LLM；解析逗號、正負號、貨幣、百分比與 thousand/million/billion，再做一對一 math.isclose(rel_tol=1e-9, abs_tol=1e-9)。百分比保留 percentage-point flag，故裸 0.12 不等於 12%。",
        "5. Text 先以 NFKC 與空白規格化精確比對；未決者須維持完整主體、趨勢、數字、時間、範圍、基準與否定極性才接受。",
        "6. Judge request 隱去 case/model 與 gold/prediction 身份，A/B 順序由雜湊隨機交換；strict JSON schema、confidence ≥ 0.8 且 evidence span 可在 sourceText 驗證才計 TP。",
        "7. 每 run 獨立計 micro P/R/F1；主指標 hybrid6_macro_f1 為六欄 F1 等權平均。All-10 報 mean、sample SD、min/max；top-3 只作補充，沒有跨 run 聯集。",
        "8. 固定抽取 10% 語義決策交換 A/B 重判；不一致者第三次裁決，完整保存在 semantic_audit.json。",
        "9. 格式錯誤不修補正式結果：該列以空 prediction 評分；repair_predictions.nonformal.jsonl 僅供稽核。",
        "",
        "## Converter diagnostics",
        "",
        f"- collapse warnings：{len(report['diagnostics']['converterCollapseWarnings'])}",
        f"- dominance warnings：{len(report['diagnostics']['converterDominanceWarnings'])}",
    ])
    for warning in report["diagnostics"]["converterDominanceWarnings"][:30]:
        lines.append(
            f"- dominance: {warning['outputId']} vs {warning['controlId']}, "
            f"Δ={warning['absoluteDelta']:.6f}"
        )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_config(config_path)
    evaluation_bundle = workspace_path(config["evaluationBundle"]).resolve()
    bundle_integrity = verify_manifest(evaluation_bundle)
    gold_path = workspace_path(config["goldPath"]).resolve()
    if sha256_file(gold_path) != config["goldSha256"]:
        raise ProtocolError("gold target SHA-256 mismatch")
    gold_document = read_json(gold_path)
    targets = gold_document.get("rows") if isinstance(gold_document, dict) else None
    if not isinstance(targets, list) or len(targets) != int(config["expectedRows"]):
        raise ProtocolError("gold target row count mismatch")
    selected_targets = targets[: args.limit] if args.limit > 0 else targets

    output_root = args.output_root.resolve()
    generation_config_path = output_root / "generation_config.snapshot.json"
    generation_config = read_json(generation_config_path)
    if generation_config.get("protocol") != "experiment6-narrative2-full-v2":
        raise ProtocolError("generation config snapshot protocol mismatch")
    manifest_dir = output_root / "manifests"
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    if not manifest_paths:
        raise ProtocolError(f"no run manifests found in {manifest_dir}")
    manifests = [read_json(path) for path in manifest_paths]
    manifest_keys = [(item.get("outputId"), item.get("run")) for item in manifests]
    if len(set(manifest_keys)) != len(manifests):
        raise ProtocolError("run manifest files have duplicate outputId/run keys")
    if any(not output_id or not isinstance(run, int) for output_id, run in manifest_keys):
        raise ProtocolError("run manifest file has invalid outputId/run")

    partial = bool(args.only_case or args.limit > 0)
    if not partial:
        expected_total = (
            int(config["expectedOfficialCases"]) + int(config["expectedDiagnosticCases"])
        ) * int(config["expectedRuns"])
        official_ids = {str(item["outputId"]) for item in manifests if item.get("official")}
        control_ids = {str(item["outputId"]) for item in manifests if not item.get("official")}
        run_coverage = {
            output_id: {int(item["run"]) for item in manifests if item["outputId"] == output_id}
            for output_id in official_ids | control_ids
        }
        incomplete = {
            output_id: sorted(runs)
            for output_id, runs in run_coverage.items()
            if runs != set(range(1, int(config["expectedRuns"]) + 1))
        }
        blocked = [
            item for item in manifests
            if item.get("status") not in {"completed", "completed_with_format_errors"}
        ]
        gate = {
            "expectedCaseRuns": expected_total,
            "reportedCaseRuns": len(manifests),
            "officialCases": len(official_ids),
            "controlCases": len(control_ids),
            "incompleteRunCoverage": incomplete,
            "blockedCaseRuns": blocked,
        }
        if (
            len(manifests) != expected_total
            or len(official_ids) != int(config["expectedOfficialCases"])
            or len(control_ids) != int(config["expectedDiagnosticCases"])
            or incomplete
            or blocked
        ):
            progress_path = output_root / "evaluation" / "progress_only.json"
            write_json(progress_path, {
                "time": utc_now(),
                "status": "incomplete_no_ranking",
                "gate": gate,
            })
            raise ProtocolError(f"full-matrix completion gate failed; see {progress_path}")

    if args.only_case:
        selected = set(args.only_case)
        manifests = [item for item in manifests if item.get("outputId") in selected]
        missing = selected - {str(item.get("outputId") or "") for item in manifests}
        if missing:
            raise ProtocolError(f"unknown --only-case values: {sorted(missing)}")
        expected_runs = set(range(1, int(config["expectedRuns"]) + 1))
        incomplete = {
            output_id: sorted(
                int(item["run"])
                for item in manifests
                if str(item["outputId"]) == output_id
            )
            for output_id in selected
            if {
                int(item["run"])
                for item in manifests
                if str(item["outputId"]) == output_id
            } != expected_runs
        }
        if incomplete:
            raise ProtocolError(
                f"partial case evaluation requires all runs 1-{len(expected_runs)}: {incomplete}"
            )

    evaluation_root = output_root / "evaluation"
    write_json(evaluation_root / "evaluation_config.snapshot.json", config)
    self_test_path = evaluation_root / "fixed_evaluator_self_test.json"
    self_test = subprocess.run(
        [
            sys.executable, "-B",
            str(evaluation_bundle / "evaluate_narrative2_annotations.py"),
            "self-test", "--output", str(self_test_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if self_test.returncode != 0:
        raise ProtocolError(f"fixed evaluator self-test failed: {self_test.stderr}")

    results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    manifests_by_key = {
        (str(item["outputId"]), int(item["run"])): item for item in manifests
    }
    for manifest in sorted(manifests, key=lambda item: (str(item["outputId"]), int(item["run"]))):
        case_dir = (
            evaluation_root / "cases" / str(manifest["outputId"])
            / f"run_{int(manifest['run']):02d}"
        )
        fixed = run_fixed_baseline(evaluation_bundle, manifest, case_dir)
        try:
            hybrid = evaluate_case(
                manifest,
                selected_targets,
                config,
                case_dir,
                args.judge_disabled,
                judge_checkpoint_path=evaluation_root / "judge_checkpoint.jsonl",
            )
        except (ProtocolError, JudgeError) as error:
            blocker = {
                "outputId": manifest.get("outputId"),
                "run": manifest.get("run"),
                "status": "blocked",
                "failureCategory": (
                    "judge_blocked" if isinstance(error, JudgeError)
                    else "evaluation_protocol_blocked"
                ),
                "error": str(error),
                "fixedBaseline": fixed,
            }
            blockers.append(blocker)
            write_json(case_dir / "evaluation_blocker.json", blocker)
            continue
        hybrid["fixedBaseline"] = fixed
        blocker_path = case_dir / "evaluation_blocker.json"
        if blocker_path.is_file():
            blocker_path.replace(case_dir / "evaluation_blocker.resolved.json")
        write_json(case_dir / "metrics.json", hybrid)
        write_json(case_dir / "hybrid_evaluation.json", hybrid)
        results.append(hybrid)

    if blockers:
        write_json(evaluation_root / "evaluation_progress.json", {
            "time": utc_now(),
            "status": "blocked_no_ranking",
            "completedCaseRuns": len(results),
            "blockedCaseRuns": blockers,
        })
        raise ProtocolError("evaluation has blocked case-runs; no ranking published")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["outputId"])].append(result)
    aggregate_config = {
        **generation_config,
        "topK": int(config["topK"]),
    }
    aggregates = [
        aggregate_case(
            output_id,
            run_results,
            manifests_by_key,
            aggregate_config,
        )
        for output_id, run_results in sorted(grouped.items())
    ]
    official = [
        item for item in aggregates
        if not item["model"]["output_id"].startswith("control_converter_")
    ]
    controls = [
        item for item in aggregates
        if item["model"]["output_id"].startswith("control_converter_")
    ]
    if not partial and (len(official) != 54 or len(controls) != 4):
        raise ProtocolError("aggregated result count is not 54 official + 4 controls")
    diagnostics = converter_diagnostics(
        output_root, aggregates, manifests, config
    ) if not partial else {
        "converterCollapseWarnings": [],
        "converterDominanceWarnings": [],
        "uniqueness": [],
        "rankingInterpretationStatus": "development_partial_no_ranking",
    }
    warning_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for warning in diagnostics["converterCollapseWarnings"]:
        warning_by_id[warning["left"]].append(warning)
        warning_by_id[warning["right"]].append(warning)
    for warning in diagnostics["converterDominanceWarnings"]:
        warning_by_id[warning["outputId"]].append(warning)
    for item in official:
        item["scores"]["warnings"] = warning_by_id[item["model"]["output_id"]]

    ordering = sorted(
        official,
        key=lambda item: (
            item["scores"]["all_10"]["hybrid6_macro_f1"]["mean"],
            item["scores"]["all_10"]["format_compliance_rate"]["mean"],
        ),
        reverse=True,
    )
    report = {
        "time": utc_now(),
        "protocol": config["protocol"],
        "status": "development_partial_no_ranking" if partial else "completed",
        "experimentId": output_root.name,
        "completedOfficialCases": len(official),
        "completedControlCases": len(controls),
        "formalPredictions": len(official) * len(selected_targets) * int(config["expectedRuns"]),
        "controlPredictions": len(controls) * len(selected_targets) * int(config["expectedRuns"]),
        "dataSha256": generation_config["inputWorkbook"]["sha256"],
        "goldTargets": str(gold_path),
        "goldTargetsSha256": sha256_file(gold_path),
        "evaluationBundleSha256": bundle_integrity["manifestSha256"],
        "exploratoryOrdering": ordering,
        "controls": controls,
        "diagnostics": diagnostics,
    }
    write_json(evaluation_root / "evaluation_report.json", report)
    (evaluation_root / "evaluation_report.md").write_text(
        markdown_summary(report), encoding="utf-8"
    )
    write_json(evaluation_root / "evaluation_progress.json", {
        "time": report["time"],
        "status": report["status"],
        "completedCaseRuns": len(results),
        "blockedCaseRuns": [],
        "completedOfficialCases": len(official),
        "completedControlCases": len(controls),
        "formalPredictions": report["formalPredictions"],
        "rankingInterpretationStatus": diagnostics["rankingInterpretationStatus"],
    })
    if not partial:
        result_path = output_root / "experiment6_results.json"
        write_json(result_path, official)
        tsv_path = output_root / "experiment6_results.tsv"
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "prompt", "runtime", "output_file", "scores"],
                delimiter="\t",
            )
            writer.writeheader()
            for item in official:
                writer.writerow({
                    key: json.dumps(item[key], ensure_ascii=False, sort_keys=True)
                    for key in ("model", "prompt", "runtime", "output_file", "scores")
                })
    return report

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate narrative2 bindings with field-specific methods."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "experiment6_narrative2_evaluation.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--judge-disabled", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = build(parse_args(argv))
    except (ProtocolError, JudgeError) as error:
        print(
            json.dumps(
                {
                    "time": utc_now(),
                    "protocol": "narrative2-hybrid-v2",
                    "status": "blocked",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return (
        0
        if report["status"] in {"completed", "development_partial_no_ranking"}
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
