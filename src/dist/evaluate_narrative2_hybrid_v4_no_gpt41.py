#!/usr/bin/env python3
"""Experiment 6 narrative2 hybrid-v4 evaluator for the 34-case no-GPT-4.1 scope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import evaluate_narrative2_hybrid as base
from experiment6_paths import PATHS


REPO_ROOT = PATHS.repo
WORKSPACE_ROOT = PATHS.workspace
FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
PROTOCOL = "narrative2-hybrid-v4-no-gpt41"
JUDGE_VERSION = "narrative2-semantic-judge-v4-example26-fixed-index"
BASE_JUDGE_SYSTEM_PROMPT = base.JUDGE_SYSTEM_PROMPT
MODE_SUFFIX = {
    "zero-shot": "z",
    "many-shot": "m",
    "dynamic-shot": "d",
}


class ProtocolError(RuntimeError):
    """Raised when generation or evaluation artifacts violate the v4 contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def workspace_path(raw: str | Mapping[str, Any]) -> Path:
    if isinstance(raw, Mapping):
        return PATHS.resolve_locator(raw)
    path = Path(raw)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    return rows


def expected_output_sources(
    generation_config: Mapping[str, Any], excluded_source_ids: set[str]
) -> dict[str, str]:
    """Derive the exact formal output IDs from the frozen generation matrix."""

    outputs: dict[str, str] = {}
    for part in generation_config.get("parts", []):
        for case in part.get("cases", []):
            source_id = str(case["sourceId"])
            if source_id in excluded_source_ids:
                continue
            output_id = str(case["outputId"])
            if output_id in outputs:
                raise ProtocolError(f"duplicate output ID in generation matrix: {output_id}")
            outputs[output_id] = source_id
        for model in part.get("models", []):
            source_id = str(model["sourceId"])
            if source_id in excluded_source_ids:
                continue
            for prompt_mode in part.get("promptModes", []):
                try:
                    suffix = MODE_SUFFIX[str(prompt_mode)]
                except KeyError as error:
                    raise ProtocolError(
                        f"unsupported matrix prompt mode: {prompt_mode}"
                    ) from error
                output_id = f"{model['outputStem']}_{suffix}"
                if output_id in outputs:
                    raise ProtocolError(
                        f"duplicate output ID in generation matrix: {output_id}"
                    )
                outputs[output_id] = source_id
    return outputs


def validate_manifest_scope(
    manifests: Sequence[Mapping[str, Any]],
    expected_outputs: Mapping[str, str],
    excluded_source_ids: set[str],
) -> None:
    """Reject excluded or mislabeled artifacts before any score is computed."""

    excluded_manifests = [
        item for item in manifests if str(item.get("sourceId")) in excluded_source_ids
    ]
    if excluded_manifests:
        raise ProtocolError(
            "excluded-source manifests are present in the formal no-GPT-4.1 scope"
        )
    unexpected_manifests = [
        item
        for item in manifests
        if str(item.get("outputId")) not in expected_outputs
        or expected_outputs.get(str(item.get("outputId")))
        != str(item.get("sourceId"))
    ]
    if unexpected_manifests:
        raise ProtocolError("manifest output/source is outside the 34-case matrix")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for value in values
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    if tp == fp == fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def zero_counts() -> dict[str, dict[str, int]]:
    return {field: {"tp": 0, "fp": 0, "fn": 0} for field in FIELDS}


def add_counts(
    target: dict[str, dict[str, int]], source: Mapping[str, Mapping[str, int]]
) -> None:
    for field in FIELDS:
        for key in ("tp", "fp", "fn"):
            target[field][key] += int(source[field][key])


def scalar_counts(passed: bool) -> dict[str, int]:
    return {"tp": 1, "fp": 0, "fn": 0} if passed else {
        "tp": 0, "fp": 1, "fn": 1
    }


def normalize_object_exact(value: Any) -> str:
    """Versioned ObjectName exact path: NFKC, trim, casefold, whitespace."""

    return " ".join(base.normalize_unicode(value).strip().casefold().split())


def normalize_text_exact(value: Any) -> str:
    """Versioned Text exact path: outer trim and lowercase only."""

    return str(value).strip().lower()


def data_name_equal(left: Any, right: Any) -> bool:
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and left.strip().lower() == right.strip().lower()
    )


def position_equal(left: Any, right: Any) -> bool:
    """Typed JSON identity; array/value order is hard, object key order is not."""

    return base.fixed_canonical(left) == base.fixed_canonical(right)


def is_absent(value: Any) -> bool:
    """Match the authoritative fixed evaluator missing-value contract."""

    if value is None:
        return True
    if isinstance(value, list):
        return not value or all(is_absent(item) for item in value)
    if isinstance(value, str):
        return value.strip().lower() in {"", "none"}
    return False


def trend_alias(value: Any, aliases: Mapping[str, str]) -> str | None:
    if is_absent(value):
        return None
    normalized = base.normalize_soft_string(value)
    return aliases.get(normalized, normalized)


def is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def strict_numeric_array(value: Any) -> list[float] | None:
    if is_absent(value):
        return []
    if not isinstance(value, list) or not all(is_finite_number(item) for item in value):
        return None
    return [float(item) for item in value]


def numeric_equal(left: Any, right: Any, abs_tol: float, rel_tol: float) -> bool:
    gold = strict_numeric_array(left)
    prediction = strict_numeric_array(right)
    if gold is None or prediction is None or len(gold) != len(prediction):
        return False
    remaining = list(prediction)
    for gold_value in gold:
        match = next(
            (
                index
                for index, predicted_value in enumerate(remaining)
                if math.isclose(
                    gold_value,
                    predicted_value,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )
            ),
            None,
        )
        if match is None:
            return False
        remaining.pop(match)
    return not remaining


def validate_prediction_binding(binding: Any, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(binding, dict):
        return [f"{where} must be an object"]
    missing = set(FIELDS) - set(binding)
    extras = set(binding) - set(FIELDS)
    if missing:
        errors.append(f"{where} missing fields {sorted(missing)}")
    if extras:
        errors.append(f"{where} has extra fields {sorted(extras)}")
    if missing or extras:
        return errors
    objects = binding["ObjectName"]
    if not (
        isinstance(objects, list)
        and objects
        and all(
            isinstance(item, str) and item.strip() and not is_absent(item)
            for item in objects
        )
    ):
        errors.append(f"{where}.ObjectName must be a non-empty string array")
    if not isinstance(binding["DataName"], str):
        errors.append(f"{where}.DataName must be a string")
    position = binding["Position"]
    if not isinstance(position, list) or not all(
        isinstance(item, dict) for item in position
    ):
        errors.append(f"{where}.Position must be an array of objects")
    trend = binding["Trend"]
    if not is_absent(trend) and not isinstance(trend, str):
        errors.append(f"{where}.Trend must be a string or absent")
    if strict_numeric_array(binding["Num"]) is None:
        errors.append(f"{where}.Num must be a finite JSON number array or absent")
    if not isinstance(binding["Text"], str):
        errors.append(f"{where}.Text must be a string")
    return errors


def row_gate(
    gold: Sequence[Mapping[str, Any]], prediction_record: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    if not prediction_record.get("formatValid"):
        errors.append("generation_format_invalid")
    predicted = prediction_record.get("result")
    if not isinstance(predicted, list):
        return errors + ["result_must_be_array"]
    for index, binding in enumerate(predicted):
        errors.extend(validate_prediction_binding(binding, f"prediction[{index}]"))
    if len(predicted) != len(gold):
        errors.append(
            f"binding_count_mismatch: predicted={len(predicted)} gold={len(gold)}"
        )
        return errors
    if not errors:
        mismatches = [
            index
            for index, (gold_binding, prediction_binding) in enumerate(
                zip(gold, predicted)
            )
            if not (
                data_name_equal(gold_binding["DataName"], prediction_binding["DataName"])
                and position_equal(gold_binding["Position"], prediction_binding["Position"])
            )
        ]
        if mismatches:
            errors.append(f"fixed_binding_order_or_anchor_mismatch: {mismatches}")
    return errors


def rejected_counts(
    gold: Sequence[Mapping[str, Any]], predicted: Any
) -> dict[str, dict[str, int]]:
    counts = zero_counts()
    for field in FIELDS:
        counts[field]["fn"] = len(gold)
    if not isinstance(predicted, list):
        return counts
    for binding in predicted:
        if not isinstance(binding, dict):
            continue
        if (
            "ObjectName" in binding
            and isinstance(binding["ObjectName"], list)
            and bool(binding["ObjectName"])
            and all(
                isinstance(item, str)
                and bool(item.strip())
                and not is_absent(item)
                for item in binding["ObjectName"]
            )
        ):
            counts["ObjectName"]["fp"] += 1
        if "DataName" in binding and isinstance(binding["DataName"], str):
            counts["DataName"]["fp"] += 1
        if (
            "Position" in binding
            and isinstance(binding["Position"], list)
            and all(isinstance(item, dict) for item in binding["Position"])
        ):
            counts["Position"]["fp"] += 1
        if (
            "Trend" in binding
            and (is_absent(binding["Trend"]) or isinstance(binding["Trend"], str))
        ):
            counts["Trend"]["fp"] += 1
        if "Num" in binding and strict_numeric_array(binding["Num"]) is not None:
            counts["Num"]["fp"] += 1
        if "Text" in binding and isinstance(binding["Text"], str):
            counts["Text"]["fp"] += 1
    return counts


def build_semantic_plan(
    input_data: Any,
    gold: Mapping[str, Any],
    prediction: Mapping[str, Any],
    binding_index: int,
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    """Build the hybrid-v4 semantic plan with field-specific exact rules."""

    gold_objects = gold.get("ObjectName")
    predicted_objects = prediction.get("ObjectName")
    exact_pairs = base.maximum_matching(
        len(gold_objects) if isinstance(gold_objects, list) else 0,
        len(predicted_objects) if isinstance(predicted_objects, list) else 0,
        lambda left, right: (
            isinstance(gold_objects[left], str)
            and isinstance(predicted_objects[right], str)
            and normalize_object_exact(gold_objects[left])
            == normalize_object_exact(predicted_objects[right])
        ),
    )
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
    object_decision_id = f"binding_{binding_index}_ObjectName"
    decisions: list[dict[str, Any]] = []
    if unmatched_gold and unmatched_prediction:
        decisions.append({
            "decisionId": object_decision_id,
            "field": "ObjectName",
            "gold": [gold_objects[index] for index in unmatched_gold],
            "prediction": [
                predicted_objects[index] for index in unmatched_prediction
            ],
            "DataName": gold["DataName"],
            "Position": gold["Position"],
            "dataEvidence": base.build_data_evidence(input_data, gold),
        })

    gold_trend = gold.get("Trend")
    predicted_trend = prediction.get("Trend")
    trend_decision_id = f"binding_{binding_index}_Trend"
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
        decisions.append({
            "decisionId": trend_decision_id,
            "field": "Trend",
            "gold": gold_trend,
            "prediction": predicted_trend,
            "DataName": gold["DataName"],
            "Position": gold["Position"],
            "dataEvidence": base.build_data_evidence(input_data, gold),
        })

    gold_text = gold.get("Text")
    predicted_text = prediction.get("Text")
    text_decision_id = f"binding_{binding_index}_Text"
    text_deterministic = (
        isinstance(gold_text, str)
        and isinstance(predicted_text, str)
        and normalize_text_exact(gold_text) == normalize_text_exact(predicted_text)
    )
    if not text_deterministic:
        decisions.append({
            "decisionId": text_decision_id,
            "field": "Text",
            "gold": gold_text,
            "prediction": predicted_text,
            "DataName": gold["DataName"],
            "Position": gold["Position"],
        })
    return {
        "decisions": decisions,
        "goldObjects": gold_objects,
        "predictedObjects": predicted_objects,
        "exactObjectPairs": exact_pairs,
        "unmatchedGoldObjects": unmatched_gold,
        "unmatchedPredictedObjects": unmatched_prediction,
        "objectDecisionId": object_decision_id,
        "trendDecisionId": trend_decision_id,
        "trendDeterministic": trend_deterministic,
        "textDecisionId": text_decision_id,
        "textDeterministic": text_deterministic,
    }


def object_field_pass(
    plan: Mapping[str, Any], judge_results: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, list[dict[str, int]], Mapping[str, Any] | None]:
    pairs = list(plan["exactObjectPairs"])
    judgment = judge_results.get(str(plan["objectDecisionId"]))
    unmatched_gold = plan["unmatchedGoldObjects"]
    unmatched_prediction = plan["unmatchedPredictedObjects"]
    if judgment and judgment.get("accepted"):
        used_gold: set[int] = set()
        used_prediction: set[int] = set()
        for pair in judgment.get("matchedPairs", []):
            left = pair.get("goldIndex") if isinstance(pair, dict) else None
            right = pair.get("predictionIndex") if isinstance(pair, dict) else None
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
                pairs.append((unmatched_gold[left], unmatched_prediction[right]))
    gold_size = len(plan["goldObjects"]) if isinstance(plan["goldObjects"], list) else 0
    prediction_size = (
        len(plan["predictedObjects"])
        if isinstance(plan["predictedObjects"], list)
        else 0
    )
    passed = len(pairs) == gold_size == prediction_size
    return passed, [
        {"goldIndex": left, "predictionIndex": right}
        for left, right in sorted(pairs)
    ], judgment


def resolve_judge_manifest_path(
    output_root: Path, item: Mapping[str, Any]
) -> Path:
    declared = Path(str(item["path"]))
    if declared.is_file():
        return declared
    relocated = output_root / "judge_examples" / declared.name
    if relocated.is_file():
        return relocated
    raise ProtocolError(f"judge prompt prefix missing: {declared}")


def configure_judge(
    output_root: Path, config: Mapping[str, Any], evaluation_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    manifest_path = output_root / "judge_examples" / "manifest.json"
    if not manifest_path.is_file():
        raise ProtocolError(f"judge example manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("validation", {}).get("canonicalRows") != 26:
        raise ProtocolError("judge example bundle is not 26 canonical rows")
    prefix_report: dict[str, Any] = {}
    system_prompts: dict[str, str] = {}
    validation_versions: dict[str, str] = {}
    for field in ("ObjectName", "Trend", "Text"):
        item = manifest["files"]["promptPrefixes"][field]
        path = resolve_judge_manifest_path(output_root, item)
        if sha256_file(path) != item["sha256"]:
            raise ProtocolError(f"judge prompt prefix SHA mismatch: {field}")
        rendered = path.read_text(encoding="utf-8")
        system_prompt = BASE_JUDGE_SYSTEM_PROMPT + "\n\n" + rendered
        lexical_estimate = len(base.re.findall(r"\w+|[^\w\s]", system_prompt))
        max_input = int(config["evaluation"]["judge"]["maxInputTokens"])
        reserved = int(config["evaluation"]["judge"]["reservedRuntimeTokens"])
        if lexical_estimate + reserved > max_input:
            raise ProtocolError(
                f"{field} judge prompt preflight exceeds limit: "
                f"prefix={lexical_estimate} reserved={reserved} max={max_input}"
            )
        prompt_path = evaluation_root / f"judge_system_prompt_{field}_v4.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(system_prompt, encoding="utf-8")
        prompt_sha = sha256_file(prompt_path)
        system_prompts[field] = system_prompt
        validation_versions[field] = f"{JUDGE_VERSION}-{field}-{prompt_sha[:16]}"
        prefix_report[field] = {
            **item,
            "resolvedPath": str(path),
            "systemPromptPath": str(prompt_path),
            "systemPromptSha256": prompt_sha,
            "systemPromptCharacters": len(system_prompt),
            "systemPromptLexicalTokenEstimate": lexical_estimate,
        }
    prompt_set_sha = sha256_text(json.dumps(
        {
            field: prefix_report[field]["systemPromptSha256"]
            for field in sorted(prefix_report)
        },
        sort_keys=True,
        separators=(",", ":"),
    ))
    return dict(config["evaluation"]["judge"]), {
        "sha256": prompt_set_sha,
        "fieldSpecific": True,
        "fields": prefix_report,
        "reservedRuntimeTokens": int(
            config["evaluation"]["judge"]["reservedRuntimeTokens"]
        ),
        "maxInputTokens": int(config["evaluation"]["judge"]["maxInputTokens"]),
        "referenceManifest": str(manifest_path),
        "referenceManifestSha256": sha256_file(manifest_path),
    }, system_prompts, validation_versions


class FieldPromptSemanticJudge(base.SemanticJudge):
    """Route each semantic field to its own immutable 26-example prefix."""

    def __init__(
        self,
        config: Mapping[str, Any],
        checkpoint_path: Path,
        system_prompts: Mapping[str, str],
        validation_versions: Mapping[str, str],
        *,
        disabled: bool,
    ) -> None:
        super().__init__(config, checkpoint_path, disabled=disabled)
        self.system_prompts = dict(system_prompts)
        self.validation_versions = dict(validation_versions)

    def decide(
        self,
        source: str,
        source_text: str,
        decisions: Sequence[Mapping[str, Any]],
        *,
        force_opposite: bool = False,
        audit_label: str = "primary",
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for decision in decisions:
            field = str(decision.get("field") or "")
            if field not in self.system_prompts:
                raise base.JudgeError(f"unsupported semantic judge field: {field!r}")
            grouped[field].append(decision)
        results: dict[str, dict[str, Any]] = {}
        original_prompt = base.JUDGE_SYSTEM_PROMPT
        original_version = base.JUDGE_VALIDATION_VERSION
        try:
            for field in ("ObjectName", "Trend", "Text"):
                if not grouped[field]:
                    continue
                base.JUDGE_SYSTEM_PROMPT = self.system_prompts[field]
                base.JUDGE_VALIDATION_VERSION = self.validation_versions[field]
                results.update(super().decide(
                    source,
                    source_text,
                    grouped[field],
                    force_opposite=force_opposite,
                    audit_label=f"{audit_label}:{field}",
                ))
        finally:
            base.JUDGE_SYSTEM_PROMPT = original_prompt
            base.JUDGE_VALIDATION_VERSION = original_version
        return results


def evaluate_run(
    manifest: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
    case_dir: Path,
    judge: base.SemanticJudge,
) -> dict[str, Any]:
    files = manifest.get("files")
    hashes = manifest.get("hashes")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ProtocolError("generation manifest lacks files/hashes")
    prediction_path = Path(str(files["predictions"]))
    if sha256_file(prediction_path) != hashes.get("predictions"):
        raise ProtocolError(f"prediction SHA mismatch: {prediction_path}")
    predictions = read_jsonl(prediction_path)
    if len(predictions) != len(targets):
        raise ProtocolError(
            f"prediction rows={len(predictions)} expected={len(targets)}"
        )
    target_sources = [str(target["source"]) for target in targets]
    prediction_sources = [str(record.get("source") or "") for record in predictions]
    if prediction_sources != target_sources or len(set(prediction_sources)) != len(predictions):
        raise ProtocolError("prediction source order/coverage mismatch")

    aliases = {
        base.normalize_soft_string(key): base.normalize_soft_string(value)
        for key, value in evaluation_config["trendAliases"].items()
    }
    numeric = evaluation_config["numericComparison"]
    totals = zero_counts()
    row_outputs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    judge_batches = 0

    for target, prediction_record in zip(targets, predictions):
        source = str(target["source"])
        gold_bindings = target["targetBindings"]
        predicted_bindings = prediction_record.get("result")
        gate_errors = row_gate(gold_bindings, prediction_record)
        if gate_errors:
            counts = rejected_counts(gold_bindings, predicted_bindings)
            add_counts(totals, counts)
            rejected_item = {
                "source": source,
                "run": manifest["run"],
                "errors": gate_errors,
                "goldBindings": len(gold_bindings),
                "predictedBindings": (
                    len(predicted_bindings) if isinstance(predicted_bindings, list) else 0
                ),
                "parserDiagnostic": prediction_record.get("parserDiagnostic"),
                "formalPolicy": "zero-tp; parsed prediction fields are fp; gold fields are fn",
            }
            rejected.append(rejected_item)
            row_outputs.append({
                **rejected_item,
                "status": "rejected_zero",
                "fieldMetrics": {
                    field: metric(**counts[field]) for field in FIELDS
                },
            })
            continue

        assert isinstance(predicted_bindings, list)
        plans: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        source_text = str(prediction_record.get("inputText") or "")
        input_data = prediction_record.get("inputData") or ""
        for index, (gold_binding, predicted_binding) in enumerate(
            zip(gold_bindings, predicted_bindings)
        ):
            plan = build_semantic_plan(
                input_data, gold_binding, predicted_binding, index, aliases
            )
            plans.append(plan)
            decisions.extend(plan["decisions"])
        judge_results = judge.decide(source, source_text, decisions)
        judge_batches += int(bool(decisions))
        for decision in decisions:
            audit_candidates.append({
                "source": source,
                "sourceText": source_text,
                "decision": decision,
                "primary": judge_results[str(decision["decisionId"])],
            })

        row_counts = zero_counts()
        binding_outputs: list[dict[str, Any]] = []
        for index, (gold_binding, predicted_binding, plan) in enumerate(
            zip(gold_bindings, predicted_bindings, plans)
        ):
            object_pass, object_pairs, object_judgment = object_field_pass(
                plan, judge_results
            )
            trend_deterministic = plan["trendDeterministic"]
            trend_judgment = judge_results.get(str(plan["trendDecisionId"]))
            trend_pass = (
                bool(trend_deterministic)
                if trend_deterministic is not None
                else bool(trend_judgment and trend_judgment.get("accepted"))
            )
            text_judgment = judge_results.get(str(plan["textDecisionId"]))
            text_pass = bool(plan["textDeterministic"]) or bool(
                text_judgment and text_judgment.get("accepted")
            )
            passes = {
                "ObjectName": object_pass,
                "DataName": data_name_equal(
                    gold_binding["DataName"], predicted_binding["DataName"]
                ),
                "Position": position_equal(
                    gold_binding["Position"], predicted_binding["Position"]
                ),
                "Trend": trend_pass,
                "Num": numeric_equal(
                    gold_binding["Num"],
                    predicted_binding["Num"],
                    float(numeric["absoluteTolerance"]),
                    float(numeric["relativeTolerance"]),
                ),
                "Text": text_pass,
            }
            for field, passed in passes.items():
                counts = scalar_counts(passed)
                for key in counts:
                    row_counts[field][key] += counts[key]
            binding_outputs.append({
                "bindingIndex": index,
                "passes": passes,
                "ObjectNameMatchedPairs": object_pairs,
                "judge": {
                    "ObjectName": object_judgment,
                    "Trend": trend_judgment,
                    "Text": text_judgment,
                },
            })
        add_counts(totals, row_counts)
        row_outputs.append({
            "source": source,
            "run": manifest["run"],
            "status": "accepted",
            "goldBindings": len(gold_bindings),
            "predictedBindings": len(predicted_bindings),
            "fieldMetrics": {
                field: metric(**row_counts[field]) for field in FIELDS
            },
            "bindings": binding_outputs,
        })

    field_metrics = {field: metric(**totals[field]) for field in FIELDS}
    macro = {
        name: statistics.mean(field_metrics[field][name] for field in FIELDS)
        for name in ("precision", "recall", "f1")
    }
    pooled = {key: sum(totals[field][key] for field in FIELDS) for key in ("tp", "fp", "fn")}
    overall = metric(**pooled)

    audit_config = evaluation_config["audit"]
    sample_size = (
        max(1, math.ceil(len(audit_candidates) * float(audit_config["sampleRate"])))
        if audit_candidates
        else 0
    )
    audit_seed = (
        int(audit_config["seed"])
        + int(manifest["run"])
        + int(sha256_text(str(manifest["outputId"]))[:8], 16)
    )
    sample_indexes = sorted(
        random.Random(audit_seed).sample(range(len(audit_candidates)), sample_size)
        if sample_size
        else []
    )
    audit_records: list[dict[str, Any]] = []
    agreements = 0
    for candidate_index in sample_indexes:
        candidate = audit_candidates[candidate_index]
        decision = candidate["decision"]
        decision_id = str(decision["decisionId"])
        swapped = judge.decide(
            candidate["source"],
            candidate["sourceText"],
            [decision],
            force_opposite=True,
            audit_label="swap-audit-v4",
        )[decision_id]
        primary_signature = (
            bool(candidate["primary"].get("accepted")),
            tuple(sorted(
                (pair["goldIndex"], pair["predictionIndex"])
                for pair in candidate["primary"].get("matchedPairs", [])
            )),
        )
        swapped_signature = (
            bool(swapped.get("accepted")),
            tuple(sorted(
                (pair["goldIndex"], pair["predictionIndex"])
                for pair in swapped.get("matchedPairs", [])
            )),
        )
        agreement = primary_signature == swapped_signature
        agreements += int(agreement)
        third = None
        if not agreement and audit_config.get("thirdAdjudicationOnDisagreement"):
            third = judge.decide(
                candidate["source"],
                candidate["sourceText"],
                [decision],
                audit_label="third-adjudication-v4",
            )[decision_id]
        audit_records.append({
            "candidateIndex": candidate_index,
            "source": candidate["source"],
            "decision": decision,
            "primary": candidate["primary"],
            "swapped": swapped,
            "agreement": agreement,
            "thirdAdjudication": third,
        })
    audit_summary = {
        "population": len(audit_candidates),
        "sampled": len(audit_records),
        "agreements": agreements,
        "disagreements": len(audit_records) - agreements,
        "agreementRate": agreements / len(audit_records) if audit_records else None,
        "thirdAdjudications": sum(
            record["thirdAdjudication"] is not None for record in audit_records
        ),
    }

    legacy = base.legacy_exact_otn_metrics(targets, predictions)
    result = {
        "time": utc_now(),
        "protocol": PROTOCOL,
        "outputId": manifest["outputId"],
        "sourceId": manifest["sourceId"],
        "promptMode": manifest["promptMode"],
        "part": manifest["part"],
        "run": manifest["run"],
        "requestedModel": manifest["requestedModel"],
        "actualModel": manifest["actualModel"],
        "predictionPath": str(prediction_path),
        "predictionSha256": hashes["predictions"],
        "fieldMetrics": field_metrics,
        "macro": macro,
        "overallMicro": overall,
        "formatComplianceRate": manifest["formatComplianceRate"],
        "acceptedRows": len(targets) - len(rejected),
        "rejectedRows": len(rejected),
        "runtimeBlockedRows": int(manifest.get("runtimeBlockedRows") or 0),
        "judge": {
            "model": evaluation_config["judge"]["model"],
            "reasoningEffort": evaluation_config["judge"]["reasoningEffort"],
            "liveCalls": judge.live_calls,
            "cacheHits": judge.cache_hits,
            "checkpoint": str(judge.checkpoint_path),
        },
        "semanticAudit": audit_summary,
        "legacyExactOtn": legacy,
        "rows": row_outputs,
        "status": (
            "runtime_blocked_scored_no_ranking"
            if int(manifest.get("runtimeBlockedRows") or 0)
            else "completed"
        ),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(case_dir / "gold_extracted.jsonl", [
        {
            "source": target["source"],
            "excelRow": target.get("excelRow"),
            "targetBindings": target["targetBindings"],
        }
        for target in targets
    ])
    write_jsonl(case_dir / "pred_extracted.jsonl", predictions)
    write_jsonl(case_dir / "rejected_records.jsonl", rejected)
    write_json(case_dir / "semantic_audit.json", {
        "summary": audit_summary,
        "records": audit_records,
    })
    write_json(case_dir / "extraction_report.json", {
        "rows": len(predictions),
        "acceptedRows": len(targets) - len(rejected),
        "rejectedRows": len(rejected),
        "predictionSha256": hashes["predictions"],
        "failurePolicy": "rejected-zero",
    })
    write_json(case_dir / "metrics.json", result)
    write_json(case_dir / "status.json", {
        "outputId": result["outputId"],
        "run": result["run"],
        "status": result["status"],
        "macro": macro,
        "acceptedRows": result["acceptedRows"],
        "rejectedRows": result["rejectedRows"],
    })
    return result


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


def metric_summary(
    runs: Sequence[Mapping[str, Any]], field: str | None, top: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    def value(run: Mapping[str, Any], name: str) -> float:
        source = run["fieldMetrics"][field] if field else run["macro"]
        return float(source[name])

    mean = {
        name: statistics.mean(value(run, name) for run in runs)
        for name in ("precision", "recall", "f1")
    }
    sample_sd = {
        name: statistics.stdev([value(run, name) for run in runs]) if len(runs) > 1 else 0.0
        for name in ("precision", "recall", "f1")
    }
    minimum = {
        name: min(value(run, name) for run in runs)
        for name in ("precision", "recall", "f1")
    }
    maximum = {
        name: max(value(run, name) for run in runs)
        for name in ("precision", "recall", "f1")
    }
    first = top[0]
    summary = {
        "mean": mean,
        "sample_sd": sample_sd,
        "min": minimum,
        "max": maximum,
        "top1": {
            "run_id": int(first["run"]),
            **{name: value(first, name) for name in ("precision", "recall", "f1")},
        },
        "top3": {
            "run_ids": [int(run["run"]) for run in top],
            **{
                name: statistics.mean(value(run, name) for run in top)
                for name in ("precision", "recall", "f1")
            },
        },
    }
    if field is not None and all(
        all(name in run["fieldMetrics"][field] for name in ("tp", "fp", "fn"))
        for run in runs
    ):
        pooled_counts = {
            name: sum(int(run["fieldMetrics"][field][name]) for run in runs)
            for name in ("tp", "fp", "fn")
        }
        summary["pooled_micro"] = metric(**pooled_counts)
    return summary


def aggregate_case(
    output_id: str,
    run_results: Sequence[Mapping[str, Any]],
    manifests_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(run_results, key=lambda item: int(item["run"]))
    if [int(item["run"]) for item in ordered] != list(range(1, 11)):
        raise ProtocolError(f"{output_id} does not contain runs 1-10")
    manifests = [
        manifests_by_key[(output_id, int(result["run"]))] for result in ordered
    ]
    top = sorted(
        ordered,
        key=lambda item: (
            -float(item["macro"]["f1"]),
            -float(item["macro"]["precision"]),
            -float(item["macro"]["recall"]),
            int(item["run"]),
        ),
    )[:3]
    fields = {
        field: metric_summary(ordered, field, top) for field in FIELDS
    }
    output_files: list[str] = []
    for manifest, result in zip(manifests, ordered):
        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        output_files.extend(str(value) for value in files.values())
        case_dir = Path(str(result["judge"]["checkpoint"])).parent / "cases" / output_id / f"run_{int(result['run']):02d}"
        output_files.extend(str(case_dir / name) for name in (
            "metrics.json", "status.json", "gold_extracted.jsonl",
            "pred_extracted.jsonl", "extraction_report.json",
            "rejected_records.jsonl", "semantic_audit.json",
            "fixed_strict_evaluation.json",
        ))
    first = manifests[0]
    prompt_mode = str(first["promptMode"])
    builder = generation_config["promptBuilder"]
    shot_count = (
        int(builder["manyShotCount"])
        if prompt_mode == "many-shot"
        else int(builder["dynamicShotCount"])
        if prompt_mode == "dynamic-shot"
        else 0
    )
    runtime_blocked_rows = sum(int(item.get("runtimeBlockedRows") or 0) for item in manifests)
    return {
        "model": {
            "output_id": output_id,
            "requested": first["requestedModel"],
            "actual": sorted({str(item.get("actualModel") or "") for item in manifests}),
            "adapter": sorted({str(item["adapter"]) for item in manifests if item.get("adapter")}),
            "quantization": sorted({str(item["quantization"]) for item in manifests if item.get("quantization")}),
        },
        "prompt": {
            "mode": prompt_mode,
            "part": first["part"],
            "route": first["route"],
            "effective_route": first.get("effectiveRoute"),
            "shot_count": shot_count,
            "prompt_hashes": sorted({str(item["hashes"]["prompts"]) for item in manifests}),
            "data_sha256": generation_config["inputWorkbook"]["sha256"],
            "generation_protocol": generation_config["protocol"],
            "evaluation_protocol": PROTOCOL,
        },
        "runtime": {
            "completion_status": (
                "runtime_blocked_no_ranking" if runtime_blocked_rows else "completed_10_of_10"
            ),
            "runtime_blocked_rows": runtime_blocked_rows,
            "runs": [
                {
                    "run": item["run"],
                    "seed": item["seed"],
                    "runtime_seconds": item["runtimeSeconds"],
                    "status": item["status"],
                    "actual_model": item["actualModel"],
                }
                for item in manifests
            ],
            "reasoning_effort": first.get("reasoningEffort"),
            "converter_model": first.get("converterModel"),
        },
        "output_file": list(dict.fromkeys(output_files)),
        "scores": {
            "evaluation_version": PROTOCOL,
            "selection": {
                "metric": "six-field-macro-f1",
                "tie_breakers": [
                    "six-field-macro-precision",
                    "six-field-macro-recall",
                    "run-id-ascending",
                ],
                "top1_run": int(top[0]["run"]),
                "top3_runs": [int(item["run"]) for item in top],
            },
            "fields": fields,
            "overall": metric_summary(ordered, None, top),
            "runs": [
                {
                    "run": result["run"],
                    "macro": result["macro"],
                    "fields": result["fieldMetrics"],
                    "overall_micro": result["overallMicro"],
                    "format_compliance_rate": result["formatComplianceRate"],
                    "accepted_rows": result["acceptedRows"],
                    "rejected_rows": result["rejectedRows"],
                    "semantic_audit": result["semanticAudit"],
                }
                for result in ordered
            ],
            "format_compliance_rate": score_stats([
                result["formatComplianceRate"] for result in ordered
            ]),
            "strict_baseline": {
                "role": "baseline_not_used_for_formal_ranking",
                "runs": [result.get("fixedBaseline") for result in ordered],
            },
            "legacy_o_t_n": [result["legacyExactOtn"] for result in ordered],
            "completion_status": (
                "runtime_blocked_no_ranking" if runtime_blocked_rows else "completed_10_of_10"
            ),
        },
    }


def load_effective_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(config_path)
    base_eval_path = workspace_path(config["baseEvaluationConfig"])
    evaluation = read_json(base_eval_path)
    evaluation.update(config["evaluation"])
    evaluation["protocol"] = PROTOCOL
    evaluation["expectedRows"] = int(config["expectedRows"])
    evaluation["expectedOfficialCases"] = int(config["expectedOfficialCases"])
    evaluation["expectedDiagnosticCases"] = 0
    evaluation["expectedRuns"] = int(config["expectedRuns"])
    evaluation["topK"] = 3
    evaluation["evaluationBundle"] = config["evaluation"]["evaluationBundle"]
    evaluation["goldPath"] = config["evaluation"]["goldPath"]
    evaluation["goldSha256"] = config["evaluation"]["goldSha256"]
    evaluation["numericComparison"] = config["evaluation"]["numericComparison"]
    evaluation["judge"] = config["evaluation"]["judge"]
    evaluation["audit"] = config["evaluation"]["audit"]
    return config, evaluation


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 6 narrative2 hybrid-v4（不含 GPT-4.1）",
        "",
        f"- status: `{report['status']}`",
        f"- official cases: {report['completedOfficialCases']} / {report['expectedOfficialCases']}",
        f"- formal predictions: {report['formalPredictions']}",
        f"- judge: `{report['judge']['model']}` / `{report['judge']['reasoningEffort']}`",
        f"- judge prompt SHA-256: `{report['judgePrompt']['sha256']}`",
        "",
        "正式排序採十次共同六欄 macro-F1 平均；top-1/top-3 共用同一組 run。",
        "",
        "| rank | output_id | mean macro-F1 | top-1 | top-3 |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, item in enumerate(report.get("ordering", []), start=1):
        scores = item["scores"]
        lines.append(
            f"| {rank} | {item['model']['output_id']} | "
            f"{scores['overall']['mean']['f1']:.6f} | "
            f"{scores['overall']['top1']['f1']:.6f} | "
            f"{scores['overall']['top3']['f1']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def write_flat_score_tables(
    output_root: Path, aggregates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Write review-friendly tables without changing the canonical JSON scores."""

    per_run_path = output_root / "experiment6_per_run_scores.tsv"
    per_run_fields = [
        "output_id", "requested_model", "actual_model", "part", "prompt_mode",
        "route", "run", "seed", "generation_status", "runtime_seconds",
        "format_compliance_rate", "accepted_rows", "rejected_rows",
        "macro_precision", "macro_recall", "macro_f1",
    ]
    for field in FIELDS:
        for metric_name in ("tp", "fp", "fn", "precision", "recall", "f1"):
            per_run_fields.append(f"{field}_{metric_name}")
    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields, delimiter="\t")
        writer.writeheader()
        for item in aggregates:
            runtime_by_run = {
                int(run["run"]): run for run in item["runtime"]["runs"]
            }
            for score in item["scores"]["runs"]:
                run_id = int(score["run"])
                runtime = runtime_by_run[run_id]
                row: dict[str, Any] = {
                    "output_id": item["model"]["output_id"],
                    "requested_model": item["model"]["requested"],
                    "actual_model": json.dumps(
                        item["model"]["actual"], ensure_ascii=False, sort_keys=True
                    ),
                    "part": item["prompt"]["part"],
                    "prompt_mode": item["prompt"]["mode"],
                    "route": item["prompt"]["route"],
                    "run": run_id,
                    "seed": runtime["seed"],
                    "generation_status": runtime["status"],
                    "runtime_seconds": runtime["runtime_seconds"],
                    "format_compliance_rate": score["format_compliance_rate"],
                    "accepted_rows": score["accepted_rows"],
                    "rejected_rows": score["rejected_rows"],
                    "macro_precision": score["macro"]["precision"],
                    "macro_recall": score["macro"]["recall"],
                    "macro_f1": score["macro"]["f1"],
                }
                for field in FIELDS:
                    for metric_name in ("tp", "fp", "fn", "precision", "recall", "f1"):
                        row[f"{field}_{metric_name}"] = score["fields"][field][metric_name]
                writer.writerow(row)

    field_summary_path = output_root / "experiment6_field_summary.tsv"
    field_summary_fields = [
        "output_id", "requested_model", "part", "prompt_mode", "field",
        "mean_precision", "mean_recall", "mean_f1",
        "sample_sd_precision", "sample_sd_recall", "sample_sd_f1",
        "min_precision", "min_recall", "min_f1",
        "max_precision", "max_recall", "max_f1",
        "top1_run", "top1_precision", "top1_recall", "top1_f1",
        "top3_runs", "top3_precision", "top3_recall", "top3_f1",
        "pooled_tp", "pooled_fp", "pooled_fn",
        "pooled_precision", "pooled_recall", "pooled_f1",
    ]
    with field_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=field_summary_fields, delimiter="\t"
        )
        writer.writeheader()
        for item in aggregates:
            for field in FIELDS:
                summary = item["scores"]["fields"][field]
                pooled = summary["pooled_micro"]
                writer.writerow({
                    "output_id": item["model"]["output_id"],
                    "requested_model": item["model"]["requested"],
                    "part": item["prompt"]["part"],
                    "prompt_mode": item["prompt"]["mode"],
                    "field": field,
                    "mean_precision": summary["mean"]["precision"],
                    "mean_recall": summary["mean"]["recall"],
                    "mean_f1": summary["mean"]["f1"],
                    "sample_sd_precision": summary["sample_sd"]["precision"],
                    "sample_sd_recall": summary["sample_sd"]["recall"],
                    "sample_sd_f1": summary["sample_sd"]["f1"],
                    "min_precision": summary["min"]["precision"],
                    "min_recall": summary["min"]["recall"],
                    "min_f1": summary["min"]["f1"],
                    "max_precision": summary["max"]["precision"],
                    "max_recall": summary["max"]["recall"],
                    "max_f1": summary["max"]["f1"],
                    "top1_run": summary["top1"]["run_id"],
                    "top1_precision": summary["top1"]["precision"],
                    "top1_recall": summary["top1"]["recall"],
                    "top1_f1": summary["top1"]["f1"],
                    "top3_runs": json.dumps(summary["top3"]["run_ids"]),
                    "top3_precision": summary["top3"]["precision"],
                    "top3_recall": summary["top3"]["recall"],
                    "top3_f1": summary["top3"]["f1"],
                    "pooled_tp": pooled["tp"],
                    "pooled_fp": pooled["fp"],
                    "pooled_fn": pooled["fn"],
                    "pooled_precision": pooled["precision"],
                    "pooled_recall": pooled["recall"],
                    "pooled_f1": pooled["f1"],
                })
    return {
        "perRun": {
            "path": str(per_run_path),
            "rows": len(aggregates) * 10,
            "sha256": sha256_file(per_run_path),
        },
        "fieldSummary": {
            "path": str(field_summary_path),
            "rows": len(aggregates) * len(FIELDS),
            "sha256": sha256_file(field_summary_path),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    config, evaluation = load_effective_config(args.config.resolve())
    output_root = args.output_root.resolve()
    generation_config_path = output_root / "generation_config.snapshot.json"
    if not generation_config_path.is_file():
        raise ProtocolError(f"generation config snapshot missing: {generation_config_path}")
    generation_config = read_json(generation_config_path)
    allowed_snapshot_cases = {
        int(value)
        for value in config.get(
            "generationSnapshotAllowedOfficialCases",
            [config.get("generationSnapshotExpectedOfficialCases", config["expectedOfficialCases"])],
        )
    }
    if int(generation_config.get("expectedOfficialCases", -1)) not in allowed_snapshot_cases:
        raise ProtocolError(
            "generation snapshot does not match the declared source matrix size"
        )
    excluded_source_ids = {str(value) for value in config.get("excludedSourceIds", [])}
    expected_outputs = expected_output_sources(generation_config, excluded_source_ids)
    expected_output_ids = set(expected_outputs)
    if len(expected_output_ids) != int(config["expectedOfficialCases"]):
        raise ProtocolError(
            "no-GPT-4.1 matrix does not match expectedOfficialCases: "
            f"{len(expected_output_ids)} != {config['expectedOfficialCases']}"
        )

    evaluation_bundle = workspace_path(evaluation["evaluationBundle"]).resolve()
    bundle_report = base.verify_manifest(evaluation_bundle)
    gold_path = workspace_path(evaluation["goldPath"]).resolve()
    if sha256_file(gold_path) != evaluation["goldSha256"]:
        raise ProtocolError("gold target SHA-256 mismatch")
    targets = read_json(gold_path).get("rows")
    if not isinstance(targets, list) or len(targets) != int(config["expectedRows"]):
        raise ProtocolError("gold target row count mismatch")

    manifest_paths = sorted((output_root / "manifests").glob("*.json"))
    manifests = [read_json(path) for path in manifest_paths]
    keys = [(item.get("outputId"), item.get("run")) for item in manifests]
    if len(keys) != len(set(keys)):
        raise ProtocolError("duplicate outputId/run manifests")
    validate_manifest_scope(manifests, expected_outputs, excluded_source_ids)
    expected_case_runs = int(config["expectedOfficialCases"]) * int(config["expectedRuns"])
    allowed = {"completed", "completed_with_format_errors"}
    blocked = [item for item in manifests if item.get("status") not in allowed]
    official_ids = sorted({str(item["outputId"]) for item in manifests if item.get("official")})
    run_coverage = {
        output_id: sorted(
            int(item["run"]) for item in manifests if item["outputId"] == output_id
        )
        for output_id in sorted(expected_output_ids)
    }
    incomplete = {
        output_id: runs
        for output_id, runs in run_coverage.items()
        if runs != list(range(1, int(config["expectedRuns"]) + 1))
    }
    full_gate = (
        len(manifests) == expected_case_runs
        and set(official_ids) == expected_output_ids
        and not incomplete
        and not blocked
    )
    evaluation_root = output_root / "evaluation_v4_no_gpt41"
    if not full_gate and not args.only_case:
        progress = {
            "time": utc_now(),
            "status": "incomplete_no_ranking",
            "expectedCaseRuns": expected_case_runs,
            "reportedCaseRuns": len(manifests),
            "officialCases": len(official_ids),
            "incompleteRunCoverage": incomplete,
            "blockedCaseRuns": blocked,
        }
        write_json(evaluation_root / "progress_only.json", progress)
        raise ProtocolError(f"full-matrix completion gate failed; see {evaluation_root / 'progress_only.json'}")

    if args.only_case:
        selected = set(args.only_case)
        unknown = selected - expected_output_ids
        if unknown:
            raise ProtocolError(f"unknown no-GPT-4.1 output IDs: {sorted(unknown)}")
        manifests = [item for item in manifests if str(item.get("outputId")) in selected]
        for output_id in selected:
            runs = sorted(int(item["run"]) for item in manifests if item["outputId"] == output_id)
            if runs != list(range(1, 11)):
                raise ProtocolError(f"{output_id} partial evaluation requires runs 1-10")

    write_json(evaluation_root / "evaluation_config.snapshot.json", evaluation)
    self_test_path = evaluation_root / "fixed_evaluator_self_test.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(evaluation_bundle / "evaluate_narrative2_annotations.py"),
            "self-test",
            "--output",
            str(self_test_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ProtocolError(f"fixed evaluator self-test failed: {completed.stderr}")
    (
        judge_config,
        judge_prompt_report,
        judge_system_prompts,
        judge_validation_versions,
    ) = configure_judge(
        output_root, config, evaluation_root
    )
    evaluation["judge"] = judge_config
    judge = FieldPromptSemanticJudge(
        judge_config,
        evaluation_root / "judge_checkpoint.jsonl",
        judge_system_prompts,
        judge_validation_versions,
        disabled=args.judge_disabled,
    )

    manifests_by_key = {
        (str(item["outputId"]), int(item["run"])): item for item in manifests
    }
    run_results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda item: (str(item["outputId"]), int(item["run"]))):
        case_dir = evaluation_root / "cases" / str(manifest["outputId"]) / f"run_{int(manifest['run']):02d}"
        try:
            fixed_baseline = base.run_fixed_baseline(
                evaluation_bundle, manifest, case_dir
            )
            if args.report_only:
                result = read_json(case_dir / "metrics.json")
            else:
                result = evaluate_run(
                    manifest, targets, evaluation, case_dir, judge
                )
                result["fixedBaseline"] = fixed_baseline
                write_json(case_dir / "metrics.json", result)
            run_results.append(result)
        except (ProtocolError, base.JudgeError, OSError, KeyError, ValueError) as error:
            blocker = {
                "outputId": manifest.get("outputId"),
                "run": manifest.get("run"),
                "status": "blocked",
                "error": str(error),
            }
            blockers.append(blocker)
            write_json(case_dir / "evaluation_blocker.json", blocker)
    if blockers:
        write_json(evaluation_root / "evaluation_progress.json", {
            "time": utc_now(),
            "status": "blocked_no_ranking",
            "completedCaseRuns": len(run_results),
            "blockedCaseRuns": blockers,
        })
        raise ProtocolError("evaluation has blocked case-runs; no ranking published")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in run_results:
        grouped[str(result["outputId"])].append(result)
    aggregates = [
        aggregate_case(output_id, results, manifests_by_key, generation_config)
        for output_id, results in sorted(grouped.items())
    ]
    ordering = sorted(
        aggregates,
        key=lambda item: (
            item["scores"]["overall"]["mean"]["f1"],
            item["scores"]["overall"]["mean"]["precision"],
            item["scores"]["overall"]["mean"]["recall"],
        ),
        reverse=True,
    )
    partial = bool(args.only_case)
    if not partial and len(aggregates) != int(config["expectedOfficialCases"]):
        raise ProtocolError(
            "aggregated result count mismatch: "
            f"{len(aggregates)} != {config['expectedOfficialCases']}"
        )
    report = {
        "time": utc_now(),
        "protocol": PROTOCOL,
        "status": "development_partial_no_ranking" if partial else "completed",
        "experimentId": output_root.name,
        "expectedOfficialCases": int(config["expectedOfficialCases"]),
        "completedOfficialCases": len(aggregates),
        "formalPredictions": (
            len(aggregates) * len(targets) * int(config["expectedRuns"])
        ),
        "dataSha256": generation_config["inputWorkbook"]["sha256"],
        "goldTargets": str(gold_path),
        "goldTargetsSha256": sha256_file(gold_path),
        "evaluationBundleSha256": bundle_report["manifestSha256"],
        "judge": {
            "model": judge_config["model"],
            "reasoningEffort": judge_config["reasoningEffort"],
            "liveCalls": judge.live_calls,
            "cacheHits": judge.cache_hits,
        },
        "judgePrompt": judge_prompt_report,
        "ordering": ordering,
    }
    if not partial:
        report["scoreTables"] = write_flat_score_tables(output_root, aggregates)
    write_json(evaluation_root / "evaluation_report.json", report)
    (evaluation_root / "evaluation_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    if not partial:
        write_json(output_root / "experiment6_results.json", aggregates)
        with (output_root / "experiment6_results.tsv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "prompt", "runtime", "output_file", "scores"],
                delimiter="\t",
            )
            writer.writeheader()
            for item in aggregates:
                writer.writerow({
                    key: json.dumps(item[key], ensure_ascii=False, sort_keys=True)
                    for key in writer.fieldnames
                })
    write_json(evaluation_root / "evaluation_progress.json", {
        "time": report["time"],
        "status": report["status"],
        "completedCaseRuns": len(run_results),
        "completedOfficialCases": len(aggregates),
        "formalPredictions": report["formalPredictions"],
        "blockedCaseRuns": [],
    })
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
    parser.add_argument("--only-case", action="append", default=[])
    parser.add_argument("--judge-disabled", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = build(parse_args(argv))
    except (ProtocolError, base.JudgeError) as error:
        print(json.dumps({
            "time": utc_now(),
            "protocol": PROTOCOL,
            "status": "blocked",
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
