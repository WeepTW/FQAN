#!/usr/bin/env python3
"""Fixed-protocol evaluator for FinFlier narrative annotations.

This is the Python counterpart of ``evaluate_narrative2_tn_fixed.mjs``.  It
compares the six binding fields with the same outer-whitespace/case
normalization and the same Trend/Num presence rules, while adding explicit
input gates so malformed records never silently reduce an evaluation
denominator.

The evaluator itself uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
TARGET_FIELDS = ("ObjectName", "Trend", "Num")
PRESENCE_FIELDS = ("Trend", "Num")
ABSENT = "__ABSENT__"
NON_FINITE = "__NON_FINITE__"


class ProtocolError(ValueError):
    """Raised when an input violates the fixed evaluation protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def is_absent(value: Any) -> bool:
    """Match the JavaScript evaluator's ``isAbsent`` semantics exactly."""

    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0 or all(is_absent(item) for item in value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized == "" or normalized == "none"
    return False


def canonical(value: Any) -> Any:
    """Return a typed canonical form suitable for strict structural equality.

    Type tags are necessary in Python because ``True == 1``.  Finite integers
    and floats intentionally share one numeric tag so 12 equals 12.0, matching
    JavaScript JSON.stringify(Number) behavior.
    """

    if is_absent(value):
        return ("absent", ABSENT)
    if isinstance(value, str):
        return ("string", value.strip().lower())
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return ("number", NON_FINITE)
        if not math.isfinite(number):
            return ("number", NON_FINITE)
        return ("number", number)
    if isinstance(value, list):
        return ("array", tuple(canonical(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                (str(key), canonical(value[key]))
                for key in sorted(value, key=lambda item: str(item))
            ),
        )
    return (type(value).__name__, value)


def same_fixed(left: Any, right: Any) -> bool:
    """Fixed equality used for every evaluated field."""

    return canonical(left) == canonical(right)


def finite_numeric_array(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        try:
            if not math.isfinite(float(item)):
                return False
        except OverflowError:
            return False
    return True


def non_empty_string_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(item, str)
            and item.strip() != ""
            and item.strip().lower() != "none"
            for item in value
        )
    )


def _strict_constant(token: str) -> Any:
    raise ProtocolError(f"non-standard JSON constant is not allowed: {token}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"), parse_constant=_strict_constant
        )
    except OSError as error:
        raise ProtocolError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProtocolError(
            f"invalid JSON in {path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error


def load_jsonl(path: Path) -> tuple[list[tuple[int, Any]], list[dict[str, Any]]]:
    records: list[tuple[int, Any]] = []
    rejected: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ProtocolError(f"cannot read {path}: {error}") from error
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line, parse_constant=_strict_constant)
        except (json.JSONDecodeError, ProtocolError) as error:
            rejected.append(
                {
                    "line": line_number,
                    "source": None,
                    "run": None,
                    "errors": [f"invalid JSONL: {error}"],
                    "raw": raw_line,
                }
            )
            continue
        records.append((line_number, value))
    return records, rejected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fraction(passes: int, tested: int) -> dict[str, Any]:
    return {
        "passes": passes,
        "tested": tested,
        "rate": passes / tested if tested else None,
        "display": f"{passes}/{tested}",
    }


def exact_metrics(passes: int, tested: int) -> dict[str, Any]:
    """Return exact-match counts and their equivalent TP/FP/FN metrics.

    Under this protocol an exact field match is one TP.  A mismatch is one FP
    and one FN.  Consequently precision, recall, F1, and exact-match rate are
    identical whenever at least one field is tested.
    """

    failures = tested - passes
    precision = passes / tested if tested else None
    recall = passes / tested if tested else None
    f1 = passes / tested if tested else None
    return {
        **fraction(passes, tested),
        "tp": passes,
        "fp": failures,
        "fn": failures,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exactMatchRate": passes / tested if tested else None,
    }


def _extract_targets(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise ProtocolError("targets document must be a JSON object")
    raw_targets = document.get("targets", document.get("rows"))
    if not isinstance(raw_targets, list):
        raise ProtocolError("targets document must contain targets[] or rows[]")

    targets: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for index, target in enumerate(raw_targets):
        where = f"targets[{index}]"
        if not isinstance(target, dict):
            raise ProtocolError(f"{where} must be an object")
        source = target.get("source")
        if not isinstance(source, str) or not source:
            raise ProtocolError(f"{where}.source must be a non-empty string")
        if source in seen_sources:
            raise ProtocolError(f"duplicate target source: {source}")
        seen_sources.add(source)
        baseline = target.get("baselineBindings")
        goal = target.get("targetBindings")
        if not isinstance(baseline, list) or not isinstance(goal, list):
            raise ProtocolError(
                f"{where} must contain baselineBindings[] and targetBindings[]"
            )
        raw_binding_count = target.get("bindingCount")
        binding_count = len(goal) if raw_binding_count is None else raw_binding_count
        if isinstance(binding_count, bool) or not isinstance(binding_count, int):
            raise ProtocolError(f"{where}.bindingCount must be an integer")
        if binding_count < 0:
            raise ProtocolError(f"{where}.bindingCount must not be negative")
        if len(baseline) != binding_count or len(goal) != binding_count:
            raise ProtocolError(
                f"{where} bindingCount={binding_count}, baseline={len(baseline)}, "
                f"target={len(goal)}"
            )
        for binding_index, binding in enumerate(baseline):
            binding_errors = _require_binding_fields(
                binding, f"{where}.baselineBindings[{binding_index}]"
            )
            if binding_errors:
                raise ProtocolError("; ".join(binding_errors))
        for binding_index, binding in enumerate(goal):
            binding_errors = _require_binding_fields(
                binding, f"{where}.targetBindings[{binding_index}]"
            )
            if binding_errors:
                raise ProtocolError("; ".join(binding_errors))
        normalized = dict(target)
        normalized["bindingCount"] = binding_count
        targets.append(normalized)
    return targets


def _require_binding_fields(binding: Any, where: str) -> list[str]:
    if not isinstance(binding, dict):
        return [f"{where} must be an object"]
    return [f"{where} missing field {field}" for field in FIELDS if field not in binding]


def _validate_prediction_binding(binding: Any, where: str) -> list[str]:
    errors = _require_binding_fields(binding, where)
    if errors or not isinstance(binding, dict):
        return errors

    if not non_empty_string_array(binding["ObjectName"]):
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
        errors.append(
            f"{where}.Trend must be a string or an allowed missing representation"
        )

    num = binding["Num"]
    if not is_absent(num) and not finite_numeric_array(num):
        errors.append(
            f"{where}.Num must be a finite JSON number array or an allowed "
            "missing representation"
        )
    if not isinstance(binding["Text"], str):
        errors.append(f"{where}.Text must be a string")
    return errors


def _identity(binding: Mapping[str, Any]) -> tuple[Any, Any]:
    return canonical(binding.get("DataName")), canonical(binding.get("Position"))


def _detect_reordered_bindings(
    predicted: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> bool:
    """Detect a pure permutation without turning ordinary field errors into gates."""

    goals = target["targetBindings"]
    if len(predicted) < 2 or len(predicted) != len(goals):
        return False
    predicted_ids = [_identity(binding) for binding in predicted]
    goal_ids = [_identity(binding) for binding in goals]
    if predicted_ids == goal_ids:
        return False
    # Canonical identities are tuples and therefore sortable only through repr.
    return sorted(map(repr, predicted_ids)) == sorted(map(repr, goal_ids))


def _normalize_row_prediction(
    value: Any, source_hint: str | None = None, run_hint: int | None = None
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if isinstance(value, list):
        record: dict[str, Any] = {"result": value}
    elif isinstance(value, dict):
        record = dict(value)
    else:
        return None, ["prediction row must be an object or a binding array"]

    source = record.get("source", source_hint)
    if source_hint is not None and "source" in record and source != source_hint:
        errors.append(
            f"prediction source {source!r} does not match requested source "
            f"{source_hint!r}"
        )
    if not isinstance(source, str) or not source:
        errors.append("prediction source must be a non-empty string")
    record["source"] = source

    run = record.get("run", run_hint if run_hint is not None else 1)
    if isinstance(run, bool) or not isinstance(run, int) or run < 1:
        errors.append("prediction run must be a positive integer")
    record["run"] = run

    result = record.get("result")
    if not isinstance(result, list):
        errors.append("prediction result must be an array")
    return record, errors


def _gate_row(
    record: Mapping[str, Any],
    target_by_source: Mapping[str, Mapping[str, Any]],
    where: str,
) -> list[str]:
    errors: list[str] = []
    source = record.get("source")
    target = target_by_source.get(source) if isinstance(source, str) else None
    if target is None:
        return [f"{where} has unknown source {source!r}"]
    result = record.get("result")
    if not isinstance(result, list):
        return [f"{where}.result must be an array"]
    expected_count = target["bindingCount"]
    if len(result) != expected_count:
        errors.append(
            f"{where} binding count mismatch: expected {expected_count}, "
            f"received {len(result)}"
        )
        return errors
    for binding_index, binding in enumerate(result):
        errors.extend(
            _validate_prediction_binding(
                binding, f"{where}.result[{binding_index}]"
            )
        )
    if not errors and _detect_reordered_bindings(result, target):
        errors.append(f"{where} binding order does not match the target order")
    return errors


def _load_vocabulary(
    explicit_path: Path | None, targets_path: Path
) -> tuple[set[str], str | None]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    else:
        candidates.extend(
            [
                targets_path.parent / "trend_vocabulary.json",
                Path(__file__).resolve().parent / "trend_vocabulary.json",
                Path.cwd() / "trend_vocabulary.json",
            ]
        )
    vocab_path = next((path for path in candidates if path.is_file()), None)
    if vocab_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise ProtocolError(
            f"trend vocabulary file was not found; searched: {searched}"
        )
    raw = load_json(vocab_path)
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        values = raw.get("vocabulary", raw.get("trends", list(raw.values())))
    else:
        raise ProtocolError("trend vocabulary must be an array or object")
    if not isinstance(values, list):
        raise ProtocolError("trend vocabulary values must be an array")
    flattened: list[Any] = []
    for item in values:
        flattened.extend(item if isinstance(item, list) else [item])
    vocab = {
        item.strip().lower()
        for item in flattened
        if isinstance(item, str)
    }
    return vocab, str(vocab_path.resolve())


def _new_counts() -> dict[str, dict[str, int]]:
    return {field: {"passes": 0, "tested": 0} for field in FIELDS}


def _add_count(counts: dict[str, dict[str, int]], field: str, passed: bool) -> None:
    counts[field]["tested"] += 1
    if passed:
        counts[field]["passes"] += 1


def _field_metrics(counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    return {
        field: exact_metrics(counts[field]["passes"], counts[field]["tested"])
        for field in FIELDS
    }


def _all_field_metric(counts: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    passes = sum(counts[field]["passes"] for field in FIELDS)
    tested = sum(counts[field]["tested"] for field in FIELDS)
    return exact_metrics(passes, tested)


def _evaluate(
    targets: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    vocabulary: set[str],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    results_by_source: dict[str, list[dict[str, Any]]] = {
        target["source"]: [] for target in targets
    }
    for record in valid_records:
        results_by_source[record["source"]].append(record)
    for records in results_by_source.values():
        records.sort(key=lambda item: item["run"])

    field_totals = _new_counts()
    optimization_totals = {
        field: {"passes": 0, "tested": 0} for field in TARGET_FIELDS
    }
    rows: list[dict[str, Any]] = []
    object_structure_passes = 0
    object_structure_tested = 0
    num_type_passes = 0
    num_type_tested = 0
    present_trends = 0
    present_trends_in_vocab = 0

    for target in targets:
        source_records = results_by_source[target["source"]]
        row_counts = _new_counts()
        binding_rows: list[dict[str, Any]] = []
        for binding_index in range(target["bindingCount"]):
            baseline = target["baselineBindings"][binding_index]
            goal = target["targetBindings"][binding_index]
            field_counts = _new_counts()
            presence: dict[str, dict[str, Any]] = {}
            for field in PRESENCE_FIELDS:
                presence[field] = {
                    "expectedPresent": not is_absent(baseline[field]),
                    "trueNegativePasses": 0,
                    "trueNegativeTested": 0,
                    "presencePasses": 0,
                    "presenceTested": 0,
                    "valuePasses": 0,
                    "valueTested": 0,
                }
            run_details: list[dict[str, Any]] = []
            for record in source_records:
                generated = record["result"][binding_index]
                pass_flags: dict[str, bool] = {}
                field_details: dict[str, dict[str, Any]] = {}
                for field in FIELDS:
                    if field in PRESENCE_FIELDS:
                        expected_present = not is_absent(baseline[field])
                        generated_present = not is_absent(generated[field])
                        item = presence[field]
                        item["presenceTested"] += 1
                        if expected_present == generated_present:
                            item["presencePasses"] += 1
                        if expected_present:
                            item["valueTested"] += 1
                            if generated_present and same_fixed(
                                generated[field], goal[field]
                            ):
                                item["valuePasses"] += 1
                        else:
                            item["trueNegativeTested"] += 1
                            if not generated_present:
                                item["trueNegativePasses"] += 1
                        passed = (
                            generated_present
                            and same_fixed(generated[field], goal[field])
                            if expected_present
                            else not generated_present
                        )
                    else:
                        passed = same_fixed(generated[field], goal[field])

                    pass_flags[field] = passed
                    field_details[field] = {
                        "pass": passed,
                        "score": 1 if passed else 0,
                    }
                    _add_count(field_counts, field, passed)
                    _add_count(row_counts, field, passed)
                    _add_count(field_totals, field, passed)
                    if field in TARGET_FIELDS:
                        optimization_totals[field]["tested"] += 1
                        if passed:
                            optimization_totals[field]["passes"] += 1

                object_structure_tested += 1
                if non_empty_string_array(generated["ObjectName"]):
                    object_structure_passes += 1
                num_type_tested += 1
                if finite_numeric_array(generated["Num"]):
                    num_type_passes += 1
                if not is_absent(generated["Trend"]):
                    present_trends += 1
                    if (
                        isinstance(generated["Trend"], str)
                        and generated["Trend"].strip().lower() in vocabulary
                    ):
                        present_trends_in_vocab += 1
                run_details.append(
                    {
                        "run": record["run"],
                        "passes": pass_flags,
                        "fields": field_details,
                    }
                )

            binding_rows.append(
                {
                    "binding": binding_index,
                    "precision": _field_metrics(field_counts),
                    "allFields": _all_field_metric(field_counts),
                    "targetThresholdPass": {
                        field: (
                            field_counts[field]["tested"] == 6
                            and field_counts[field]["passes"] / 6 > 0.8
                        )
                        for field in TARGET_FIELDS
                    },
                    "presence": {
                        field: {
                            "expectedPresent": values["expectedPresent"],
                            "presence": fraction(
                                values["presencePasses"],
                                values["presenceTested"],
                            ),
                            "trueNegative": fraction(
                                values["trueNegativePasses"],
                                values["trueNegativeTested"],
                            ),
                            "valueExact": fraction(
                                values["valuePasses"], values["valueTested"]
                            ),
                            "combined": exact_metrics(
                                field_counts[field]["passes"],
                                field_counts[field]["tested"],
                            ),
                        }
                        for field, values in presence.items()
                    },
                    "runs": run_details,
                }
            )
        rows.append(
            {
                "row": target.get("excelRow"),
                "excelRow": target.get("excelRow"),
                "source": target["source"],
                "completedRuns": len(source_records),
                "precision": _field_metrics(row_counts),
                "allFields": _all_field_metric(row_counts),
                "bindings": binding_rows,
            }
        )

    all_field_precision = _field_metrics(field_totals)
    optimization_precision = {
        field: exact_metrics(
            optimization_totals[field]["passes"],
            optimization_totals[field]["tested"],
        )
        for field in TARGET_FIELDS
    }
    summary = {
        "completedResults": len(valid_records),
        "rejectedResults": len(rejected),
        "expectedResults": len(targets) * 6,
        "rows": len(targets),
        "bindings": sum(target["bindingCount"] for target in targets),
        "fixedComparisonNormalization": {
            "absentEquivalent": [
                "JSON null",
                "None (case-insensitive)",
                "[]",
                "empty or whitespace-only string",
                "arrays whose members are all absent",
            ],
            "strings": (
                "trim outer whitespace and compare case-insensitively; "
                "internal whitespace and punctuation remain significant"
            ),
            "numbers": (
                "finite JSON numbers compare by numeric value; numeric strings "
                "fail type requirements"
            ),
            "arrays": "length, order, and normalized members must match",
            "objects": "key order is ignored; keys and normalized values must match",
        },
        "allFieldPrecision": all_field_precision,
        "allFieldsCombined": _all_field_metric(field_totals),
        "optimizationFieldPrecision": optimization_precision,
        "optimizationFieldsAbove80Percent": {
            field: (
                optimization_totals[field]["tested"] > 0
                and optimization_totals[field]["passes"]
                / optimization_totals[field]["tested"]
                > 0.8
            )
            for field in TARGET_FIELDS
        },
        "objectNameNonEmptyStringArray": fraction(
            object_structure_passes, object_structure_tested
        ),
        "numFiniteNumericArray": fraction(num_type_passes, num_type_tested),
        "trendVocabularyAmongPresentTrends": fraction(
            present_trends_in_vocab, present_trends
        ),
        "trendVocabularyAtLeast90Percent": (
            present_trends > 0 and present_trends_in_vocab / present_trends >= 0.9
        ),
    }
    return {"summary": summary, "rows": rows, "rejected": rejected}


def _prepare_records(
    raw_records: Iterable[tuple[int, Any]],
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_by_source = {target["source"]: target for target in targets}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line, raw in raw_records:
        record, errors = _normalize_row_prediction(raw)
        source = record.get("source") if record else None
        run = record.get("run") if record else None
        if record is not None and not errors:
            errors.extend(_gate_row(record, target_by_source, f"line {line}"))
        key = (
            (source, run)
            if isinstance(source, str)
            and isinstance(run, int)
            and not isinstance(run, bool)
            else None
        )
        if key is not None:
            if key in seen:
                errors.append(
                    f"duplicate (source, run): ({source!r}, {run})"
                )
            else:
                seen.add(key)
        if errors:
            rejected.append(
                {
                    "line": line,
                    "source": source,
                    "run": run,
                    "errors": errors,
                }
            )
        else:
            assert record is not None
            accepted.append(record)
    return accepted, rejected


def _base_document(
    command: str,
    targets_path: Path,
    prediction_path: Path | None,
    vocabulary_path: str | None,
) -> dict[str, Any]:
    return {
        "generatedAt": utc_now(),
        "protocol": "narrative2-fixed-python-v1",
        "command": command,
        "targetsPath": str(targets_path.resolve()),
        "predictionPath": (
            str(prediction_path.resolve()) if prediction_path is not None else None
        ),
        "vocabularyPath": vocabulary_path,
    }


def command_compare_batch(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    targets_path = Path(args.targets)
    predictions_path = Path(args.predictions)
    targets = _extract_targets(load_json(targets_path))
    vocabulary, vocabulary_path = _load_vocabulary(
        Path(args.vocabulary) if args.vocabulary else None, targets_path
    )
    raw_records, parse_rejected = load_jsonl(predictions_path)
    valid, gate_rejected = _prepare_records(raw_records, targets)
    evaluated = _evaluate(
        targets, valid, vocabulary, parse_rejected + gate_rejected
    )
    document = {
        **_base_document(
            "compare-batch", targets_path, predictions_path, vocabulary_path
        ),
        "valid": len(parse_rejected) + len(gate_rejected) == 0,
        **evaluated,
    }
    return document, 0


def command_compare_row(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    targets_path = Path(args.targets)
    prediction_path = Path(args.prediction)
    targets = _extract_targets(load_json(targets_path))
    target_by_source = {target["source"]: target for target in targets}
    vocabulary, vocabulary_path = _load_vocabulary(
        Path(args.vocabulary) if args.vocabulary else None, targets_path
    )
    record, errors = _normalize_row_prediction(
        load_json(prediction_path), source_hint=args.source, run_hint=args.run
    )
    if record is not None and not errors:
        errors.extend(_gate_row(record, target_by_source, "prediction"))
    if errors:
        document = {
            **_base_document(
                "compare-row", targets_path, prediction_path, vocabulary_path
            ),
            "valid": False,
            "errors": errors,
        }
        return document, 2
    assert record is not None
    target = target_by_source[record["source"]]
    evaluated = _evaluate([target], [record], vocabulary, [])
    document = {
        **_base_document(
            "compare-row", targets_path, prediction_path, vocabulary_path
        ),
        "valid": True,
        **evaluated,
    }
    return document, 0


def command_compare_binding(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    targets_path = Path(args.targets)
    prediction_path = Path(args.prediction)
    targets = _extract_targets(load_json(targets_path))
    target_by_source = {target["source"]: target for target in targets}
    target = target_by_source.get(args.source)
    errors: list[str] = []
    if isinstance(args.run, bool) or not isinstance(args.run, int) or args.run < 1:
        errors.append("run must be a positive integer")
    if target is None:
        errors.append(f"unknown source {args.source!r}")
    elif args.binding_index < 0 or args.binding_index >= target["bindingCount"]:
        errors.append(
            f"binding-index {args.binding_index} is outside 0.."
            f"{target['bindingCount'] - 1}"
        )
    binding = load_json(prediction_path)
    errors.extend(_validate_prediction_binding(binding, "prediction"))
    vocabulary, vocabulary_path = _load_vocabulary(
        Path(args.vocabulary) if args.vocabulary else None, targets_path
    )
    if errors:
        document = {
            **_base_document(
                "compare-binding", targets_path, prediction_path, vocabulary_path
            ),
            "valid": False,
            "source": args.source,
            "binding": args.binding_index,
            "errors": errors,
        }
        return document, 2

    assert target is not None and isinstance(binding, dict)
    synthetic_target = dict(target)
    synthetic_target["bindingCount"] = 1
    synthetic_target["baselineBindings"] = [
        target["baselineBindings"][args.binding_index]
    ]
    synthetic_target["targetBindings"] = [
        target["targetBindings"][args.binding_index]
    ]
    record = {"source": args.source, "run": args.run, "result": [binding]}
    evaluated = _evaluate([synthetic_target], [record], vocabulary, [])
    binding_result = evaluated["rows"][0]["bindings"][0]
    binding_result["binding"] = args.binding_index
    document = {
        **_base_document(
            "compare-binding", targets_path, prediction_path, vocabulary_path
        ),
        "valid": True,
        "row": target.get("excelRow"),
        "excelRow": target.get("excelRow"),
        "source": args.source,
        "binding": args.binding_index,
        "run": args.run,
        "precision": binding_result["precision"],
        "allFields": binding_result["allFields"],
        "targetThresholdPass": binding_result["targetThresholdPass"],
        "presence": binding_result["presence"],
        "fields": binding_result["runs"][0]["fields"],
    }
    return document, 0


def command_self_test(_: argparse.Namespace) -> tuple[dict[str, Any], int]:
    checks: list[tuple[str, bool]] = [
        ("outer whitespace and case", same_fixed(" AbC ", "abc")),
        ("internal whitespace significant", not same_fixed("a  b", "a b")),
        ("punctuation significant", not same_fixed("a.", "a")),
        ("None missing", is_absent(" NoNe ")),
        ("null missing", is_absent(None)),
        ("empty array missing", is_absent([])),
        ("nested missing array", is_absent([None, " ", ["None"]])),
        ("literal null not missing", not is_absent("null")),
        ("literal undefined not missing", not is_absent("undefined")),
        ("integer equals double", same_fixed(12, 12.0)),
        (
            "JSON numbers use JavaScript double precision",
            same_fixed(9007199254740993, 9007199254740992),
        ),
        ("numeric string differs", not same_fixed("12", 12)),
        ("boolean differs from number", not same_fixed(True, 1)),
        ("object key order ignored", same_fixed({"b": 2, "a": 1}, {"a": 1, "b": 2})),
        ("array order significant", not same_fixed([1, 2], [2, 1])),
        ("finite numeric array", finite_numeric_array([1, 2.5])),
        ("boolean numeric array rejected", not finite_numeric_array([True])),
        ("numeric string array rejected", not finite_numeric_array(["12"])),
        ("overflowing numeric array rejected", not finite_numeric_array([10**400])),
        ("object-name structure", non_empty_string_array(["US CPI"])),
        ("missing object-name rejected", not non_empty_string_array(["None"])),
        ("six-run threshold five passes", 5 / 6 > 0.8),
        ("six-run threshold four fails", not 4 / 6 > 0.8),
    ]
    failures = [name for name, passed in checks if not passed]
    document = {
        "generatedAt": utc_now(),
        "protocol": "narrative2-fixed-python-v1",
        "command": "self-test",
        "valid": not failures,
        "checks": [
            {"name": name, "pass": passed, "score": 1 if passed else 0}
            for name, passed in checks
        ],
        "summary": fraction(len(checks) - len(failures), len(checks)),
        "failures": failures,
    }
    return document, 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FinFlier narrative bindings with a fixed protocol."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--targets", required=True, help="Gold targets JSON")
        subparser.add_argument(
            "--vocabulary",
            help=(
                "Trend vocabulary JSON; defaults to trend_vocabulary.json beside "
                "the targets or evaluator"
            ),
        )
        subparser.add_argument("--output", required=True, help="Output JSON path")

    binding = subparsers.add_parser(
        "compare-binding", help="Compare one binding"
    )
    common(binding)
    binding.add_argument("--source", required=True)
    binding.add_argument("--binding-index", required=True, type=int)
    binding.add_argument("--prediction", required=True, help="Binding JSON")
    binding.add_argument("--run", type=int, default=1)
    binding.set_defaults(handler=command_compare_binding)

    row = subparsers.add_parser("compare-row", help="Compare one row")
    common(row)
    row.add_argument("--source", required=True)
    row.add_argument("--prediction", required=True, help="Row JSON")
    row.add_argument("--run", type=int, default=1)
    row.set_defaults(handler=command_compare_row)

    batch = subparsers.add_parser("compare-batch", help="Compare a JSONL batch")
    common(batch)
    batch.add_argument("--predictions", required=True, help="Predictions JSONL")
    batch.set_defaults(handler=command_compare_batch)

    self_test = subparsers.add_parser(
        "self-test", help="Run evaluator protocol self-tests"
    )
    self_test.add_argument("--output", help="Optional output JSON path")
    self_test.set_defaults(handler=command_self_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document, return_code = args.handler(args)
    except ProtocolError as error:
        document = {
            "generatedAt": utc_now(),
            "protocol": "narrative2-fixed-python-v1",
            "command": args.command,
            "valid": False,
            "fatalError": str(error),
        }
        return_code = 2
    output = getattr(args, "output", None)
    if output:
        write_json(Path(output), document)
    print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
