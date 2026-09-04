"""Shared Experiment 6 data-binding extraction helpers."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

NONE_TOKENS = {"", "none", "null", "nan", "n/a", "na", "[]", "[ ]"}
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
TEXT_FIELD_RE = re.compile(
    r"(?is)(?:^|[\n\r]|(?:^|\s)[-*]\s+|(?:^|\s)\d+[.)]\s*)"
    r"(ObjectName|Trend|Num|Numerical)\s*[:=]\s*"
    r"(.+?)(?=(?:\s+-\s*(?:ObjectName|DataName|Position|Trend|Num|Numerical|Text)\s*[:=])|[\n\r]|$)"
)
TYPE_ALIASES = {
    "subject": "subject",
    "subjects": "subject",
    "object": "subject",
    "objectname": "subject",
    "trend": "trend",
    "trends": "trend",
    "numerical": "numerical",
    "numeric": "numerical",
    "number": "numerical",
    "num": "numerical",
}

GENERIC_VALUE_HINTS = (
    "the name of",
    "this field",
    "in this case",
    "could be",
    "would be",
    "not explicitly",
    "not applicable",
    "no specific",
    "left empty",
    "valid string",
    "a list of",
    "the object being",
    "the data point",
    "the numerical value",
    "information in the text",
    "not possible to determine",
    "the continents",
    "the countries",
    "the vaccines",
    "specific data",
    "being referred to",
    "being described",
    "it is not",
    "it could",
)


@dataclass(frozen=True)
class BindingItem:
    case_id: str
    kind: str
    text: str


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_kind(value: Any) -> str | None:
    key = re.sub(r"[^a-z]", "", str(value or "").lower())
    return TYPE_ALIASES.get(key)


def case_id_for(record: dict[str, Any], fallback: str) -> str:
    for key in ("case_id", "CaseId", "id", "uid", "question_id", "Source"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def none_token(value: Any) -> str:
    text = normalize_text(value)
    text = text.strip(" \t\r\n\"'`")
    if text in {"[none]", "['none']", '["none"]'}:
        return "none"
    return text


def is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(is_none_like(item) for item in value)
    if isinstance(value, dict):
        return False
    return none_token(value) in NONE_TOKENS


def strip_think_and_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return re.sub(r"<think>.*?</think>", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()


def coerce_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = strip_think_and_fences(value)
    if not text:
        raise ValueError("empty JSON-like value")
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    first_object = text.find("{")
    last_object = text.rfind("}")
    first_array = text.find("[")
    last_array = text.rfind("]")
    candidates = []
    if first_object >= 0 and last_object > first_object:
        candidates.append(text[first_object : last_object + 1])
    if first_array >= 0 and last_array > first_array:
        candidates.append(text[first_array : last_array + 1])
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(candidate)
            except Exception:
                pass
    raise ValueError("value does not contain parseable JSON")


def scalar_values(value: Any) -> list[Any]:
    if is_none_like(value):
        return []
    if isinstance(value, (list, tuple, set)):
        values: list[Any] = []
        for item in value:
            values.extend(scalar_values(item))
        return values
    if isinstance(value, dict):
        for key in ("text", "value", "name", "label"):
            if key in value:
                return scalar_values(value[key])
        return []
    return [value]


def looks_like_generic_field_explanation(value: Any) -> bool:
    text = normalize_text(value)
    if not text:
        return True
    if any(hint in text for hint in GENERIC_VALUE_HINTS):
        return True
    if len(text.split()) > 18 and not NUMBER_RE.search(text):
        return True
    return False


def values_from_text_field(raw_value: str) -> list[str]:
    text = raw_value.strip().strip("`")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(" .;")
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            loaded = loader(text)
        except Exception:
            continue
        return [str(value) for value in scalar_values(loaded)]
    bracket = re.search(r"\[([^\]]+)\]", text)
    if bracket:
        inner = bracket.group(1)
        parts = [part.strip(" \t\"'`") for part in inner.split(",")]
        return [part for part in parts if part]
    quoted = re.findall(r'"([^"]+)"|`([^`]+)`', text)
    values = [left or right for left, right in quoted if left or right]
    if values:
        return values
    if "," in text and len(text) <= 160:
        return [part.strip(" \t\"'`") for part in text.split(",") if part.strip(" \t\"'`")]
    return [text.strip(" \t\"'`")]


def normalize_number(value: Any) -> tuple[float | None, str | None]:
    if is_none_like(value):
        return None, None
    match = NUMBER_RE.search(str(value))
    if not match:
        return None, None
    number = match.group(0).replace(",", "")
    if number.endswith("%"):
        number = number[:-1]
    try:
        numeric = float(number)
    except ValueError:
        return None, None
    text = f"{numeric:.12g}"
    return numeric, text


def normalize_subject_or_trend(value: Any) -> str | None:
    if is_none_like(value):
        return None
    text = normalize_text(value)
    return text if text and none_token(text) not in NONE_TOKENS else None


def item_dict(item: BindingItem) -> dict[str, str]:
    return {"type": item.kind, "text": item.text}


def canonical_items_from_items(raw_items: Any, case_id: str, row_number: int) -> tuple[list[BindingItem], list[dict[str, Any]], list[str]]:
    items: list[BindingItem] = []
    records: list[dict[str, Any]] = []
    none_fields: list[str] = []
    if not isinstance(raw_items, list):
        return items, records, ["items"]
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            none_fields.append(f"items[{index}]")
            continue
        kind = normalize_kind(raw.get("type") or raw.get("kind") or raw.get("category") or raw.get("label"))
        value = raw.get("text") or raw.get("value") or raw.get("name")
        if kind == "numerical":
            numeric, text = normalize_number(value)
            if text is None:
                none_fields.append(f"items[{index}].text")
                continue
            items.append(BindingItem(case_id, kind, text))
            records.append(base_record(case_id, row_number, index, "items", "Num", None, None, numeric, []))
        elif kind in {"subject", "trend"}:
            text = normalize_subject_or_trend(value)
            if text is None:
                none_fields.append(f"items[{index}].text")
                continue
            items.append(BindingItem(case_id, kind, text))
            records.append(
                base_record(
                    case_id,
                    row_number,
                    index,
                    "items",
                    "ObjectName" if kind == "subject" else "Trend",
                    text if kind == "subject" else None,
                    text if kind == "trend" else None,
                    None,
                    [],
                )
            )
    return items, records, none_fields


def base_record(
    case_id: str,
    row_number: int,
    result_index: int | None,
    source_schema: str,
    field: str | None,
    object_name: str | None,
    trend: str | None,
    num: float | None,
    none_fields: list[str],
    status: str = "extracted",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_row": row_number,
        "result_index": result_index,
        "source_schema": source_schema,
        "field": field,
        "object_name": object_name,
        "trend": trend,
        "num": num,
        "none_of_predict_fields": none_fields,
        "status": status,
        "error": error,
    }


def semi_structured_text_items(text: str, case_id: str, row_number: int) -> tuple[list[BindingItem], list[dict[str, Any]], list[str]]:
    metric_items: list[BindingItem] = []
    extraction_records: list[dict[str, Any]] = []
    none_fields: list[str] = []
    seen: set[tuple[str, str]] = set()

    for index, match in enumerate(TEXT_FIELD_RE.finditer(text)):
        raw_field, raw_value = match.group(1), match.group(2)
        kind = normalize_kind(raw_field)
        if kind is None:
            continue
        if looks_like_generic_field_explanation(raw_value):
            none_fields.append(f"semi_structured[{index}].{raw_field}")
            continue
        values = values_from_text_field(raw_value)
        if not values:
            none_fields.append(f"semi_structured[{index}].{raw_field}")
            continue
        field_had_value = False
        for value in values:
            if looks_like_generic_field_explanation(value):
                none_fields.append(f"semi_structured[{index}].{raw_field}")
                continue
            if kind == "numerical":
                numeric, normalized = normalize_number(value)
                if normalized is None:
                    none_fields.append(f"semi_structured[{index}].{raw_field}")
                    continue
                item_key = (kind, normalized)
                if item_key in seen:
                    continue
                seen.add(item_key)
                field_had_value = True
                metric_items.append(BindingItem(case_id, kind, normalized))
                extraction_records.append(
                    base_record(case_id, row_number, index, "semi_structured_text", "Num", None, None, numeric, [])
                )
                continue
            normalized = normalize_subject_or_trend(value)
            if normalized is None:
                none_fields.append(f"semi_structured[{index}].{raw_field}")
                continue
            item_key = (kind, normalized)
            if item_key in seen:
                continue
            seen.add(item_key)
            field_had_value = True
            extraction_records.append(
                base_record(
                    case_id,
                    row_number,
                    index,
                    "semi_structured_text",
                    "ObjectName" if kind == "subject" else "Trend",
                    normalized if kind == "subject" else None,
                    normalized if kind == "trend" else None,
                    None,
                    [],
                )
            )
            metric_items.append(BindingItem(case_id, kind, normalized))
        if not field_had_value:
            none_fields.append(f"semi_structured[{index}].{raw_field}")

    return metric_items, extraction_records, none_fields


def result_payload_from_record(record: dict[str, Any]) -> tuple[Any, str]:
    for key in ("result", "Result"):
        if key in record:
            return coerce_jsonish(record[key]), key
    for key in ("Binding", "binding"):
        if key in record:
            return coerce_jsonish(record[key]), key
    for key in ("prediction", "output", "final"):
        if key in record:
            payload = coerce_jsonish(record[key])
            if isinstance(payload, dict):
                for nested in ("result", "Result", "Binding", "binding"):
                    if nested in payload:
                        return coerce_jsonish(payload[nested]), f"{key}.{nested}"
            return payload, key
    if any(key in record for key in ("ObjectName", "Trend", "Num", "Numerical")):
        return record, "top_level_binding"
    return [], "empty"


def ensure_result_list(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("result", "Result", "Binding", "binding"):
            if key in payload:
                return ensure_result_list(payload[key])
        return [payload]
    if isinstance(payload, list):
        return payload
    if is_none_like(payload):
        return []
    return [payload]


def extract_result_items(
    record: dict[str, Any],
    fallback_case_id: str,
    row_number: int,
    strict: bool,
) -> tuple[list[BindingItem], list[dict[str, Any]], dict[str, Any]]:
    case_id = case_id_for(record, fallback_case_id)
    warnings: list[str] = []
    errors: list[str] = []

    if isinstance(record.get("items"), list):
        items, records, none_fields = canonical_items_from_items(record.get("items"), case_id, row_number)
        if not items and not records:
            none_fields = none_fields or ["items"]
            records.append(base_record(case_id, row_number, None, "items", None, None, None, None, none_fields, "none_of_predict"))
        return items, records, {
            "case_id": case_id,
            "source_row": row_number,
            "source_schema": "items",
            "items": len(items),
            "extraction_records": len(records),
            "none_of_predict_fields": none_fields,
            "warnings": warnings,
            "errors": errors,
        }

    try:
        payload, source_schema = result_payload_from_record(record)
        result_items = ensure_result_list(payload)
    except Exception as exc:
        for key in ("prediction", "output", "final", "raw_prediction"):
            raw_text = record.get(key)
            if isinstance(raw_text, str) and raw_text.strip():
                items, records, none_fields = semi_structured_text_items(raw_text, case_id, row_number)
                if items or records:
                    warning = f"json_parse_failed_used_semi_structured_text: {exc}"
                    if not records and none_fields:
                        records.append(
                            base_record(
                                case_id,
                                row_number,
                                None,
                                "semi_structured_text",
                                None,
                                None,
                                None,
                                None,
                                none_fields,
                                "none_of_predict",
                                str(exc),
                            )
                        )
                    return items, records, {
                        "case_id": case_id,
                        "source_row": row_number,
                        "source_schema": "semi_structured_text",
                        "items": len(items),
                        "extraction_records": len(records),
                        "none_of_predict_fields": none_fields,
                        "warnings": [*warnings, warning],
                        "errors": [str(exc)],
                    }
        if strict:
            raise
        error = str(exc)
        return [], [base_record(case_id, row_number, None, "invalid_prediction_result", None, None, None, None, ["result"], "none_of_predict", error)], {
            "case_id": case_id,
            "source_row": row_number,
            "source_schema": "invalid_prediction_result",
            "items": 0,
            "extraction_records": 1,
            "none_of_predict_fields": ["result"],
            "warnings": warnings,
            "errors": [error],
        }

    metric_items: list[BindingItem] = []
    extraction_records: list[dict[str, Any]] = []
    none_fields: list[str] = []
    if not result_items:
        none_fields.append("result")
        extraction_records.append(base_record(case_id, row_number, None, source_schema, None, None, None, None, ["result"], "none_of_predict"))
    for index, raw in enumerate(result_items):
        if not isinstance(raw, dict):
            field_name = f"result[{index}]"
            none_fields.append(field_name)
            extraction_records.append(base_record(case_id, row_number, index, source_schema, None, None, None, None, [field_name], "none_of_predict"))
            continue

        field_had_value = False
        object_values = scalar_values(raw.get("ObjectName") if "ObjectName" in raw else raw.get("object_name"))
        if not object_values:
            none_fields.append(f"result[{index}].ObjectName")
        for value in object_values:
            text = normalize_subject_or_trend(value)
            if text is None:
                none_fields.append(f"result[{index}].ObjectName")
                continue
            field_had_value = True
            metric_items.append(BindingItem(case_id, "subject", text))
            extraction_records.append(base_record(case_id, row_number, index, source_schema, "ObjectName", text, None, None, []))

        trend_values = scalar_values(raw.get("Trend") if "Trend" in raw else raw.get("trend"))
        if not trend_values:
            none_fields.append(f"result[{index}].Trend")
        for value in trend_values:
            text = normalize_subject_or_trend(value)
            if text is None:
                none_fields.append(f"result[{index}].Trend")
                continue
            field_had_value = True
            metric_items.append(BindingItem(case_id, "trend", text))
            extraction_records.append(base_record(case_id, row_number, index, source_schema, "Trend", None, text, None, []))

        num_values = scalar_values(raw.get("Num") if "Num" in raw else raw.get("Numerical"))
        if not num_values:
            none_fields.append(f"result[{index}].Num")
        for value in num_values:
            numeric, text = normalize_number(value)
            if text is None:
                none_fields.append(f"result[{index}].Num")
                continue
            field_had_value = True
            metric_items.append(BindingItem(case_id, "numerical", text))
            extraction_records.append(base_record(case_id, row_number, index, source_schema, "Num", None, None, numeric, []))

        if not field_had_value:
            extraction_records.append(base_record(case_id, row_number, index, source_schema, None, None, None, None, none_fields, "none_of_predict"))

    return metric_items, extraction_records, {
        "case_id": case_id,
        "source_row": row_number,
        "source_schema": source_schema,
        "items": len(metric_items),
        "extraction_records": len(extraction_records),
        "none_of_predict_fields": none_fields,
        "warnings": warnings,
        "errors": errors,
    }


def items_to_jsonl_row(case_id: str, items: Iterable[BindingItem]) -> dict[str, Any]:
    return {"case_id": case_id, "items": [item_dict(item) for item in items if item.case_id == case_id]}



def stringify(value: Any) -> str:
    return "" if value is None else str(value)


def load_target_csv_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    from build_few10_generator_input import normalize_question

    csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit >= 0:
        rows = rows[:limit]

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw_source_row = stringify(row.get("")).strip()
        source_csv_row: int | str = index
        if raw_source_row:
            source_csv_row = int(raw_source_row) if raw_source_row.isdigit() else raw_source_row
        question = stringify(row.get("Question")).strip()
        if not question and stringify(row.get("Narrative_Text")).strip():
            question = "What data-text bindings are described in the narrative?"
        text = stringify(row.get("Sentence")).strip()
        if not text:
            text = " ".join(
                part for part in [stringify(row.get("Pre_Text")).strip(), stringify(row.get("Post_Text")).strip()] if part
            )
        output.append(
            {
                "id": stringify(row.get("id")).strip() or str(source_csv_row),
                "source_id": stringify(row.get("id")).strip() or str(source_csv_row),
                "selection_key": f"row:{source_csv_row}|question:{normalize_question(question)}",
                "normalized_question": normalize_question(question),
                "question": question,
                "answer": stringify(row.get("GT_Answer")).strip(),
                "program": stringify(row.get("GT_Program")).strip(),
                "text": text,
                "table": stringify(row.get("Tables")).strip(),
                "table_text": stringify(row.get("Table_Text")).strip(),
                "retrieved": [part.strip() for part in stringify(row.get("Rel_Fact")).split(";") if part.strip()],
                "binding_result": stringify(row.get("Binding_Result")).strip(),
                "binding_reason": stringify(row.get("Binding_Reason")).strip(),
                "prompt_mode": stringify(row.get("Prompt_Mode")).strip(),
                "source_csv_row": source_csv_row,
                "source_csv_position": index,
                "source": str(path),
                "retrieved_source": "experiment7_selection_cache_materialization",
                "flow_scope": "experiment7_selection_cache_materialization",
            }
        )
    return output


def binding_payload_from_row(row: dict[str, Any]) -> Any:
    for key in ("items", "result", "Result", "Binding", "binding"):
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    if stringify(row.get("binding_result")).strip():
        return row["binding_result"]
    if any(row.get(key) not in (None, "") for key in ("ObjectName", "Trend", "Num", "Numerical")):
        return row
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_selection_cache(args: argparse.Namespace) -> dict[str, Any]:
    from build_few10_generator_input import load_csv_examples, load_matched_examples
    from example_selection import (
        build_example_lookup,
        cache_items_by_key,
        load_selection_cache,
        prompt_type_train_csv,
        selected_from_cache,
    )

    if args.target_input_json is not None:
        target_rows = load_matched_examples(args.target_input_json, args.limit)
        target_source = args.target_input_json
        target_source_mode = "matched_retriever_artifact"
    else:
        target_rows = load_target_csv_rows(args.target_input_csv, args.limit)
        target_source = args.target_input_csv
        target_source_mode = "target_csv_dataset"

    train_csv = prompt_type_train_csv(args.target_prompt_type, args.prompt_type_train_csv)
    if train_csv is None:
        raise SystemExit("materialize-selection-cache requires --target-prompt-type or --prompt-type-train-csv")
    if not train_csv.is_file():
        raise SystemExit(f"prompt-type train CSV missing: {train_csv}")

    cache_payload = load_selection_cache(args.selection_cache_json)
    cache_map = cache_items_by_key(cache_payload)
    example_lookup = build_example_lookup(load_csv_examples(train_csv, -1, "finqa_train_formal"))

    materialized_rows: list[dict[str, Any]] = []
    extracted_rows: list[dict[str, Any]] = []
    unresolved_selected_examples: dict[str, list[str]] = {}
    cache_hit_rows = 0
    cache_missing_rows = 0
    extracted_materialized_rows = 0
    skipped_no_binding_payload_rows = 0
    cache_match_key_types: dict[str, int] = {}
    row_number_collision_rows = 0
    non_selection_key_cache_match_rows = 0

    for row_number, row in enumerate(target_rows, start=1):
        selected_examples, cache_item, missing_ids = selected_from_cache(
            row, cache_map, example_lookup, args.shot_number
        )
        materialized = dict(row)
        materialized["selected_examples"] = selected_examples
        materialized["selected_example_ids"] = list(cache_item.get("selected_example_ids") or []) if cache_item else []
        materialized["selected_example_source_rows"] = (
            list(cache_item.get("selected_example_source_rows") or []) if cache_item else []
        )
        materialized["target_prompt_type"] = args.target_prompt_type
        materialized["selection_cache_json"] = str(args.selection_cache_json)
        materialized["prompt_type_train_csv"] = str(train_csv)
        cache_lookup_key = cache_item.get("_cache_lookup_key") if cache_item else None
        cache_lookup_key_type = cache_item.get("_cache_lookup_key_type") if cache_item else None
        cache_lookup_key_type_text = stringify(cache_lookup_key_type).strip()
        if cache_lookup_key_type_text:
            cache_match_key_types[cache_lookup_key_type_text] = cache_match_key_types.get(cache_lookup_key_type_text, 0) + 1
        if cache_lookup_key_type_text.startswith("legacy_"):
            row_number_collision_rows += 1
        selected_missing_ids = list(missing_ids)
        if cache_item and selected_examples and cache_lookup_key_type_text != "selection_key":
            non_selection_key_cache_match_rows += 1
            selected_missing_ids.append(f"non_selection_key_cache_match:{cache_lookup_key_type_text or 'missing'}")
        formal_finder_ready = bool(
            cache_item
            and not selected_missing_ids
            and selected_examples
            and cache_lookup_key_type_text == "selection_key"
        )
        materialized["example_selection"] = {
            "selection_key": materialized.get("selection_key"),
            "selection_status": cache_item.get("selection_status") if cache_item else "target_row_not_in_selection_cache",
            "selection_binding_status": "passed" if formal_finder_ready else "blocked_example_selection_cache",
            "formal_finder_ready": formal_finder_ready,
            "cache_key_found": cache_lookup_key is not None,
            "cache_key_found_value": cache_lookup_key,
            "cache_match_key_type": cache_lookup_key_type,
            "allow_legacy_selection_binding": False,
            "selected_rows": len(selected_examples),
            "missing_selected_ids": selected_missing_ids,
        }
        materialized_rows.append(materialized)

        row_key = stringify(materialized.get("selection_key")).strip() or stringify(materialized.get("id")).strip()
        if not formal_finder_ready:
            cache_missing_rows += 1
            if row_key:
                unresolved_selected_examples[row_key] = selected_missing_ids or ["target_row_not_in_selection_cache"]
        else:
            cache_hit_rows += 1

        payload = binding_payload_from_row(materialized)
        if payload is None:
            skipped_no_binding_payload_rows += 1
            continue
        record = dict(materialized)
        if all(key not in record for key in ("items", "result", "Result", "Binding", "binding")):
            record["result"] = payload
        case_id = stringify(record.get("id")).strip() or f"row_{row_number}"
        items, _, _ = extract_result_items(record, case_id, row_number, strict=args.strict)
        extracted_rows.append(items_to_jsonl_row(case_id, items))
        extracted_materialized_rows += 1

    write_jsonl(args.output_jsonl, materialized_rows)
    if args.extracted_jsonl is not None:
        write_jsonl(args.extracted_jsonl, extracted_rows)

    report = {
        "kind": "experiment7_selection_cache_materialization",
        "selection_cache_json": str(args.selection_cache_json),
        "target_source": str(target_source),
        "target_source_mode": target_source_mode,
        "target_prompt_type": args.target_prompt_type,
        "prompt_type_train_csv": str(train_csv),
        "rows": len(materialized_rows),
        "shot_number": args.shot_number,
        "cache_hit_rows": cache_hit_rows,
        "cache_missing_rows": cache_missing_rows,
        "extracted_materialized_rows": extracted_materialized_rows,
        "skipped_no_binding_payload_rows": skipped_no_binding_payload_rows,
        "cache_match_key_types": cache_match_key_types,
        "row_number_collision_rows": row_number_collision_rows,
        "non_selection_key_cache_match_rows": non_selection_key_cache_match_rows,
        "output_jsonl": str(args.output_jsonl),
        "extracted_jsonl": str(args.extracted_jsonl) if args.extracted_jsonl is not None else None,
        "unresolved_selected_examples": unresolved_selected_examples,
        "selection_summary": cache_payload.get("selection_summary"),
    }
    write_json(args.report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    materialize = subparsers.add_parser(
        "materialize-selection-cache",
        help="Backfill selected examples for a target dataset/prompt-type from the shared Experiment 7 selection cache.",
    )
    source = materialize.add_mutually_exclusive_group(required=True)
    source.add_argument("--target-input-json", type=Path)
    source.add_argument("--target-input-csv", type=Path)
    materialize.add_argument("--selection-cache-json", type=Path, required=True)
    materialize.add_argument(
        "--target-prompt-type",
        choices=["raw", "original", "zero-shot", "many-shot", "dynamic-shot"],
        required=True,
    )
    materialize.add_argument("--prompt-type-train-csv", type=Path)
    materialize.add_argument("--output-jsonl", type=Path, required=True)
    materialize.add_argument("--report-json", type=Path, required=True)
    materialize.add_argument("--extracted-jsonl", type=Path)
    materialize.add_argument("--shot-number", type=int, default=4)
    materialize.add_argument("--limit", type=int, default=-1)
    materialize.add_argument("--strict", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command != "materialize-selection-cache":
        raise SystemExit("Specify subcommand: materialize-selection-cache")
    report = materialize_selection_cache(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
