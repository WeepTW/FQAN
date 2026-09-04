"""RetFact JSON-schema helpers for retriever training and matching."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from typing import Any


ORIGINAL_PROMPT_MODES = {"original", "orig"}
RAW_PROMPT_MODES = {"raw", "raw-finqa", "finqa-raw", "finqa-r"}
PLAIN_RETFACT_PROMPT_MODES = ORIGINAL_PROMPT_MODES | RAW_PROMPT_MODES
SCHEMA_PROMPT_MODES = {
    "zero-shot",
    "zero_shot",
    "many-shot",
    "many_shot",
    "dynamic-shot",
    "dynamic_shot",
    "new_prompt_zero_shot",
    "new_prompt_many_shot",
    "new_prompt_few_shot",
    "new_prompt_dynamic_shot",
}


@dataclass(frozen=True)
class RetFactSchemaResult:
    ret_fact: str
    valid: bool
    errors: tuple[str, ...]


SCHEMA_REQUIRED_FIELD_NAMES = (
    "RetFact",
    "Binding",
    "ObjectName",
    "DataName",
    "Position",
    "Begin",
    "End",
    "Trend",
    "Num",
    "Text",
    "Reason",
)
SCHEMA_CRITICAL_PUNCTUATION = frozenset("{}[]:,")
RETRIEVER_RETFACT_GENERATION_MAX_CHARS = int(
    os.environ.get("RETRIEVER_RETFACT_GENERATION_MAX_CHARS", "4096")
)
FORMAT_BACKENDS = {"auto", "assembler", "canonical", "jsonformer", "lmfe", "model", "off", "wrapper"}
CANONICAL_RETRIEVER_SCHEMA_VERSION = "new_prompt_retfact_binding_reason_v1"
CANONICAL_TOP_LEVEL_KEYS = ("RetFact", "Binding", "Reason")
CANONICAL_BINDING_KEYS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
CANONICAL_POSITION_KEYS = ("Begin", "End")
CANONICAL_DEFAULT_BINDING: dict[str, Any] = {
    "ObjectName": [],
    "DataName": "",
    "Position": [{"Begin": [], "End": []}],
    "Trend": "None",
    "Num": [],
    "Text": "",
}
RETRIEVER_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "RetFact": {
            "type": "string",
            "minLength": 1,
            "maxLength": RETRIEVER_RETFACT_GENERATION_MAX_CHARS,
        },
        "Binding": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "ObjectName": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                    "DataName": {"type": "string"},
                    "Position": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "Begin": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "maxItems": 10,
                                },
                                "End": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "maxItems": 10,
                                },
                            },
                            "required": ["Begin", "End"],
                            "additionalProperties": False,
                        },
                    },
                    "Trend": {"type": "string"},
                    "Num": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                        "maxItems": 10,
                    },
                    "Text": {"type": "string"},
                },
                "required": ["ObjectName", "DataName", "Position", "Trend", "Num", "Text"],
                "additionalProperties": False,
            },
        },
        "Reason": {"type": "string"},
    },
    "required": ["RetFact", "Binding", "Reason"],
    "additionalProperties": False,
}


def prompt_mode_from_path(path: str | Path) -> str:
    normalized = str(path).lower().replace("\\", "/").replace("_", "-")
    if "finqa-raw" in normalized or "/data/src/finder/" in normalized:
        return "raw"
    if "zero-shot" in normalized:
        return "zero-shot"
    if "many-shot" in normalized or "few-shot" in normalized:
        return "many-shot"
    if "dynamic-shot" in normalized:
        return "dynamic-shot"
    return "original"


def normalize_prompt_mode(prompt_mode: str | None, csv_path: str | Path | None = None) -> str:
    value = (prompt_mode or "auto").strip()
    if not value or value == "auto":
        return prompt_mode_from_path(csv_path or "")
    key = value.lower().replace("_", "-")
    if key in ORIGINAL_PROMPT_MODES:
        return "original"
    if key in RAW_PROMPT_MODES:
        return "raw"
    if key in {"zero-shot", "new-prompt-zero-shot"}:
        return "zero-shot"
    if key in {"many-shot", "few-shot", "new-prompt-many-shot", "new-prompt-few-shot"}:
        return "many-shot"
    if key in {"dynamic-shot", "new-prompt-dynamic-shot"}:
        return "dynamic-shot"
    raise ValueError(f"Unsupported prompt mode for retriever schema handling: {prompt_mode}")


def schema_required(prompt_mode: str | None, csv_path: str | Path | None = None) -> bool:
    return normalize_prompt_mode(prompt_mode, csv_path) not in {"original", "raw"}


def default_binding() -> dict[str, Any]:
    return deepcopy(CANONICAL_DEFAULT_BINDING)


def build_retfact_schema(ret_fact: str) -> str:
    payload = canonical_retfact_payload(ret_fact)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def canonical_retfact_payload(ret_fact: str) -> dict[str, Any]:
    return {
        "RetFact": normalize_retfact_text(ret_fact),
        "Binding": [default_binding()],
        "Reason": "",
    }


def assembler_metadata() -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_RETRIEVER_SCHEMA_VERSION,
        "top_level_keys": list(CANONICAL_TOP_LEVEL_KEYS),
        "binding_keys": list(CANONICAL_BINDING_KEYS),
        "position_keys": list(CANONICAL_POSITION_KEYS),
        "binding_policy": "single_default_binding_object",
        "reason_policy": "empty_string",
        "retfact_policy": "extract_or_normalize_model_text",
    }


def schema_parse_gate(text: str) -> bool:
    result = parse_retfact_schema(text)
    return result.valid and bool(result.ret_fact.strip())


def _mark_span(mask: list[bool], start: int, end: int) -> None:
    start = max(0, start)
    end = min(len(mask), end)
    for index in range(start, end):
        mask[index] = True


def _iter_literal_spans(text: str, literal: str):
    start = 0
    while True:
        index = text.find(literal, start)
        if index < 0:
            break
        yield index, index + len(literal)
        start = index + max(1, len(literal))


def schema_and_retfact_char_masks(target_text: str) -> tuple[list[bool], list[bool]]:
    text = str(target_text)
    schema_mask = [False] * len(text)
    retfact_mask = [False] * len(text)

    for index, char in enumerate(text):
        if char in SCHEMA_CRITICAL_PUNCTUATION:
            schema_mask[index] = True

    for field_name in SCHEMA_REQUIRED_FIELD_NAMES:
        literal = json.dumps(field_name, ensure_ascii=False, separators=(",", ":"))
        for start, end in _iter_literal_spans(text, literal):
            _mark_span(schema_mask, start, end)

    payload = extract_json_object(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("RetFact"), str):
        return schema_mask, retfact_mask

    key_literal = json.dumps("RetFact", ensure_ascii=False, separators=(",", ":"))
    key_index = text.find(key_literal)
    if key_index < 0:
        return schema_mask, retfact_mask
    colon_index = text.find(":", key_index + len(key_literal))
    if colon_index < 0:
        return schema_mask, retfact_mask

    value_literal = json.dumps(payload["RetFact"], ensure_ascii=False, separators=(",", ":"))
    value_index = text.find(value_literal, colon_index + 1)
    if value_index < 0:
        return schema_mask, retfact_mask
    if len(value_literal) >= 2 and value_literal[0] == value_literal[-1] == '"':
        _mark_span(retfact_mask, value_index + 1, value_index + len(value_literal) - 1)
    else:
        _mark_span(retfact_mask, value_index, value_index + len(value_literal))
    return schema_mask, retfact_mask


def _token_overlaps(mask: list[bool], start: int, end: int) -> bool:
    if end <= start:
        return False
    return any(mask[start:end])


def schema_and_retfact_token_masks(tokenizer, target_text: str) -> tuple[list[bool], list[bool]]:
    schema_chars, retfact_chars = schema_and_retfact_char_masks(target_text)
    try:
        encoded = tokenizer(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
    except (NotImplementedError, TypeError, KeyError):
        encoded = tokenizer(target_text, add_special_tokens=False)
        token_count = len(encoded["input_ids"])
        return [False] * token_count, [False] * token_count

    schema_mask = []
    retfact_mask = []
    for start, end in offsets:
        schema_mask.append(_token_overlaps(schema_chars, start, end))
        retfact_mask.append(_token_overlaps(retfact_chars, start, end))
    return schema_mask, retfact_mask


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    for index in range(max(0, limit)):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def label_for_prompt_mode(ret_fact: str, prompt_mode: str | None, csv_path: str | Path | None = None) -> str:
    if schema_required(prompt_mode, csv_path):
        return build_retfact_schema(ret_fact)
    return str(ret_fact)


def retfact_label_for_training(ret_fact: str, prompt_mode: str | None = None, csv_path: str | Path | None = None) -> str:
    """Return the retriever training target.

    Original prompt mode keeps FINDER's plain RetFact target.  Non-original
    prompt modes train against the JSON schema target; their trainer-side loss
    masks decide which target tokens contribute to optimization.
    """

    return label_for_prompt_mode(ret_fact, prompt_mode, csv_path)


def normalize_retfact_text(text: str) -> str:
    return " ".join(str(text).split())


def _strip_code_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|csv|text)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = _strip_code_fence(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            value, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_retfact_schema(payload: Any) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, ("payload is not a JSON object",)
    allowed_top_keys = {"RetFact", "Binding", "Reason"}
    missing_top_keys = allowed_top_keys - set(payload)
    extra_top_keys = set(payload) - allowed_top_keys
    if missing_top_keys:
        errors.append(f"missing required top-level keys: {sorted(missing_top_keys)}")
    if extra_top_keys:
        errors.append(f"unexpected top-level keys: {sorted(extra_top_keys)}")
    if not isinstance(payload.get("RetFact"), str):
        errors.append("RetFact must be a string")
    elif not payload["RetFact"].strip():
        errors.append("RetFact must be a non-empty string")
    if not isinstance(payload.get("Reason"), str):
        errors.append("Reason must be a string")
    binding = payload.get("Binding")
    if not isinstance(binding, list):
        errors.append("Binding must be a list")
    elif len(binding) > 10:
        errors.append("Binding must contain at most 10 objects")
    else:
        for index, item in enumerate(binding):
            if not isinstance(item, dict):
                errors.append(f"Binding[{index}] must be an object")
                continue
            allowed_binding_keys = {"ObjectName", "DataName", "Position", "Trend", "Num", "Text"}
            missing_binding_keys = allowed_binding_keys - set(item)
            extra_binding_keys = set(item) - allowed_binding_keys
            if missing_binding_keys:
                errors.append(f"Binding[{index}] missing keys: {sorted(missing_binding_keys)}")
            if extra_binding_keys:
                errors.append(f"Binding[{index}] unexpected keys: {sorted(extra_binding_keys)}")
            if not isinstance(item.get("ObjectName"), list):
                errors.append(f"Binding[{index}].ObjectName must be a list")
            elif len(item["ObjectName"]) > 10:
                errors.append(f"Binding[{index}].ObjectName must contain at most 10 items")
            elif not all(isinstance(value, str) for value in item["ObjectName"]):
                errors.append(f"Binding[{index}].ObjectName items must be strings")
            if not isinstance(item.get("DataName"), str):
                errors.append(f"Binding[{index}].DataName must be a string")
            if not isinstance(item.get("Position"), list):
                errors.append(f"Binding[{index}].Position must be a list")
            elif len(item["Position"]) > 10:
                errors.append(f"Binding[{index}].Position must contain at most 10 objects")
            else:
                for pos_index, position in enumerate(item["Position"]):
                    if not isinstance(position, dict):
                        errors.append(f"Binding[{index}].Position[{pos_index}] must be an object")
                        continue
                    allowed_position_keys = {"Begin", "End"}
                    missing_position_keys = allowed_position_keys - set(position)
                    extra_position_keys = set(position) - allowed_position_keys
                    if missing_position_keys:
                        errors.append(
                            f"Binding[{index}].Position[{pos_index}] missing keys: "
                            f"{sorted(missing_position_keys)}"
                        )
                    if extra_position_keys:
                        errors.append(
                            f"Binding[{index}].Position[{pos_index}] unexpected keys: "
                            f"{sorted(extra_position_keys)}"
                        )
                    if not isinstance(position.get("Begin"), list):
                        errors.append(f"Binding[{index}].Position[{pos_index}].Begin must be a list")
                    elif len(position["Begin"]) > 10:
                        errors.append(
                            f"Binding[{index}].Position[{pos_index}].Begin must contain at most 10 items"
                        )
                    elif not all(isinstance(value, (int, float)) for value in position["Begin"]):
                        errors.append(
                            f"Binding[{index}].Position[{pos_index}].Begin items must be numbers"
                        )
                    if not isinstance(position.get("End"), list):
                        errors.append(f"Binding[{index}].Position[{pos_index}].End must be a list")
                    elif len(position["End"]) > 10:
                        errors.append(
                            f"Binding[{index}].Position[{pos_index}].End must contain at most 10 items"
                        )
                    elif not all(isinstance(value, (int, float)) for value in position["End"]):
                        errors.append(
                            f"Binding[{index}].Position[{pos_index}].End items must be numbers"
                        )
            if not isinstance(item.get("Trend"), str):
                errors.append(f"Binding[{index}].Trend must be a string")
            if not isinstance(item.get("Num"), list):
                errors.append(f"Binding[{index}].Num must be a list")
            elif len(item["Num"]) > 10:
                errors.append(f"Binding[{index}].Num must contain at most 10 items")
            elif not all(isinstance(value, (int, float, str)) for value in item["Num"]):
                errors.append(f"Binding[{index}].Num items must be numbers or strings")
            if not isinstance(item.get("Text"), str):
                errors.append(f"Binding[{index}].Text must be a string")
    return not errors, tuple(errors)


def parse_retfact_schema(text: str) -> RetFactSchemaResult:
    payload = extract_json_object(text)
    if payload is None:
        return RetFactSchemaResult(ret_fact="", valid=False, errors=("no JSON object found",))
    valid, errors = validate_retfact_schema(payload)
    ret_fact = payload.get("RetFact") if isinstance(payload.get("RetFact"), str) else ""
    return RetFactSchemaResult(ret_fact=ret_fact, valid=valid, errors=errors)


def _decode_json_string_fragment(value: str) -> str:
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(value)
    except json.JSONDecodeError:
        return ""
    return parsed if isinstance(parsed, str) else ""


def _retfact_value_fragment(text: str) -> str:
    match = re.search(r'"RetFact"\s*:\s*"', str(text))
    if not match:
        return ""
    start = match.end() - 1
    decoded = _decode_json_string_fragment(str(text)[start:])
    if decoded:
        return decoded

    remainder = str(text)[match.end() :]
    stop_match = re.search(
        r'"\s*,\s*"(?:Binding|Reason|ObjectName|DataName|Position|Begin|End|Trend|Num|Text)"\s*:',
        remainder,
    )
    if stop_match:
        remainder = remainder[: stop_match.start()]
    return remainder.strip().strip('"').strip()


def retfact_text_from_model_output(text: str) -> str:
    """Extract the RetFact string from raw model output.

    This accepts three cases: a valid schema object, a partially generated
    RetFact JSON prefix, or a plain RetFact string from the RetFact-only route.
    """

    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return normalize_retfact_text(result.ret_fact)
    fragment = _retfact_value_fragment(text)
    if fragment.strip():
        return normalize_retfact_text(fragment)
    return normalize_retfact_text(text)


def prediction_for_prompt_mode(text: str, prompt_mode: str | None, csv_path: str | Path | None = None) -> str:
    ret_fact = retfact_text_from_model_output(text)
    if schema_required(prompt_mode, csv_path):
        return build_retfact_schema(ret_fact)
    return ret_fact


def schema_object_like_output(text: str) -> bool:
    normalized = str(text).lstrip()
    if normalized.startswith("{"):
        return True
    return bool(re.search(r'[\"\'](?:RetFact|Binding|Reason)[\"\']\s*:', normalized))


def retfact_text_for_schema_assembly(text: str) -> str:
    """Return RetFact text only when schema assembly is defensible.

    Plain RetFact text is valid input for assembly. JSON-like schema attempts
    without a parseable RetFact are left invalid so the match gate can surface
    the failure.
    """

    fragment = _retfact_value_fragment(text)
    if fragment.strip():
        return normalize_retfact_text(fragment)
    if schema_object_like_output(text):
        return ""
    return normalize_retfact_text(text)


def canonical_schema_prediction(text: str) -> str:
    payload = extract_json_object(text)
    valid, _ = validate_retfact_schema(payload)
    if not valid:
        ret_fact = retfact_text_for_schema_assembly(text)
        if ret_fact.strip():
            return build_retfact_schema(ret_fact)
        return normalize_retfact_text(text)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def assembler_schema_prediction(text: str) -> str:
    ret_fact = retfact_text_from_model_output(text)
    if ret_fact.strip():
        return build_retfact_schema(ret_fact)
    return normalize_retfact_text(text)


def model_schema_prediction(text: str) -> str:
    payload = extract_json_object(text)
    valid, _ = validate_retfact_schema(payload)
    if valid:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return normalize_retfact_text(text)


def structured_prediction_for_prompt_mode(
    text: str,
    prompt_mode: str | None,
    csv_path: str | Path | None = None,
) -> str:
    if schema_required(prompt_mode, csv_path):
        return canonical_schema_prediction(text)
    return retfact_text_from_model_output(text)


def resolve_format_backend(
    requested: str | None,
    prompt_mode: str | None,
    csv_path: str | Path | None = None,
) -> str:
    if not schema_required(prompt_mode, csv_path):
        return "off"
    value = (requested or "auto").strip().lower().replace("_", "-")
    if value == "auto":
        return "assembler"
    if value == "wrapper":
        return "assembler"
    if value not in FORMAT_BACKENDS:
        raise ValueError(
            "--structured-output/format_backend must be one of: "
            "auto, assembler, canonical, jsonformer, lmfe, model, off, wrapper"
        )
    return value


def format_backend_prediction(
    text: str,
    prompt_mode: str | None,
    backend: str,
    csv_path: str | Path | None = None,
) -> str:
    resolved = resolve_format_backend(backend, prompt_mode, csv_path)
    if resolved == "off":
        return str(text) if schema_required(prompt_mode, csv_path) else retfact_text_from_model_output(text)
    if resolved == "assembler":
        return assembler_schema_prediction(text) if schema_required(prompt_mode, csv_path) else retfact_text_from_model_output(text)
    if resolved == "canonical":
        return canonical_schema_prediction(text) if schema_required(prompt_mode, csv_path) else retfact_text_from_model_output(text)
    if resolved == "model":
        return model_schema_prediction(text) if schema_required(prompt_mode, csv_path) else retfact_text_from_model_output(text)
    return structured_prediction_for_prompt_mode(text, prompt_mode, csv_path)


def schema_like_output(text: str) -> bool:
    normalized = str(text)
    return any(token in normalized for token in ("RetFact", "Binding", "ObjectName", "Reason"))


def retfact_text_for_matching(text: str) -> str:
    result = parse_retfact_schema(text)
    if result.valid and result.ret_fact.strip():
        return result.ret_fact
    if schema_like_output(text):
        return ""
    return str(text)


def schema_invalid(text: str) -> bool:
    return not parse_retfact_schema(text).valid
