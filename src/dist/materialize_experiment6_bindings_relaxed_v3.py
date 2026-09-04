#!/usr/bin/env python3
"""Losslessly normalize Experiment 6 Binding candidates for diagnosis.

This materializer consumes the frozen unified-v2 candidate root.  It never
reads gold or judge data.  Missing Binding fields are represented by empty
values, recoverable container/type errors are normalized, and Num values are
kept verbatim (including strings with units).  Raw model output is recovered
only from an explicit answer region or from a narrowly recognized malformed
FLAN Binding shape.  Prompt echoes, empty responses, and degenerate token
streams remain unavailable and are retained in the row-level audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from materialize_experiment6_binding_candidates import (
    candidate_id,
    read_json,
    read_jsonl,
    sha256_file,
    stable_sha256,
    utc_now,
    write_json,
    write_jsonl,
)
from validate_experiment6_bindings_v2 import validate_output as validate_v2


PROTOCOL = "experiment6-binding-materialization-relaxed-v3-unified34"
SOURCE_PROTOCOL = "experiment6-binding-materialization-v2-unified34"
POLICY = "lossless-six-field-shape-normalization-v1"
BINDING_KEYS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
UNAVAILABLE_STATUSES = {
    "unavailable_empty",
    "unavailable_degenerate_tokens",
    "unavailable_prompt_echo",
    "unavailable_no_explicit_output_region",
    "unavailable_no_binding_structure",
}


class RelaxedMaterializationError(RuntimeError):
    """Raised when a source or output invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RelaxedMaterializationError(message)


def json_compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def blank_binding() -> dict[str, Any]:
    return {
        "ObjectName": [],
        "DataName": "",
        "Position": [],
        "Trend": "",
        "Num": [],
        "Text": "",
    }


def value_for_key(value: Mapping[str, Any], key: str) -> tuple[Any, str | None]:
    if key in value:
        return value[key], None
    lower = {str(item_key).lower(): item_key for item_key in value}
    actual = lower.get(key.lower())
    if actual is None:
        return None, None
    return value[actual], f"canonicalize-key:{actual}->{key}"


def normalize_object_names(value: Any) -> tuple[list[str], list[str]]:
    if value is None:
        return [], ["fill-missing:ObjectName"]
    if isinstance(value, list):
        return [json_compact(item) for item in value], []
    return [json_compact(value)], ["wrap-scalar:ObjectName"]


def coordinate(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [value[0], value[1]]
    return None


def normalize_position(value: Any) -> tuple[list[dict[str, list[Any]]], list[str]]:
    if value is None:
        return [], ["fill-missing:Position"]
    operations: list[str] = []
    if isinstance(value, Mapping):
        values: list[Any] = [value]
        operations.append("wrap-object:Position")
    elif coordinate(value) is not None and not any(isinstance(item, (list, tuple, Mapping)) for item in value):
        point = coordinate(value)
        return [{"Begin": point, "End": list(point or [])}], ["coordinate-pair-to-range:Position"]
    elif isinstance(value, list):
        if len(value) == 1 and coordinate(value[0]) is not None:
            point = coordinate(value[0])
            return [{"Begin": point, "End": list(point or [])}], ["nested-coordinate-to-range:Position"]
        values = value
    else:
        return [], ["replace-invalid-with-empty:Position"]

    normalized: list[dict[str, list[Any]]] = []
    for item in values:
        if coordinate(item) is not None:
            point = coordinate(item)
            normalized.append({"Begin": point, "End": list(point or [])})
            operations.append("coordinate-pair-to-range:Position")
            continue
        if not isinstance(item, Mapping):
            operations.append("drop-invalid-item:Position")
            continue
        begin = item.get("Begin", item.get("Start", item.get("Bound")))
        end = item.get("End", begin)
        begin_pair = coordinate(begin)
        end_pair = coordinate(end)
        if begin_pair is None and end_pair is None:
            operations.append("drop-invalid-item:Position")
            continue
        if begin_pair is None:
            begin_pair = list(end_pair or [])
            operations.append("copy-end-to-begin:Position")
        if end_pair is None:
            end_pair = list(begin_pair)
            operations.append("copy-begin-to-end:Position")
        normalized.append({"Begin": begin_pair, "End": end_pair})
    if not normalized and values:
        operations.append("replace-invalid-with-empty:Position")
    return normalized, operations


def normalize_num(value: Any) -> tuple[list[Any], list[str]]:
    if value is None:
        return [], ["fill-missing:Num"]
    if isinstance(value, list):
        return value, []
    return [value], ["wrap-scalar:Num"]


def normalize_binding(value: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return a six-field Binding without inventing semantic content."""
    operations: list[str] = []
    quality: list[str] = []
    if not isinstance(value, Mapping):
        item = blank_binding()
        operations.append("nonobject-to-blank-binding")
        quality.append("no-binding-object")
        if isinstance(value, list) and coordinate(value) is not None:
            item["Position"], position_ops = normalize_position(value)
            operations.extend(position_ops)
            quality.append("interpreted-coordinate-only")
        elif isinstance(value, list) and value and all(isinstance(part, str) for part in value):
            item["ObjectName"] = list(value)
            operations.append("string-array-to-objectname")
            quality.append("interpreted-objectname-only")
        return item, operations, quality

    item = blank_binding()
    known_lower = {key.lower() for key in BINDING_KEYS}
    extras = [str(key) for key in value if str(key).lower() not in known_lower]
    if extras:
        operations.append("drop-extra-fields")
        quality.append("extra-fields-retained-in-audit")

    raw, key_op = value_for_key(value, "ObjectName")
    item["ObjectName"], field_ops = normalize_object_names(raw)
    operations.extend(([key_op] if key_op else []) + field_ops)
    for key in ("DataName", "Trend", "Text"):
        raw, key_op = value_for_key(value, key)
        if key_op:
            operations.append(key_op)
        if raw is None:
            operations.append(f"fill-missing:{key}")
            item[key] = ""
        elif isinstance(raw, str):
            item[key] = raw
        else:
            item[key] = json_compact(raw)
            operations.append(f"serialize-nonstring:{key}")
    raw_position, key_op = value_for_key(value, "Position")
    if key_op:
        operations.append(key_op)
    item["Position"], position_ops = normalize_position(raw_position)
    operations.extend(position_ops)
    raw_num, key_op = value_for_key(value, "Num")
    if key_op:
        operations.append(key_op)
    item["Num"], num_ops = normalize_num(raw_num)
    operations.extend(num_ops)
    if any(isinstance(number, str) and re.search(r"[^0-9eE+.,\-\s]", number) for number in item["Num"]):
        quality.append("Num-retains-nonnumeric-content")
    return item, list(dict.fromkeys(operations)), list(dict.fromkeys(quality))


def payload_bindings(payload: Any) -> tuple[list[dict[str, Any]], list[str], list[str], str]:
    operations: list[str] = []
    quality: list[str] = []
    result: Any
    if isinstance(payload, Mapping) and "result" in payload:
        result = payload.get("result")
        if not isinstance(payload.get("reason", ""), str):
            operations.append("serialize-nonstring:reason")
    elif isinstance(payload, Mapping) and "Binding" in payload:
        result = payload.get("Binding")
        operations.append("Binding-wrapper-to-result")
    elif isinstance(payload, Mapping) and any(str(key).lower() in {name.lower() for name in BINDING_KEYS} for key in payload):
        result = [payload]
        operations.append("wrap-binding-object")
    elif isinstance(payload, list) and coordinate(payload) is not None:
        result = [payload]
        operations.append("wrap-coordinate-result")
    elif isinstance(payload, list):
        result = payload
    elif payload is None or isinstance(payload, Mapping):
        result = []
        operations.append("nonbinding-payload-to-empty-result")
        quality.append("no-binding-structure")
    else:
        result = [payload]
        operations.append("wrap-scalar-result")
        quality.append("no-binding-structure")

    if result is None:
        result = []
        operations.append("null-result-to-empty-array")
    elif isinstance(result, Mapping):
        result = [result]
        operations.append("wrap-result-object")
    elif not isinstance(result, list):
        result = [result]
        operations.append("wrap-result-scalar")

    normalized: list[dict[str, Any]] = []
    for value in result:
        binding, binding_ops, binding_quality = normalize_binding(value)
        normalized.append(binding)
        operations.extend(binding_ops)
        quality.extend(binding_quality)
    status = "relaxed_payload_empty" if not normalized else "relaxed_payload_recovered"
    return normalized, list(dict.fromkeys(operations)), list(dict.fromkeys(quality)), status


def contains_binding_shape(value: Any) -> bool:
    binding_names = {key.lower() for key in BINDING_KEYS}
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        return bool(keys & binding_names) or "result" in keys or "binding" in keys
    if isinstance(value, list):
        return not value or any(contains_binding_shape(item) for item in value)
    return False


def extract_balanced_json(text: str) -> tuple[Any | None, str | None]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if contains_binding_shape(value):
            return value, "json-fragment"
    return None, None


def explicit_output_region(raw: str) -> tuple[str | None, str | None]:
    begin = re.search(r"\[BEGIN\](.*?)\[END\]", raw, flags=re.IGNORECASE | re.DOTALL)
    if begin:
        return begin.group(1).strip(), "begin-end"
    response = re.search(r"response\s+to\s+the\s+latest\s+input\s*[:\-]?", raw, flags=re.IGNORECASE)
    if response:
        return raw[response.end():].strip(), "latest-input-response"
    marker = re.search(r"##\s*Output\s+(content|results?)\b\s*[:\-]?", raw, flags=re.IGNORECASE)
    if marker:
        region = raw[marker.end():]
        region = re.split(r"\n\s*(?:#|##)\s*(?:Example|Related)\b", region, maxsplit=1, flags=re.IGNORECASE)[0]
        return region.strip(), f"output-{marker.group(1).lower()}"
    if re.match(r"^\s*##\s*Output\s+\{", raw, flags=re.IGNORECASE):
        return re.sub(r"^\s*##\s*Output\s+", "", raw, count=1, flags=re.IGNORECASE).strip(), "leading-output"
    if re.match(r"^\s*\{", raw):
        return raw.strip(), "leading-json-object"
    return None, None


def repair_mistral_closing_brace(region: str) -> tuple[str, bool]:
    repaired, count = re.subn(
        r'("Text"\s*:\s*"(?:[^"\\]|\\.)*")\s*\]\s*,\s*"reason"',
        r'\1}],"reason"',
        region,
        count=1,
        flags=re.DOTALL,
    )
    return repaired, count == 1


def decode_json_region(region: str) -> tuple[Any | None, list[str]]:
    operations: list[str] = []
    value, _ = extract_balanced_json(region)
    if value is not None:
        return value, operations
    repaired, changed = repair_mistral_closing_brace(region)
    if changed:
        operations.append("insert-missing-binding-closing-brace")
    repaired2, count = re.subn(r",\s*([}\]])", r"\1", repaired)
    if count:
        operations.append("remove-trailing-comma")
    value, _ = extract_balanced_json(repaired2)
    return value, operations


def quoted_field(region: str, key: str) -> str | None:
    match = re.search(
        rf'["\']?{re.escape(key)}["\']?\s*:\s*("(?:[^"\\]|\\.)*"|\'[^\']*\')',
        region,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    token = match.group(1)
    if token.startswith("\'"):
        return token[1:-1]
    try:
        return str(json.loads(token))
    except json.JSONDecodeError:
        return token[1:-1]


def pseudo_binding(region: str, *, allow_flan_shape: bool) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    flan_shape = bool(re.match(r'^\s*\[\s*["\'](?:objectName|ObjectName)["\']\s*:', region))
    if not (allow_flan_shape and flan_shape) and not re.search(r'["\']?ObjectName["\']?\s*:', region, flags=re.IGNORECASE):
        return None, [], []
    item = blank_binding()
    operations = ["parse-pseudo-binding"]
    quality: list[str] = []
    object_match = re.search(
        r'["\']?ObjectName["\']?\s*:\s*\[\s*("(?:[^"\\]|\\.)*"|\'[^\']*\')',
        region,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if object_match:
        token = object_match.group(1)
        item["ObjectName"] = [token[1:-1]]
    for key in ("DataName", "Trend", "Text"):
        value = quoted_field(region, key)
        if value is not None:
            item[key] = value
    begin = re.search(r'["\']?Begin["\']?\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', region, flags=re.IGNORECASE)
    end = re.search(r'["\']?End["\']?\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', region, flags=re.IGNORECASE)
    if begin or end:
        first = [int((begin or end).group(1)), int((begin or end).group(2))]
        last = [int((end or begin).group(1)), int((end or begin).group(2))]
        item["Position"] = [{"Begin": first, "End": last}]
    num = re.search(r'["\']?Num["\']?\s*:\s*\[(.*?)\]', region, flags=re.IGNORECASE | re.DOTALL)
    if num:
        try:
            parsed = json.loads(f"[{num.group(1)}]")
            item["Num"] = parsed
        except json.JSONDecodeError:
            if num.group(1).strip():
                item["Num"] = [num.group(1).strip()]
                operations.append("retain-unparsed-num-content")
    populated = sum(bool(item[key]) for key in BINDING_KEYS)
    if populated < 2:
        return None, operations, ["insufficient-pseudo-fields"]
    if flan_shape:
        quality.append("known-flan-malformed-shape")
    return item, operations, quality


def looks_prompt_echo(raw: str) -> bool:
    indicators = (
        "# Financial data-text binding",
        "## Binding coordinate contract",
        "## Chart data (lossless compact form)",
        "Return exactly {\"result\"",
        "# Example",
        "## Prompt template",
    )
    return sum(marker.lower() in raw.lower() for marker in indicators) >= 2


def has_substantial_content(bindings: Sequence[Mapping[str, Any]]) -> bool:
    for binding in bindings:
        text = str(binding.get("Text") or "")
        if "## Example" not in text and (
            any(str(value).strip() for value in binding.get("ObjectName") or [])
            or str(binding.get("DataName") or "").strip()
            or text.strip()
            or bool(binding.get("Num"))
        ):
            return True
    return False


def recover_unavailable_raw(raw: str) -> tuple[list[dict[str, Any]] | None, list[str], list[str], str, str | None]:
    stripped = raw.strip()
    if not stripped:
        return None, [], [], "unavailable_empty", None
    if re.match(r"^\[?##_rowOC__CO_", stripped) or (
        len(stripped) < 600 and stripped.count("[") >= 10 and not re.search(r"[A-Za-z]{4}", stripped)
    ):
        return None, [], [], "unavailable_degenerate_tokens", None

    flan_shape = bool(re.match(r'^\s*\[\s*["\'](?:objectName|ObjectName)["\']\s*:', stripped))
    if flan_shape:
        binding, operations, quality = pseudo_binding(stripped, allow_flan_shape=True)
        if binding is not None:
            return [binding], operations, quality, "raw_flan_shape_recovered", "known-flan-shape"

    region, marker = explicit_output_region(stripped)
    if region is None:
        status = "unavailable_prompt_echo" if looks_prompt_echo(stripped) else "unavailable_no_explicit_output_region"
        return None, [], [], status, None
    value, operations = decode_json_region(region)
    if value is not None:
        bindings, normalize_ops, quality, _ = payload_bindings(value)
        operations.extend(normalize_ops)
        if bindings and has_substantial_content(bindings):
            return bindings, list(dict.fromkeys(operations)), quality, "raw_json_recovered", marker
    binding, pseudo_ops, quality = pseudo_binding(region, allow_flan_shape=False)
    if binding is not None and has_substantial_content([binding]):
        return [binding], list(dict.fromkeys(operations + pseudo_ops)), quality, "raw_pseudo_recovered", marker
    status = "unavailable_prompt_echo" if looks_prompt_echo(stripped) else "unavailable_no_binding_structure"
    return None, operations, quality, status, marker


def relaxed_binding_valid(value: Any) -> tuple[bool, str]:
    if not isinstance(value, Mapping) or set(value.keys()) != set(BINDING_KEYS):
        return False, "binding_keys_or_order"
    if not isinstance(value["ObjectName"], list) or not all(isinstance(item, str) for item in value["ObjectName"]):
        return False, "ObjectName_not_string_array"
    if not all(isinstance(value[key], str) for key in ("DataName", "Trend", "Text")):
        return False, "string_field_type"
    if not isinstance(value["Num"], list):
        return False, "Num_not_array"
    if not isinstance(value["Position"], list):
        return False, "Position_not_array"
    for position in value["Position"]:
        if not isinstance(position, Mapping) or set(position) != {"Begin", "End"}:
            return False, "Position_item_shape"
        if coordinate(position["Begin"]) is None or coordinate(position["End"]) is None:
            return False, "Position_coordinate_shape"
    return True, "valid"


def write_inventory(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_inventory.tsv":
            rows.append((str(path.relative_to(root)), path.stat().st_size, sha256_file(path)))
    with (root / "sha256_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(rows)


def build(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    require(not output_root.exists(), f"output root already exists: {output_root}")
    source_validation = validate_v2(source_root)
    source_dataset = read_json(source_root / "dataset_manifest.json")
    require(source_dataset.get("protocol") == SOURCE_PROTOCOL, "source protocol mismatch")
    source_dataset_hash = sha256_file(source_root / "dataset_manifest.json")
    source_binding_hash = sha256_file(source_root / "binding.jsonl")
    fingerprint = stable_sha256(
        {
            "protocol": PROTOCOL,
            "policy": POLICY,
            "materializerSha256": sha256_file(Path(__file__).resolve()),
            "sourceDatasetManifestSha256": source_dataset_hash,
            "sourceBindingSha256": source_binding_hash,
        }
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent))
    all_rows: list[dict[str, Any]] = []
    all_bindings: list[dict[str, Any]] = []
    all_audit: list[dict[str, Any]] = []
    all_unrecoverable: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []

    for source_manifest_path in sorted((source_root / "manifests").glob("*.json")):
        source_manifest = read_json(source_manifest_path)
        output_id = str(source_manifest["outputId"])
        run = int(source_manifest["run"])
        source_group = str(source_manifest["sourceGroup"])
        seed = source_manifest.get("seed")
        v2_rows = read_jsonl(Path(source_manifest["files"]["binding"]))
        v2_predictions = read_jsonl(Path(source_manifest["files"]["predictions"]))
        v2_rejected = {
            (int(row["index"]), str(row["source"])): row
            for row in read_jsonl(Path(source_manifest["files"]["rejectedRows"]))
        }
        require(len(v2_rows) == len(v2_predictions) == 85, f"row count mismatch: {output_id} run {run}")
        run_rows: list[dict[str, Any]] = []
        run_bindings: list[dict[str, Any]] = []
        run_audit: list[dict[str, Any]] = []
        run_unrecoverable: list[dict[str, Any]] = []
        run_predictions: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        operation_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        for v2_row, prediction in zip(v2_rows, v2_predictions):
            index = int(v2_row["index"])
            source = str(v2_row["source"])
            require((index, source) == (int(prediction["index"]), str(prediction["source"])), "prediction order mismatch")
            source_status = str(v2_row["candidateStatus"])
            operations: list[str] = []
            quality: list[str] = []
            marker: str | None = None
            payload: Any = None
            raw = str(prediction.get("rawResponse") or "")
            if bool(v2_row.get("schemaValid")):
                bindings = list(v2_row["Binding"])
                status = "source_v2_valid"
                relaxed_valid = True
            elif source_status == "repair_schema_invalid":
                rejected = v2_rejected[(index, source)]
                payload = rejected.get("repairPayload")
                bindings, operations, quality, status = payload_bindings(payload)
                relaxed_valid = True
            elif source_status == "repair_unavailable":
                bindings, operations, quality, status, marker = recover_unavailable_raw(raw)
                relaxed_valid = bindings is not None
                if bindings is None:
                    bindings = []
            else:
                bindings = []
                status = "unavailable_no_repair_record"
                relaxed_valid = False
            for binding in bindings:
                valid, reason = relaxed_binding_valid(binding)
                require(valid, f"relaxed Binding invalid: {output_id} run {run} index {index}: {reason}")
            status_counts[status] += 1
            operation_counts.update(operations)
            quality_counts.update(quality)
            prediction_out = dict(prediction)
            prediction_out["result"] = bindings
            prediction_out["formatValid"] = relaxed_valid
            prediction_out["bindingCandidate"] = {
                "protocol": PROTOCOL,
                "status": status,
                "diagnosticOnly": True,
                "claimEligible": False,
                "sourceV2Status": source_status,
                "recoveryMarker": marker,
                "repairOperations": operations,
                "qualityFlags": quality,
            }
            run_predictions.append(prediction_out)
            row = {
                "schemaVersion": 3,
                "protocol": PROTOCOL,
                "outputId": output_id,
                "run": run,
                "seed": seed,
                "index": index,
                "source": source,
                "sourceGroup": source_group,
                "candidateStatus": status,
                "sourceV2Status": source_status,
                "strictSchemaValid": bool(v2_row.get("schemaValid")),
                "formatValid": relaxed_valid,
                "relaxedSchemaValid": relaxed_valid,
                "bindingCount": len(bindings),
                "Binding": bindings,
                "repairOperations": operations,
                "qualityFlags": quality,
                "diagnosticOnly": True,
                "claimEligible": False,
                "goldAccessed": False,
                "provenance": {
                    "sourceV2Root": str(source_root),
                    "sourceV2Manifest": str(source_manifest_path),
                    "sourceV2ManifestSha256": sha256_file(source_manifest_path),
                    "sourceV2BindingSha256": source_manifest["hashes"]["binding"],
                    "sourceGenerationManifest": source_manifest["source"]["manifest"],
                    "sourceGenerationManifestSha256": source_manifest["source"]["manifestSha256"],
                    "rawResponseSha256": prediction.get("rawResponseSha256") or stable_sha256(raw),
                },
            }
            run_rows.append(row)
            for binding_index, binding in enumerate(bindings):
                record = {
                    "schemaVersion": 3,
                    "protocol": PROTOCOL,
                    "candidateId": candidate_id(fingerprint, output_id, run, source, index, binding_index),
                    "outputId": output_id,
                    "run": run,
                    "seed": seed,
                    "index": index,
                    "source": source,
                    "sourceGroup": source_group,
                    "bindingIndex": binding_index,
                    **binding,
                    "candidateStatus": status,
                    "sourceV2Status": source_status,
                    "repairOperations": operations,
                    "qualityFlags": quality,
                    "diagnosticOnly": True,
                    "claimEligible": False,
                }
                run_bindings.append(record)
            if source_status in {"repair_schema_invalid", "repair_unavailable"}:
                audit = {
                    "schemaVersion": 3,
                    "protocol": PROTOCOL,
                    "outputId": output_id,
                    "run": run,
                    "seed": seed,
                    "index": index,
                    "source": source,
                    "sourceGroup": source_group,
                    "sourceV2Status": source_status,
                    "decision": status,
                    "recovered": relaxed_valid,
                    "bindingCount": len(bindings),
                    "recoveryMarker": marker,
                    "repairOperations": operations,
                    "qualityFlags": quality,
                    "repairPayload": payload,
                    "rawResponse": raw,
                    "rawResponseSha256": prediction.get("rawResponseSha256") or stable_sha256(raw),
                    "goldAccessed": False,
                }
                run_audit.append(audit)
                if not relaxed_valid:
                    run_unrecoverable.append(audit)

        relative = Path("cases") / output_id / f"run_{run:02d}"
        stage_dir = staging / relative
        final_dir = output_root / relative
        paths = {
            "predictions": stage_dir / "predictions.binding_candidates.jsonl",
            "binding": stage_dir / "binding.jsonl",
            "bindings": stage_dir / "bindings.jsonl",
            "repairAudit": stage_dir / "repair_audit.jsonl",
            "unrecoverableRows": stage_dir / "unrecoverable_rows.jsonl",
        }
        write_jsonl(paths["predictions"], run_predictions)
        write_jsonl(paths["binding"], run_rows)
        os.link(paths["binding"], stage_dir / "rows.jsonl")
        write_jsonl(paths["bindings"], run_bindings)
        write_jsonl(paths["repairAudit"], run_audit)
        write_jsonl(paths["unrecoverableRows"], run_unrecoverable)
        files = {name: str(final_dir / path.name) for name, path in paths.items()}
        files["rows"] = str(final_dir / "rows.jsonl")
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        hashes["rows"] = sha256_file(stage_dir / "rows.jsonl")
        manifest = {
            "schemaVersion": 3,
            "protocol": PROTOCOL,
            "policy": POLICY,
            "status": "completed_diagnostic_binding_candidates",
            "official": False,
            "diagnosticOnly": True,
            "claimEligible": False,
            "goldAccessed": False,
            "outputId": output_id,
            "run": run,
            "seed": seed,
            "sourceGroup": source_group,
            "expectedRows": 85,
            "acceptedRows": sum(row["formatValid"] for row in run_rows),
            "rejectedRows": len(run_unrecoverable),
            "bindingCount": len(run_bindings),
            "candidateStatusCounts": dict(sorted(status_counts.items())),
            "repairOperationCounts": dict(sorted(operation_counts.items())),
            "qualityFlagCounts": dict(sorted(quality_counts.items())),
            "files": files,
            "hashes": hashes,
            "source": {
                "generationRoot": source_manifest["source"]["generationRoot"],
                "manifest": source_manifest["source"]["manifest"],
                "manifestSha256": source_manifest["source"]["manifestSha256"],
                "predictions": source_manifest["source"]["predictions"],
                "predictionsSha256": source_manifest["source"]["predictionsSha256"],
                "nonformalRepair": source_manifest["source"]["nonformalRepair"],
                "nonformalRepairSha256": source_manifest["source"]["nonformalRepairSha256"],
                "generationProtocol": source_manifest["source"]["generationProtocol"],
                "compatibilityFingerprint": source_manifest["source"]["compatibilityFingerprint"],
            },
            "sourceV2": {
                "root": str(source_root),
                "manifest": str(source_manifest_path),
                "manifestSha256": sha256_file(source_manifest_path),
                "protocol": SOURCE_PROTOCOL,
            },
            "route": source_manifest.get("route"),
            "declaredRoute": source_manifest.get("declaredRoute"),
            "effectiveRoute": source_manifest.get("effectiveRoute"),
            "compatibilityFingerprint": fingerprint,
        }
        write_json(staging / "manifests" / f"{output_id}__run_{run:02d}.json", manifest)
        run_manifests.append(manifest)
        all_rows.extend(run_rows)
        all_bindings.extend(run_bindings)
        all_audit.extend(run_audit)
        all_unrecoverable.extend(run_unrecoverable)

    require(len(all_rows) == 28900, "aggregate row count mismatch")
    require(len(all_audit) == 8418, "problem-row audit count mismatch")
    binding_path = staging / "binding.jsonl"
    write_jsonl(binding_path, all_rows)
    os.link(binding_path, staging / "rows.jsonl")
    write_jsonl(staging / "bindings.jsonl", all_bindings)
    write_jsonl(staging / "repair_audit.jsonl", all_audit)
    write_jsonl(staging / "unrecoverable_rows.jsonl", all_unrecoverable)
    statuses = Counter(str(row["candidateStatus"]) for row in all_rows)
    v2_statuses = Counter(str(row["sourceV2Status"]) for row in all_rows)
    recovered_unavailable = sum(row["sourceV2Status"] == "repair_unavailable" and row["formatValid"] for row in all_rows)
    recovered_schema_invalid = sum(row["sourceV2Status"] == "repair_schema_invalid" and row["formatValid"] for row in all_rows)
    cases = sorted({str(row["outputId"]) for row in all_rows})
    dataset = {
        "schemaVersion": 3,
        "protocol": PROTOCOL,
        "policy": POLICY,
        "status": "complete",
        "createdAt": utc_now(),
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "goldAccessed": False,
        "outputRoot": str(output_root),
        "compatibilityFingerprint": fingerprint,
        "source": {
            "root": str(source_root),
            "protocol": SOURCE_PROTOCOL,
            "datasetManifest": str(source_root / "dataset_manifest.json"),
            "datasetManifestSha256": source_dataset_hash,
            "bindingSha256": source_binding_hash,
            "validation": source_validation,
        },
        "counts": {
            "cases": len(cases),
            "caseRuns": len(run_manifests),
            "rows": len(all_rows),
            "bindings": len(all_bindings),
            "acceptedRows": sum(row["formatValid"] for row in all_rows),
            "unrecoverableRows": len(all_unrecoverable),
            "auditedProblemRows": len(all_audit),
            "schemaInvalidRowsRecovered": recovered_schema_invalid,
            "unavailableRowsRecovered": recovered_unavailable,
            "sourceV2Status": dict(sorted(v2_statuses.items())),
            "candidateStatus": dict(sorted(statuses.items())),
        },
        "files": {
            "binding": str(output_root / "binding.jsonl"),
            "rows": str(output_root / "rows.jsonl"),
            "bindings": str(output_root / "bindings.jsonl"),
            "repairAudit": str(output_root / "repair_audit.jsonl"),
            "unrecoverableRows": str(output_root / "unrecoverable_rows.jsonl"),
        },
        "limitations": [
            "Diagnostic and claim-ineligible; it does not replace formal predictions or rankings.",
            "Shape repair does not assert that the recovered binding is semantically correct.",
            "Num strings and units are preserved rather than coerced to numbers.",
            "Prompt echoes, empty responses, and degenerate token streams remain unavailable.",
            "Four GPT-4.1 cases are intentionally excluded by the frozen v2 source matrix.",
        ],
    }
    dataset["hashes"] = {name: sha256_file(staging / Path(path).name) for name, path in dataset["files"].items()}
    write_json(staging / "dataset_manifest.json", dataset)
    (staging / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 6 relaxed Binding dataset (v3)",
                "",
                "This is a diagnostic, claim-ineligible projection of the frozen 34-case v2 root.",
                "Missing fields are blank-filled, recoverable types are normalized, and Num strings/units are retained.",
                f"All 1,333 schema-invalid rows and all 7,085 unavailable rows have individual decisions in `repair_audit.jsonl`.",
                "Unrecoverable prompt echoes, empty outputs, and degenerate token streams remain in `unrecoverable_rows.jsonl`.",
                "No gold or judge data was read.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_inventory(staging)
    os.replace(staging, output_root)
    return dataset


def validate_output(root: Path) -> dict[str, Any]:
    root = root.resolve()
    dataset = read_json(root / "dataset_manifest.json")
    require(dataset.get("protocol") == PROTOCOL, "dataset protocol mismatch")
    require(dataset.get("official") is False and dataset.get("diagnosticOnly") is True, "policy mismatch")
    require(dataset.get("claimEligible") is False and dataset.get("goldAccessed") is False, "claim/gold policy mismatch")
    source = dataset["source"]
    require(sha256_file(Path(source["datasetManifest"])) == source["datasetManifestSha256"], "source manifest SHA mismatch")
    require(sha256_file(Path(source["root"]) / "binding.jsonl") == source["bindingSha256"], "source binding SHA mismatch")
    for name, path_value in dataset["files"].items():
        path = Path(path_value)
        require(path.is_file(), f"aggregate file missing: {path}")
        require(sha256_file(path) == dataset["hashes"][name], f"aggregate SHA mismatch: {name}")
    rows = read_jsonl(Path(dataset["files"]["binding"]))
    rows_alias = read_jsonl(Path(dataset["files"]["rows"]))
    bindings = read_jsonl(Path(dataset["files"]["bindings"]))
    audit = read_jsonl(Path(dataset["files"]["repairAudit"]))
    unavailable = read_jsonl(Path(dataset["files"]["unrecoverableRows"]))
    require(rows == rows_alias, "binding/rows alias differs")
    require(len(rows) == int(dataset["counts"]["rows"]) == 28900, "row count mismatch")
    require(len(audit) == int(dataset["counts"]["auditedProblemRows"]) == 8418, "audit count mismatch")
    require(len(unavailable) == int(dataset["counts"]["unrecoverableRows"]), "unrecoverable count mismatch")
    row_keys = [(str(row["outputId"]), int(row["run"]), str(row["source"]), int(row["index"])) for row in rows]
    require(len(row_keys) == len(set(row_keys)), "duplicate row key")
    audit_keys = [(str(row["outputId"]), int(row["run"]), str(row["source"]), int(row["index"])) for row in audit]
    require(len(audit_keys) == len(set(audit_keys)), "duplicate audit key")
    require(sum(row["sourceV2Status"] == "repair_schema_invalid" for row in audit) == 1333, "schema-invalid audit count mismatch")
    require(sum(row["sourceV2Status"] == "repair_unavailable" for row in audit) == 7085, "unavailable audit count mismatch")
    require(all(row["recovered"] for row in audit if row["sourceV2Status"] == "repair_schema_invalid"), "not all schema-invalid rows normalized")
    require({tuple((row["outputId"], row["run"], row["source"], row["index"])) for row in unavailable} == {
        tuple((row["outputId"], row["run"], row["source"], row["index"])) for row in audit if not row["recovered"]
    }, "unrecoverable partition mismatch")
    bindings_by_row: Counter[tuple[str, int, str, int]] = Counter()
    for binding in bindings:
        value = {key: binding[key] for key in BINDING_KEYS}
        valid, reason = relaxed_binding_valid(value)
        require(valid, f"invalid long Binding: {reason}")
        key = (str(binding["outputId"]), int(binding["run"]), str(binding["source"]), int(binding["index"]))
        bindings_by_row[key] += 1
    cases: dict[str, set[int]] = defaultdict(set)
    pair_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        key = (str(row["outputId"]), int(row["run"]), str(row["source"]), int(row["index"]))
        cases[key[0]].add(key[1])
        pair_counts[(key[0], key[1])] += 1
        require(len(row["Binding"]) == int(row["bindingCount"]) == bindings_by_row[key], f"Binding count mismatch: {key}")
        require(bool(row["formatValid"]) == bool(row["relaxedSchemaValid"]), f"format flag mismatch: {key}")
        for binding in row["Binding"]:
            valid, reason = relaxed_binding_valid(binding)
            require(valid, f"invalid row Binding {key}: {reason}")
    require(len(cases) == 34 and all(runs == set(range(1, 11)) for runs in cases.values()), "case/run matrix mismatch")
    require(len(pair_counts) == 340 and all(count == 85 for count in pair_counts.values()), "85-row coverage mismatch")
    manifest_paths = sorted((root / "manifests").glob("*.json"))
    require(len(manifest_paths) == 340, "manifest count mismatch")
    per_run_rows: list[dict[str, Any]] = []
    per_run_bindings: list[dict[str, Any]] = []
    per_run_audit: list[dict[str, Any]] = []
    per_run_unavailable: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        require(manifest.get("protocol") == PROTOCOL, "run protocol mismatch")
        for name, path_value in manifest["files"].items():
            path = Path(path_value)
            require(path.is_file() and sha256_file(path) == manifest["hashes"][name], f"run artifact mismatch: {path}")
        run_rows = read_jsonl(Path(manifest["files"]["binding"]))
        predictions = read_jsonl(Path(manifest["files"]["predictions"]))
        require(len(run_rows) == len(predictions) == 85, "run row count mismatch")
        for row, prediction in zip(run_rows, predictions):
            require(row["Binding"] == prediction["result"], "prediction Binding mismatch")
            require(bool(row["formatValid"]) == bool(prediction["formatValid"]), "prediction format mismatch")
        per_run_rows.extend(run_rows)
        per_run_bindings.extend(read_jsonl(Path(manifest["files"]["bindings"])))
        per_run_audit.extend(read_jsonl(Path(manifest["files"]["repairAudit"])))
        per_run_unavailable.extend(read_jsonl(Path(manifest["files"]["unrecoverableRows"])))
    require(per_run_rows == rows, "aggregate/per-run rows differ")
    require(per_run_bindings == bindings, "aggregate/per-run bindings differ")
    require(per_run_audit == audit, "aggregate/per-run audit differs")
    require(per_run_unavailable == unavailable, "aggregate/per-run unavailable differs")
    inventory_path = root / "sha256_inventory.tsv"
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path != inventory_path}
    require({row["relative_path"] for row in inventory} == actual, "inventory coverage mismatch")
    for row in inventory:
        path = root / row["relative_path"]
        require(path.stat().st_size == int(row["size_bytes"]) and sha256_file(path) == row["sha256"], f"inventory mismatch: {path}")
    return {
        "status": "valid",
        "protocol": PROTOCOL,
        "root": str(root),
        "counts": dataset["counts"],
        "checks": {
            "complete34CaseMatrixExcludingGpt41": True,
            "all1333SchemaInvalidRowsNormalized": True,
            "all7085UnavailableRowsIndividuallyAudited": True,
            "rawResponsesRetained": True,
            "sourceV2HashesVerified": True,
            "perRunAggregateIdentity": True,
            "inventoryCoverageAndHashes": True,
            "goldAccessedDuringMaterialization": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "materialize":
            report = build(args.source_root, args.output_root)
        else:
            report = validate_output(args.root)
            if args.report:
                write_json(args.report, report)
    except (RelaxedMaterializationError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
