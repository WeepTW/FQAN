#!/usr/bin/env python3
"""Make every relaxed-v3 Experiment 6 row readable and evaluable.

The source relaxed-v3 root is immutable. Existing six-field Bindings are
copied verbatim. Previously unavailable rows become valid empty results unless
a narrowly delimited answer region contains a mechanically recoverable
Binding. RetFact and Reason are retained as auxiliary fields.

No gold or judge data is read. Prompt/examples are evidence, not answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


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
from materialize_experiment6_bindings_relaxed_v3 import (
    BINDING_KEYS,
    blank_binding,
    decode_json_region,
    payload_bindings,
    relaxed_binding_valid,
    validate_output as validate_v3,
)


PROTOCOL = "experiment6-binding-materialization-repaired-v4-unified34"
SOURCE_PROTOCOL = "experiment6-binding-materialization-relaxed-v3-unified34"
POLICY = "row-readable-lossless-auxiliary-preservation-v1"
EXPECTED_ROWS = 28_900
EXPECTED_CASE_RUNS = 340
EXPECTED_CASES = 34
EXPECTED_RUNS = set(range(1, 11))
EXPECTED_ROWS_PER_RUN = 85


class RepairedMaterializationError(RuntimeError):
    """Raised when source or output invariants fail."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairedMaterializationError(message)


def unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value != ""))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def flatten_auxiliary(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return unique([stringify(item) for item in value])
    return [stringify(value)]


def auxiliary_fields(payload: Any) -> tuple[list[str], list[str]]:
    """Collect explicit RetFact/Reason fields from parsed model payloads."""
    retfacts: list[str] = []
    reasons: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                folded = str(key).casefold()
                if folded == "retfact":
                    retfacts.extend(flatten_auxiliary(child))
                elif folded == "reason":
                    reasons.extend(flatten_auxiliary(child))
                elif folded in {"result", "binding", "bindings"}:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return unique(retfacts), unique(reasons)


def explicit_answer_region(raw: str) -> tuple[str | None, str | None]:
    """Return only a bounded answer region; never scan prompt examples."""
    match = re.search(r"\[BEGIN\](.*?)\[END\]", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), "BEGIN-END"
    response = re.search(
        r"response\s+to\s+the\s+latest\s+input\s*[:\-]?",
        raw,
        flags=re.IGNORECASE,
    )
    if response:
        return raw[response.end() :].strip(), "latest-input-response"
    marker = re.search(
        r"(?im)^\s*#{0,3}\s*Output\s+(?:result|results|content|data)\s*:\s*",
        raw,
    )
    if marker:
        region = raw[marker.end() :]
        region = re.split(
            r"\s+#{1,3}\s*(?:Example|Examples|Related)\b",
            region,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return region.strip(), "output-marker"
    leading = re.match(r"^\s*#{1,3}\s*Output\s+(?=[\[{])", raw, flags=re.IGNORECASE)
    if leading:
        return raw[leading.end() :].strip(), "leading-output"
    return None, None


def parse_quoted_or_bare_list(text: str) -> list[str]:
    values: list[str] = []
    for token in re.split(r"\s*,\s*", text.strip()):
        token = token.strip().strip("\"'")
        if token:
            values.append(token)
    return values


def field_string(region: str, field: str) -> str | None:
    match = re.search(
        rf'["\']?{re.escape(field)}["\']?\s*[:=]\s*'
        r'(""[^\n]*?""|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
        region,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    token = match.group(1)
    if token.startswith('""') and token.endswith('""'):
        return token[2:-2]
    try:
        return str(json.loads(token))
    except json.JSONDecodeError:
        return token[1:-1]


def parse_num_content(content: str) -> list[Any]:
    stripped = content.strip()
    if not stripped:
        return []
    try:
        value = json.loads(f"[{stripped}]")
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [
            token.strip().strip("\"'")
            for token in re.split(r"\s*,\s*", stripped)
            if token.strip()
        ]


def pseudo_binding(region: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse a six-field pseudo-object without guessing semantic values."""
    if not re.search(r'["\']?ObjectName["\']?\s*[:=]', region, flags=re.IGNORECASE):
        return None, []
    binding = blank_binding()
    operations = ["parse-explicit-pseudo-binding"]
    object_match = re.search(
        r'["\']?ObjectName["\']?\s*[:=]\s*\[(.*?)\]',
        region,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if object_match:
        binding["ObjectName"] = parse_quoted_or_bare_list(object_match.group(1))
    for field in ("DataName", "Trend", "Text"):
        value = field_string(region, field)
        if value is not None:
            binding[field] = value
    position_pattern = re.compile(
        r'["\']?Begin["\']?\s*[:=]\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        r'.*?'
        r'["\']?End["\']?\s*[:=]\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]',
        flags=re.IGNORECASE | re.DOTALL,
    )
    binding["Position"] = [
        {
            "Begin": [int(match.group(1)), int(match.group(2))],
            "End": [int(match.group(3)), int(match.group(4))],
        }
        for match in position_pattern.finditer(region)
    ]
    num_match = re.search(
        r'["\']?Num["\']?\s*[:=]\s*\[(.*?)\]',
        region,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if num_match:
        binding["Num"] = parse_num_content(num_match.group(1))
    if sum(bool(binding[field]) for field in BINDING_KEYS) < 2:
        return None, operations
    return binding, operations


def recover_explicit_answer(
    raw: str,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], str | None]:
    region, marker = explicit_answer_region(raw)
    if region is None:
        return [], [], [], [], None
    operations: list[str] = []
    if region.strip() in {"[]", "{}", '{"result":[]}', '{"result": [], "reason": ""}'}:
        return [], [], [], ["preserve-explicit-empty-result"], marker
    payload, json_operations = decode_json_region(region)
    operations.extend(json_operations)
    if payload not in (None, [], {}):
        bindings, normalize_operations, _, _ = payload_bindings(payload)
        retfacts, reasons = auxiliary_fields(payload)
        return bindings, retfacts, reasons, unique(operations + normalize_operations), marker
    binding, pseudo_operations = pseudo_binding(region)
    if binding is not None:
        return [binding], [], [], unique(operations + pseudo_operations), marker
    return [], [], [], operations, marker


def row_output_kind(
    raw: str,
    source_status: str,
    bindings: Sequence[Mapping[str, Any]],
    marker: str | None,
) -> str:
    if bindings and source_status.startswith("unavailable_"):
        return "recovered_binding"
    if bindings:
        return "binding_result"
    if not raw.strip():
        return "empty_result"
    if source_status == "unavailable_degenerate_tokens":
        return "gibberish_row"
    if source_status == "unavailable_prompt_echo":
        return "prompt_echo_row"
    if marker:
        return "explicit_empty_or_nonbinding_result"
    return "nonbinding_row"


def pure_gibberish_text(value: str) -> tuple[bool, str | None]:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        return False, None
    if "##_rowOC__CO_" in normalized:
        return True, "known-degenerate-token-prefix"
    words = re.findall(r"[^\W\d_]{2,}|\d+(?:\.\d+)?", normalized, flags=re.UNICODE)
    symbols = sum(not char.isalnum() and not char.isspace() for char in normalized)
    if not words and symbols / max(len(normalized), 1) >= 0.55:
        return True, "symbol-stream-without-lexical-token"
    tokens = re.findall(r"\S+", normalized)
    if len(tokens) >= 12 and len(set(tokens)) <= 2:
        return True, "repeated-token-stream"
    if len(normalized) >= 24 and len(set(normalized)) <= 4:
        return True, "low-character-diversity-stream"
    return False, None


def binding_gibberish(binding: Mapping[str, Any]) -> tuple[bool, str | None]:
    values = [
        *[stringify(item) for item in binding.get("ObjectName") or []],
        stringify(binding.get("DataName")),
        stringify(binding.get("Trend")),
        *[stringify(item) for item in binding.get("Num") or []],
        stringify(binding.get("Text")),
    ]
    compact = " ".join(value for value in values if value.strip())
    if compact.strip(" .-") == "":
        return False, None
    return pure_gibberish_text(compact)


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
    source_validation = validate_v3(source_root)
    source_dataset_path = source_root / "dataset_manifest.json"
    source_dataset = read_json(source_dataset_path)
    require(source_dataset.get("protocol") == SOURCE_PROTOCOL, "source protocol mismatch")
    fingerprint = stable_sha256(
        {
            "protocol": PROTOCOL,
            "policy": POLICY,
            "sourceDatasetSha256": sha256_file(source_dataset_path),
            "sourceBindingsSha256": sha256_file(source_root / "bindings.jsonl"),
        }
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))

    all_rows: list[dict[str, Any]] = []
    all_bindings: list[dict[str, Any]] = []
    all_audit: list[dict[str, Any]] = []
    all_gibberish: list[dict[str, Any]] = []
    all_nonbinding: list[dict[str, Any]] = []
    all_duplicates: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for source_manifest_path in sorted((source_root / "manifests").glob("*.json")):
        source_manifest = read_json(source_manifest_path)
        output_id = str(source_manifest["outputId"])
        run = int(source_manifest["run"])
        seed = int(source_manifest["seed"])
        source_group = str(source_manifest["sourceGroup"])
        source_rows = read_jsonl(Path(source_manifest["files"]["rows"]))
        source_predictions = read_jsonl(Path(source_manifest["files"]["predictions"]))
        source_audit = {
            (int(row["index"]), str(row["source"])): row
            for row in read_jsonl(Path(source_manifest["files"]["repairAudit"]))
        }
        require(
            len(source_rows) == len(source_predictions) == EXPECTED_ROWS_PER_RUN,
            "source run row count mismatch",
        )
        predictions_by_key = {
            (int(row["index"]), str(row["source"])): row for row in source_predictions
        }
        run_rows: list[dict[str, Any]] = []
        run_predictions: list[dict[str, Any]] = []
        run_bindings: list[dict[str, Any]] = []
        run_audit: list[dict[str, Any]] = []
        run_gibberish: list[dict[str, Any]] = []
        run_nonbinding: list[dict[str, Any]] = []
        run_duplicates: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()

        for source_row in source_rows:
            index = int(source_row["index"])
            source = str(source_row["source"])
            prediction = predictions_by_key[(index, source)]
            raw = str(prediction.get("rawResponse") or "")
            source_status = str(source_row["candidateStatus"])
            bindings = list(source_row["Binding"])
            operations = list(source_row.get("repairOperations") or [])
            retfacts: list[str] = []
            reasons = flatten_auxiliary(prediction.get("reason"))
            marker: str | None = None
            audit_source = source_audit.get((index, source))
            if audit_source:
                aux_retfacts, aux_reasons = auxiliary_fields(audit_source.get("repairPayload"))
                retfacts.extend(aux_retfacts)
                reasons.extend(aux_reasons)
            if not bool(source_row.get("formatValid")):
                recovered, explicit_retfacts, explicit_reasons, explicit_operations, marker = (
                    recover_explicit_answer(raw)
                )
                bindings = recovered
                retfacts.extend(explicit_retfacts)
                reasons.extend(explicit_reasons)
                operations.extend(explicit_operations)
            retfacts = unique(retfacts)
            reasons = unique(reasons)
            for binding in bindings:
                valid, valid_reason = relaxed_binding_valid(binding)
                require(
                    valid,
                    f"invalid repaired Binding: {output_id}/run {run}/{source}: {valid_reason}",
                )

            kind = row_output_kind(raw, source_status, bindings, marker)
            status = (
                "source_v3_preserved"
                if bool(source_row.get("formatValid"))
                else f"row_readable_{kind}"
            )
            status_counts[status] += 1
            raw_gibberish, raw_gibberish_reason = pure_gibberish_text(raw)
            row = {
                "schemaVersion": 4,
                "protocol": PROTOCOL,
                "outputId": output_id,
                "run": run,
                "seed": seed,
                "index": index,
                "source": source,
                "sourceGroup": source_group,
                "candidateStatus": status,
                "sourceV3Status": source_status,
                "rowOutputKind": kind,
                "rowReadable": True,
                "formatValid": True,
                "strictSchemaValid": bool(source_row.get("strictSchemaValid")),
                "bindingCount": len(bindings),
                "result": bindings,
                "Binding": bindings,
                "RetFact": retfacts,
                "Reason": reasons,
                "reason": "\n".join(reasons),
                "rawResponse": raw,
                "rawResponseSha256": prediction.get("rawResponseSha256")
                or stable_sha256(raw),
                "repairOperations": unique(operations),
                "recoveryMarker": marker,
                "gibberish": raw_gibberish,
                "gibberishReason": raw_gibberish_reason,
                "diagnosticOnly": True,
                "claimEligible": False,
                "goldAccessed": False,
                "provenance": {
                    "sourceV3Root": str(source_root),
                    "sourceV3Manifest": str(source_manifest_path),
                    "sourceV3ManifestSha256": sha256_file(source_manifest_path),
                },
            }
            run_rows.append(row)
            prediction_out = dict(prediction)
            prediction_out.update(
                {
                    "result": bindings,
                    "formatValid": True,
                    "reason": row["reason"],
                    "RetFact": retfacts,
                    "Reason": reasons,
                    "rowOutputKind": kind,
                    "bindingCandidate": {
                        "protocol": PROTOCOL,
                        "status": status,
                        "sourceV3Status": source_status,
                        "diagnosticOnly": True,
                        "claimEligible": False,
                        "repairOperations": row["repairOperations"],
                        "recoveryMarker": marker,
                        "gibberish": raw_gibberish,
                    },
                }
            )
            run_predictions.append(prediction_out)

            seen_bindings: dict[str, int] = {}
            for binding_index, binding in enumerate(bindings):
                digest = stable_sha256(binding)
                duplicate_of = seen_bindings.get(digest)
                seen_bindings.setdefault(digest, binding_index)
                is_gibberish, gibberish_reason = binding_gibberish(binding)
                record = {
                    "schemaVersion": 4,
                    "protocol": PROTOCOL,
                    "candidateId": candidate_id(
                        fingerprint, output_id, run, source, index, binding_index
                    ),
                    "outputId": output_id,
                    "run": run,
                    "seed": seed,
                    "index": index,
                    "source": source,
                    "sourceGroup": source_group,
                    "bindingIndex": binding_index,
                    **binding,
                    "RetFact": retfacts,
                    "Reason": reasons,
                    "candidateStatus": status,
                    "sourceV3Status": source_status,
                    "duplicateWithinRow": duplicate_of is not None,
                    "duplicateOfBindingIndex": duplicate_of,
                    "gibberish": is_gibberish,
                    "gibberishReason": gibberish_reason,
                    "diagnosticOnly": True,
                    "claimEligible": False,
                }
                run_bindings.append(record)
                if duplicate_of is not None:
                    run_duplicates.append(record)
                if is_gibberish:
                    run_gibberish.append(record)
            if raw_gibberish:
                run_gibberish.append(
                    {
                        "recordType": "row",
                        "outputId": output_id,
                        "run": run,
                        "seed": seed,
                        "index": index,
                        "source": source,
                        "rowOutputKind": kind,
                        "gibberishReason": raw_gibberish_reason,
                        "rawResponse": raw,
                        "rawResponseSha256": row["rawResponseSha256"],
                    }
                )
            if not bindings:
                run_nonbinding.append(row)
            if not bool(source_row.get("formatValid")):
                run_audit.append(
                    {
                        **row,
                        "previouslyUnrecoverable": True,
                        "bindingRecovered": bool(bindings),
                    }
                )

        relative = Path("cases") / output_id / f"run_{run:02d}"
        stage_dir = staging / relative
        final_dir = output_root / relative
        paths = {
            "predictions": stage_dir / "predictions.binding_candidates.jsonl",
            "binding": stage_dir / "binding.jsonl",
            "bindings": stage_dir / "bindings.jsonl",
            "repairAudit": stage_dir / "repair_audit.jsonl",
            "nonBindingRows": stage_dir / "non_binding_rows.jsonl",
            "gibberishRecords": stage_dir / "gibberish_records.jsonl",
            "duplicateBindings": stage_dir / "duplicate_bindings.jsonl",
        }
        write_jsonl(paths["predictions"], run_predictions)
        write_jsonl(paths["binding"], run_rows)
        os.link(paths["binding"], stage_dir / "rows.jsonl")
        write_jsonl(paths["bindings"], run_bindings)
        write_jsonl(paths["repairAudit"], run_audit)
        write_jsonl(paths["nonBindingRows"], run_nonbinding)
        write_jsonl(paths["gibberishRecords"], run_gibberish)
        write_jsonl(paths["duplicateBindings"], run_duplicates)
        files = {name: str(final_dir / path.name) for name, path in paths.items()}
        files["rows"] = str(final_dir / "rows.jsonl")
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        hashes["rows"] = sha256_file(stage_dir / "rows.jsonl")
        manifest = {
            "schemaVersion": 4,
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
            "expectedRows": EXPECTED_ROWS_PER_RUN,
            "acceptedRows": EXPECTED_ROWS_PER_RUN,
            "rejectedRows": 0,
            "bindingCount": len(run_bindings),
            "candidateStatusCounts": dict(sorted(status_counts.items())),
            "files": files,
            "hashes": hashes,
            "source": {
                **source_manifest["source"],
                "v3Root": str(source_root),
                "v3Manifest": str(source_manifest_path),
                "v3ManifestSha256": sha256_file(source_manifest_path),
            },
            "route": source_manifest.get("route"),
            "declaredRoute": source_manifest.get("declaredRoute"),
            "effectiveRoute": source_manifest.get("effectiveRoute"),
            "compatibilityFingerprint": fingerprint,
        }
        write_json(
            staging / "manifests" / f"{output_id}__run_{run:02d}.json", manifest
        )
        manifests.append(manifest)
        all_rows.extend(run_rows)
        all_bindings.extend(run_bindings)
        all_audit.extend(run_audit)
        all_gibberish.extend(run_gibberish)
        all_nonbinding.extend(run_nonbinding)
        all_duplicates.extend(run_duplicates)

    require(len(all_rows) == EXPECTED_ROWS, "aggregate row count mismatch")
    require(
        len(all_audit) == int(source_dataset["counts"]["unrecoverableRows"]),
        "v3 unrecoverable audit count mismatch",
    )
    aggregate_files = {
        "binding": staging / "binding.jsonl",
        "rows": staging / "rows.jsonl",
        "bindings": staging / "bindings.jsonl",
        "repairAudit": staging / "repair_audit.jsonl",
        "nonBindingRows": staging / "non_binding_rows.jsonl",
        "gibberishRecords": staging / "gibberish_records.jsonl",
        "duplicateBindings": staging / "duplicate_bindings.jsonl",
    }
    write_jsonl(aggregate_files["binding"], all_rows)
    os.link(aggregate_files["binding"], aggregate_files["rows"])
    write_jsonl(aggregate_files["bindings"], all_bindings)
    write_jsonl(aggregate_files["repairAudit"], all_audit)
    write_jsonl(aggregate_files["nonBindingRows"], all_nonbinding)
    write_jsonl(aggregate_files["gibberishRecords"], all_gibberish)
    write_jsonl(aggregate_files["duplicateBindings"], all_duplicates)
    cases = sorted({str(row["outputId"]) for row in all_rows})
    kinds = Counter(str(row["rowOutputKind"]) for row in all_rows)
    statuses = Counter(str(row["candidateStatus"]) for row in all_rows)
    gibberish_rows = sum(record.get("recordType") == "row" for record in all_gibberish)
    gibberish_bindings = len(all_gibberish) - gibberish_rows
    dataset = {
        "schemaVersion": 4,
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
            "datasetManifest": str(source_dataset_path),
            "datasetManifestSha256": sha256_file(source_dataset_path),
            "bindingsSha256": sha256_file(source_root / "bindings.jsonl"),
            "validation": source_validation,
        },
        "counts": {
            "cases": len(cases),
            "caseRuns": len(manifests),
            "rows": len(all_rows),
            "bindings": len(all_bindings),
            "acceptedRows": len(all_rows),
            "rejectedRows": 0,
            "rowReadable": len(all_rows),
            "previouslyUnrecoverableRows": len(all_audit),
            "previouslyUnrecoverableBindingsRecovered": sum(
                row["bindingRecovered"] for row in all_audit
            ),
            "nonBindingRows": len(all_nonbinding),
            "gibberishRows": gibberish_rows,
            "gibberishBindings": gibberish_bindings,
            "duplicateBindingsWithinRow": len(all_duplicates),
            "rowOutputKind": dict(sorted(kinds.items())),
            "candidateStatus": dict(sorted(statuses.items())),
        },
        "files": {
            name: str(output_root / path.name) for name, path in aggregate_files.items()
        },
        "limitations": [
            "Diagnostic and claim-ineligible; it does not replace formal predictions or rankings.",
            "A readable empty row is a valid no-prediction result, not an invented blank Binding.",
            "Only narrowly delimited answer regions can add Bindings; prompt/examples remain row evidence.",
            "RetFact and Reason are auxiliary and excluded from six-field scoring.",
            "Num strings and units remain unchanged; duplicates are flagged, not removed.",
            "Four GPT-4.1 cases remain excluded by the frozen v3 source matrix.",
        ],
    }
    dataset["hashes"] = {
        name: sha256_file(path) for name, path in aggregate_files.items()
    }
    write_json(staging / "dataset_manifest.json", dataset)
    (staging / "README.md").write_text(
        "# Experiment 6 repaired Binding rows (v4)\n\n"
        "Diagnostic-only projection of relaxed-v3. Every row is readable and "
        "evaluable. Empty results, RetFact, Reason, raw output, duplicates, and "
        "gibberish evidence are retained. No gold or judge data was read.\n",
        encoding="utf-8",
    )
    write_inventory(staging)
    os.replace(staging, output_root)
    return dataset


def validate_output(root: Path) -> dict[str, Any]:
    root = root.resolve()
    dataset = read_json(root / "dataset_manifest.json")
    require(dataset.get("protocol") == PROTOCOL, "dataset protocol mismatch")
    require(
        dataset.get("diagnosticOnly") is True
        and dataset.get("claimEligible") is False,
        "policy mismatch",
    )
    require(dataset.get("goldAccessed") is False, "gold access policy mismatch")
    source = dataset["source"]
    require(
        sha256_file(Path(source["datasetManifest"]))
        == source["datasetManifestSha256"],
        "source manifest SHA mismatch",
    )
    require(
        sha256_file(Path(source["root"]) / "bindings.jsonl")
        == source["bindingsSha256"],
        "source bindings SHA mismatch",
    )
    for name, path_value in dataset["files"].items():
        path = Path(path_value)
        require(
            path.is_file() and sha256_file(path) == dataset["hashes"][name],
            f"aggregate artifact mismatch: {name}",
        )
    rows = read_jsonl(Path(dataset["files"]["rows"]))
    rows_alias = read_jsonl(Path(dataset["files"]["binding"]))
    bindings = read_jsonl(Path(dataset["files"]["bindings"]))
    audit = read_jsonl(Path(dataset["files"]["repairAudit"]))
    require(rows == rows_alias, "binding/rows aggregate aliases differ")
    require(
        len(rows) == EXPECTED_ROWS == int(dataset["counts"]["rows"]),
        "aggregate row count mismatch",
    )
    require(
        len(audit)
        == int(dataset["counts"]["previouslyUnrecoverableRows"])
        == 4550,
        "repair audit count mismatch",
    )
    require(
        all(
            row.get("rowReadable") is True and row.get("formatValid") is True
            for row in rows
        ),
        "not all rows readable/evaluable",
    )
    row_keys = [
        (
            str(row["outputId"]),
            int(row["run"]),
            str(row["source"]),
            int(row["index"]),
        )
        for row in rows
    ]
    require(len(row_keys) == len(set(row_keys)), "duplicate row key")
    bindings_by_row: Counter[tuple[str, int, str, int]] = Counter()
    candidate_ids = set()
    for binding in bindings:
        candidate = str(binding["candidateId"])
        require(candidate not in candidate_ids, "duplicate candidateId")
        candidate_ids.add(candidate)
        valid, reason = relaxed_binding_valid(
            {key: binding[key] for key in BINDING_KEYS}
        )
        require(valid, f"invalid long Binding: {reason}")
        bindings_by_row[
            (
                str(binding["outputId"]),
                int(binding["run"]),
                str(binding["source"]),
                int(binding["index"]),
            )
        ] += 1
    cases: dict[str, set[int]] = defaultdict(set)
    pair_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        key = (
            str(row["outputId"]),
            int(row["run"]),
            str(row["source"]),
            int(row["index"]),
        )
        cases[key[0]].add(key[1])
        pair_counts[(key[0], key[1])] += 1
        require(row["result"] == row["Binding"], f"result/Binding mismatch: {key}")
        require(
            len(row["Binding"])
            == int(row["bindingCount"])
            == bindings_by_row[key],
            f"Binding count mismatch: {key}",
        )
        require(
            isinstance(row["RetFact"], list) and isinstance(row["Reason"], list),
            f"auxiliary type mismatch: {key}",
        )
    require(
        len(cases) == EXPECTED_CASES
        and all(runs == EXPECTED_RUNS for runs in cases.values()),
        "case/run matrix mismatch",
    )
    require(
        len(pair_counts) == EXPECTED_CASE_RUNS
        and all(count == EXPECTED_ROWS_PER_RUN for count in pair_counts.values()),
        "85-row coverage mismatch",
    )
    manifest_paths = sorted((root / "manifests").glob("*.json"))
    require(len(manifest_paths) == EXPECTED_CASE_RUNS, "run manifest count mismatch")
    per_rows: list[dict[str, Any]] = []
    per_bindings: list[dict[str, Any]] = []
    per_audit: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        require(manifest.get("protocol") == PROTOCOL, "run protocol mismatch")
        for name, path_value in manifest["files"].items():
            path = Path(path_value)
            require(
                path.is_file() and sha256_file(path) == manifest["hashes"][name],
                f"run artifact mismatch: {path}",
            )
        run_rows = read_jsonl(Path(manifest["files"]["rows"]))
        run_predictions = read_jsonl(Path(manifest["files"]["predictions"]))
        require(
            len(run_rows) == len(run_predictions) == EXPECTED_ROWS_PER_RUN,
            "run row count mismatch",
        )
        for row, prediction in zip(run_rows, run_predictions):
            require(row["Binding"] == prediction["result"], "prediction Binding mismatch")
            require(prediction.get("formatValid") is True, "prediction not evaluable")
        per_rows.extend(run_rows)
        per_bindings.extend(read_jsonl(Path(manifest["files"]["bindings"])))
        per_audit.extend(read_jsonl(Path(manifest["files"]["repairAudit"])))
    require(
        per_rows == rows and per_bindings == bindings and per_audit == audit,
        "aggregate/per-run identity mismatch",
    )
    inventory_path = root / "sha256_inventory.tsv"
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != inventory_path
    }
    require(
        {row["relative_path"] for row in inventory} == actual,
        "inventory coverage mismatch",
    )
    for record in inventory:
        path = root / record["relative_path"]
        require(
            path.stat().st_size == int(record["size_bytes"])
            and sha256_file(path) == record["sha256"],
            f"inventory mismatch: {path}",
        )
    return {
        "status": "valid",
        "protocol": PROTOCOL,
        "root": str(root),
        "counts": dataset["counts"],
        "checks": {
            "complete34CaseMatrixExcludingGpt41": True,
            "all4550RowsReadableAndEvaluable": True,
            "emptyRowsPreservedAsEmptyResults": True,
            "retFactAndReasonRetained": True,
            "gibberishAndDuplicatesAudited": True,
            "sourceV3HashesVerified": True,
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
    except (
        RepairedMaterializationError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
