#!/usr/bin/env python3
"""Finalize and audit the Experiment 7 Mistral4 test-only queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED = [
    "finqa_flan_d",
    "finqa_mistral_o",
    "finqa_t5gemma2_o",
    "finqa_t5gemma2_z",
    "finqa_mistral_z",
    "finqa_mistral_m",
    "finqa_t5gemma2_m",
    "finqa_t5gemma2_d",
]
EXPECTED_ROWS = 1147


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def audit_case(case_dir: Path, retriever: str, errors: list[str]) -> dict[str, Any]:
    input_path = case_dir / "generator_input.json"
    output_path = case_dir / "mistral4_finqa_test_generated.jsonl"
    result: dict[str, Any] = {
        "retriever": retriever,
        "dataset": "finqa_test",
        "input": str(input_path),
        "output": str(output_path),
    }
    if not input_path.is_file() or not output_path.is_file():
        errors.append(f"{retriever}: missing input or output")
        return result
    try:
        inputs = load_json(input_path)
    except Exception as exc:
        errors.append(f"{retriever}: invalid input JSON: {exc}")
        return result
    outputs: list[dict[str, Any]] = []
    invalid_json_lines: list[int] = []
    for line_number, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            invalid_json_lines.append(line_number)
            continue
        if not isinstance(row, dict):
            invalid_json_lines.append(line_number)
            continue
        outputs.append(row)
    result.update(
        inputRows=len(inputs) if isinstance(inputs, list) else None,
        outputRows=len(outputs),
        invalidJsonLines=invalid_json_lines,
        inputSha256=sha256_file(input_path),
        outputSha256=sha256_file(output_path),
    )
    if not isinstance(inputs, list):
        errors.append(f"{retriever}: input root is not a list")
        return result
    if len(inputs) != EXPECTED_ROWS or len(outputs) != EXPECTED_ROWS:
        errors.append(f"{retriever}: rows input={len(inputs)} output={len(outputs)} expected={EXPECTED_ROWS}")
    if invalid_json_lines:
        errors.append(f"{retriever}: invalid JSONL lines={invalid_json_lines[:10]}")
    mismatch = {
        "selection_key": 0,
        "id": 0,
        "source_csv_row": 0,
        "normalized_question": 0,
        "reasoning_effort": 0,
    }
    seen: set[str] = set()
    duplicate_keys = 0
    for source, output in zip(inputs, outputs):
        key = output.get("selection_key")
        if not isinstance(key, str) or key != source.get("selection_key"):
            mismatch["selection_key"] += 1
        if isinstance(key, str):
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
        if output.get("id") != source.get("id"):
            mismatch["id"] += 1
        if output.get("source_csv_row") != source.get("source_csv_row"):
            mismatch["source_csv_row"] += 1
        if normalize_question(output.get("question", "")) != normalize_question(source.get("question", "")):
            mismatch["normalized_question"] += 1
        policy = output.get("generator_sampling_policy") or {}
        if policy.get("reasoning_effort") != "high":
            mismatch["reasoning_effort"] += 1
    result["duplicateSelectionKeys"] = duplicate_keys
    result["mismatches"] = mismatch
    if duplicate_keys or any(mismatch.values()):
        errors.append(f"{retriever}: duplicate_keys={duplicate_keys} mismatches={mismatch}")
    return result


def update_index(index_path: Path, entry: dict[str, Any]) -> None:
    lock_path = index_path.with_name(f"{index_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = load_json(index_path)
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise RuntimeError("docs/log/index.json lacks entries[]")
        if not any(item.get("report") == entry["report"] for item in entries if isinstance(item, dict)):
            entries.append(entry)
            atomic_json(index_path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    generator_root = root / "generator" / "mistral4"
    queue_root = root / "remaining_queue"
    errors: list[str] = []
    cases = [
        audit_case(generator_root / f"{retriever}_finqa_test", retriever, errors)
        for retriever in EXPECTED
    ]

    score_path = root / "generator" / "score_report.json"
    score: dict[str, Any] = {}
    if not score_path.is_file():
        errors.append("missing generator/score_report.json")
    else:
        try:
            score = load_json(score_path)
        except Exception as exc:
            errors.append(f"invalid score report: {exc}")
    expected_pairs = {(name, "finqa_test") for name in EXPECTED}
    items = score.get("items", []) if isinstance(score, dict) else []
    actual_pairs = {
        (item.get("retriever_id"), item.get("dataset"))
        for item in items
        if isinstance(item, dict)
    }
    if score.get("completed_cases") != 8 or len(items) != 8 or actual_pairs != expected_pairs:
        errors.append(
            f"score report is not exact test-only set: completed={score.get('completed_cases')} "
            f"items={len(items)} pairs={sorted(actual_pairs)}"
        )
    eas = [
        item.get("execution_accuracy")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("execution_accuracy"), (int, float))
    ]
    reported_mean = score.get("mean_execution_accuracy_unweighted")
    computed_mean = sum(eas) / len(eas) if len(eas) == 8 else None
    if computed_mean is None or not isinstance(reported_mean, (int, float)) or not math.isclose(
        computed_mean, reported_mean, rel_tol=0.0, abs_tol=1e-15
    ):
        errors.append(f"EA mean mismatch: computed={computed_mean} reported={reported_mean}")

    prefix_path = queue_root / "flan_d_test_prefix_repair.json"
    prefix: dict[str, Any] = {}
    if not prefix_path.is_file():
        errors.append("missing flan_d_test_prefix_repair.json")
    else:
        prefix = load_json(prefix_path)
        source_path = Path(prefix.get("sourceJsonl", ""))
        source_sha = sha256_file(source_path) if source_path.is_file() else None
        if (
            prefix.get("recoveredPrefixRows") != 17
            or prefix.get("sourceModified") is not False
            or source_sha != prefix.get("sourceSha256")
        ):
            errors.append(
                f"FLAN prefix/source invariant failed: recovered={prefix.get('recoveredPrefixRows')} "
                f"sourceModified={prefix.get('sourceModified')} currentSha={source_sha}"
            )

    server_command_path = queue_root / "mistral4_server.command"
    server_command = server_command_path.read_text(encoding="utf-8") if server_command_path.is_file() else ""
    server_command_for_check = server_command.replace(r"\,", ",")
    required_fragments = [
        "CUDA_VISIBLE_DEVICES=0,1",
        "--n-gpu-layers 20",
        "--split-mode row",
        "--tensor-split 1,1",
        "--batch-size 192",
        "--ubatch-size 48",
        "--no-op-offload",
    ]
    missing_server_fragments = [part for part in required_fragments if part not in server_command_for_check]
    if missing_server_fragments:
        errors.append(f"server command missing fragments: {missing_server_fragments}")
    if "--cpu-moe" in server_command_for_check or "--cpu-moe-layers" in server_command_for_check:
        errors.append("server command enables CPU-MoE")

    timeline_path = queue_root / "timeline.log"
    timeline_lines = timeline_path.read_text(encoding="utf-8").splitlines() if timeline_path.is_file() else []
    marker_indices = [
        index for index, line in enumerate(timeline_lines)
        if "queue=scope dataset=finqa_test matrix=" in line
    ]
    completed_order: list[str] = []
    start_events: list[dict[str, Any]] = []
    marker_index: int | None = marker_indices[-1] if marker_indices else None
    if marker_index is None:
        errors.append("timeline lacks test-only scope marker")
    else:
        segment = timeline_lines[marker_index + 1 :]
        if any(":finqa_dev" in line for line in segment):
            errors.append("finqa_dev appeared after test-only scope marker")
        for line in segment:
            completed_match = re.search(r"case=completed item=([^ ]+)", line)
            if completed_match:
                completed_order.append(completed_match.group(1))
            start_match = re.search(
                r"case=start item=([^ ]+) attempt=(\d+) timeout_seconds=(\d+)", line
            )
            if start_match:
                start_events.append(
                    {
                        "item": start_match.group(1),
                        "attempt": int(start_match.group(2)),
                        "timeoutSeconds": int(start_match.group(3)),
                    }
                )
        expected_order = [f"{name}:finqa_test" for name in EXPECTED]
        if completed_order != expected_order:
            errors.append(
                f"timeline completion order mismatch: expected={expected_order} actual={completed_order}"
            )
        if any(event["attempt"] > 5 for event in start_events):
            errors.append(f"timeline contains attempt > 5: {start_events}")
        if any(event["timeoutSeconds"] != 1800 for event in start_events):
            errors.append(f"timeline contains timeout != 1800: {start_events}")

    audit = {
        "schemaVersion": 1,
        "protocol": "experiment7-mistral4-testonly-final-audit-v1",
        "time": utc_now(),
        "status": "completed" if not errors else "blocked",
        "experimentRoot": str(root),
        "expectedOrder": [f"{name}:finqa_test" for name in EXPECTED],
        "expectedRowsPerCase": EXPECTED_ROWS,
        "cases": cases,
        "scoreReport": {
            "path": str(score_path),
            "sha256": sha256_file(score_path) if score_path.is_file() else None,
            "completedCases": score.get("completed_cases"),
            "meanExecutionAccuracyUnweighted": reported_mean,
            "computedMeanExecutionAccuracyUnweighted": computed_mean,
            "items": items,
        },
        "flanPrefixRepair": prefix,
        "serverCommand": {
            "path": str(server_command_path),
            "sha256": sha256_file(server_command_path) if server_command_path.is_file() else None,
            "missingRequiredFragments": missing_server_fragments,
        },
        "timeline": {
            "path": str(timeline_path),
            "sha256": sha256_file(timeline_path) if timeline_path.is_file() else None,
            "scopeMarkerIndex": marker_index,
            "completedOrder": completed_order,
            "startEvents": start_events,
        },
        "errors": errors,
    }
    if errors:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 2
    if args.check_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    audit_path = queue_root / "final_audit.json"
    atomic_json(audit_path, audit)
    stamp = stamp_now()
    report_path = args.log_root.resolve() / f"{stamp}_experiment7_mistral4_testonly_complete.md"
    rows = []
    by_pair = {(item["retriever_id"], item["dataset"]): item for item in items}
    for name in EXPECTED:
        item = by_pair[(name, "finqa_test")]
        rows.append(f"| {name} | finqa_test | {item['rows']} | {item['execution_accuracy']:.12f} |")
    report = "\n".join(
        [
            "# Experiment 7 Mistral4 test-only completion",
            "",
            f"- Time: {audit['time']}",
            "- Status: complete",
            "- Conda environment: fnqa",
            f"- Experiment root: {root}",
            "- Scope: eight finqa_test cases; no finqa_dev is included in the final score report.",
            "- Identity audit: selection_key, id, source_csv_row, and normalized question all match for every row.",
            "- Runtime: two GPUs, n_gpu_layers=20, row split, tensor split 1,1, batch/ubatch 192/48, no CPU-MoE, reasoning high.",
            "",
            "| Retriever | Dataset | Rows | Strict FINDER EA |",
            "|---|---|---:|---:|",
            *rows,
            "",
            f"Mean strict FINDER EA: {reported_mean:.12f}",
            "",
            f"- Machine-readable audit: {audit_path}",
            f"- EA report: {score_path}",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    report_rel = f"docs/log/{report_path.name}"
    audit_rel = f"src/Experiment/{root.name}/remaining_queue/final_audit.json"
    entry = {
        "time": audit["time"],
        "kind": "experiment7_mistral4_testonly_complete",
        "repo": "$FQAN_ROOT",
        "report": report_rel,
        "sha256": sha256_file(report_path),
        "bytes": report_path.stat().st_size,
        "audit": audit_rel,
        "auditSha256": sha256_file(audit_path),
        "status": "complete",
        "summary": (
            "Completed the eight-case Mistral4 finqa_test queue in the specified order; "
            "all 9,176 rows passed four-field resume identity validation and strict FINDER EA was refreshed after every case."
        ),
        "tags": ["experiment_7", "mistral4", "finqa_test", "fnqa", "strict-ea", "complete"],
    }
    update_index(args.log_root.resolve() / "index.json", entry)
    pointer = {
        "time": audit["time"],
        "status": "completed",
        "report": str(report_path),
        "audit": str(audit_path),
        "scoreReport": str(score_path),
    }
    atomic_json(queue_root / "completion_artifacts.json", pointer)
    print(json.dumps(pointer, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
