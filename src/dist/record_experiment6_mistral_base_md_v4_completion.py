#!/usr/bin/env python3
"""Record an audited Mistral base m/d v4 completion in docs/log."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


FIELDS = ("ObjectName", "Trend", "Num", "Text", "Position", "DataName")
KIND = "experiment6_mistral_base_md_chat_template_v4_complete"


class RecordError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecordError(f"expected JSON object: {path}")
    return value


def logical_path(value: str, workspace: Path) -> str:
    path = Path(value)
    try:
        return "$FQAN_ROOT/" + str(path.resolve().relative_to(workspace.resolve()))
    except (OSError, ValueError):
        return value


def logicalize(value: Any, workspace: Path) -> Any:
    if isinstance(value, dict):
        return {key: logicalize(item, workspace) for key, item in value.items()}
    if isinstance(value, list):
        return [logicalize(item, workspace) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return logical_path(value, workspace)
    return value


def display(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def markdown(audit: Mapping[str, Any], json_relative: str) -> str:
    lines = [
        "# Experiment 6 Mistral base_m/base_d v4 completion",
        "",
        "> DIAGNOSTIC ONLY — formal 38-case ranking unchanged.",
        "",
        "## Coverage and contract",
        "",
        "- Cases: `6_mistral_base_m`, `6_mistral_base_d`.",
        "- Coverage: 20 case-runs × 85 Sources = 1,700 rows.",
        f"- Compatibility fingerprint: `{audit['compatibilityFingerprint']}`.",
        "- Route: direct-binding; adapter, converter, and generation cache disabled.",
        "- Native Mistral target-last chat template; prompt echo rows: 0.",
        "- Prompt identity: 170 case/Source pairs × 10 runs; stable=true; gold markers=0.",
        "- Binding projection: repaired-v4, gold-free materialization.",
        "- Text: GPT-5.5 medium, confidence 0.8, enabled evidence-gated semantic judge.",
        "",
        "## Six-field 10-run means",
        "",
        "| Case | Annotation | Precision | Recall | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for case in sorted(audit["scores"]):
        score = audit["scores"][case]
        for field in FIELDS:
            metrics = score["fields"][field]
            lines.append(
                f"| {case} | {field} | {display(metrics['precision'])} | "
                f"{display(metrics['recall'])} | {display(metrics['f1'])} |"
            )
        metrics = score["macro"]
        lines.append(
            f"| {case} | Macro (6 fields) | {display(metrics['precision'])} | "
            f"{display(metrics['recall'])} | {display(metrics['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Generation: `{audit['generationRoot']}`",
            f"- Binding projection: `{audit['bindingRoot']}`",
            f"- Evaluation: `{audit['evaluationRoot']}`",
            f"- Machine-readable audit: `{json_relative}`",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def record(audit_path: Path, log_root: Path, index_path: Path, workspace: Path) -> dict[str, Any]:
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    if audit.get("status") != "complete" or audit.get("protocol") != "experiment6-mistral-base-md-v4-completion-audit-v1":
        raise RecordError("completion audit is not complete or has wrong protocol")
    if set(audit.get("scores") or {}) != {"6_mistral_base_m", "6_mistral_base_d"}:
        raise RecordError("completion audit case set mismatch")
    for case, score in audit["scores"].items():
        if set(score.get("fields") or {}) != set(FIELDS):
            raise RecordError(f"completion audit field set mismatch: {case}")
        for field, metrics in score["fields"].items():
            if set(metrics) != {"precision", "recall", "f1"}:
                raise RecordError(f"completion audit metric set mismatch: {case} {field}")
    judge = audit.get("judge") or {}
    if (
        judge.get("model") != "gpt-5.5"
        or judge.get("reasoningEffort") != "medium"
        or float(judge.get("minimumConfidence", -1)) != 0.8
    ):
        raise RecordError("completion audit judge mismatch")
    prompt_identity = audit.get("promptIdentity") or {}
    if (
        prompt_identity.get("caseSources") != 170
        or prompt_identity.get("runsPerCaseSource") != 10
        or prompt_identity.get("stable") is not True
        or prompt_identity.get("goldMarkers") != 0
    ):
        raise RecordError("completion audit prompt identity mismatch")
    coverage = audit.get("coverage") or {}
    if coverage.get("cases") != 2 or coverage.get("caseRuns") != 20 or coverage.get("rows") != 1700:
        raise RecordError("completion audit coverage mismatch")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{stamp}_experiment6_mistral_base_md_chat_template_v4_complete"
    report_path = log_root / f"{base}.md"
    json_path = log_root / f"{base}.json"
    if report_path.exists() or json_path.exists():
        raise RecordError("completion log timestamp collision")
    logical = logicalize(audit, workspace)
    json_payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": KIND,
        "status": "complete_diagnostic_two_case",
        "completionAudit": logical,
        "completionAuditSource": logical_path(str(audit_path), workspace),
        "completionAuditSourceSha256": sha256_file(audit_path),
    }
    atomic_json(json_path, json_payload)
    json_relative = str(json_path.relative_to(workspace))
    report_path.write_text(markdown(logical, json_relative), encoding="utf-8")
    report_sha = sha256_file(report_path)
    json_sha = sha256_file(json_path)
    macros = {
        case: score["macro"] for case, score in logical["scores"].items()
    }
    entry = {
        "audit": str(json_path.relative_to(workspace)),
        "auditSha256": json_sha,
        "bytes": report_path.stat().st_size,
        "kind": KIND,
        "generationRoot": logical["generationRoot"],
        "repo": "$FQAN_ROOT",
        "report": str(report_path.relative_to(workspace)),
        "sha256": report_sha,
        "status": "complete_diagnostic_two_case",
        "summary": (
            "Completed Mistral-7B no-adaptor native target-last chat rerun for "
            "base_m/base_d (20 runs, 1,700 rows), repaired-v4 materialization, and "
            f"six-field GPT-5.5-medium evaluation; macro metrics={json.dumps(macros, sort_keys=True)}."
        ),
        "tags": [
            "experiment_6", "mistral-7b", "no-adaptor", "direct-binding",
            "native-chat-template", "repaired-v4", "v6.1.0", "gpt-5.5-medium",
            "six-field", "diagnostic-only", "complete",
        ],
        "time": json_payload["time"],
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        index = json.load(handle)
        entries = index.get("entries")
        if not isinstance(entries, list):
            raise RecordError("docs/log index entries missing")
        if any(
            item.get("kind") == KIND
            and item.get("generationRoot") == logical["generationRoot"]
            for item in entries
        ):
            raise RecordError("completion already recorded")
        entries.append(entry)
        handle.seek(0)
        json.dump(index, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "complete",
        "report": str(report_path),
        "reportSha256": report_sha,
        "audit": str(json_path),
        "auditSha256": json_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = record(args.completion_audit, args.log_root.resolve(), args.index.resolve(), args.workspace.resolve())
    except (RecordError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
