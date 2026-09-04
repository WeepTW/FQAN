#!/usr/bin/env python3
"""Write a corrected Experiment 7 FQAN status report.

This is a report-only helper for runs whose original tmux report loop was
created before the current Experiment 7 naming/status contract. It does not run
models, clear outputs, or stop tmux panes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--expt-id", required=True)
    parser.add_argument("--write-main", action="store_true", help="Also update score_report.json/.md.")
    parser.add_argument("--write-log", action="store_true", help="Append/update a indexed docs/log index entry.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_score_report(repo_root: Path, expt_id: str, engine: str) -> Path | None:
    for path in sorted((repo_root / "Experiment").glob(f"{expt_id}_*/generator/score_report.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        engines = {str(item.get("engine")) for item in payload.get("items", []) if isinstance(item, dict)}
        if engine in engines:
            return path
    return None


def read_rc(orch: Path, name: str, aliases: list[str] | None = None) -> int | None:
    for candidate in [name, *(aliases or [])]:
        path = orch / f"{candidate}.rc"
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw) if raw.isdigit() else None
    return None


def main() -> None:
    args = parse_args()
    repo = args.repo_root
    workspace = args.workspace_root
    expt = repo / "Experiment" / args.expt_id
    orch = expt / "fqan_tmux_run"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rc_status = {
        "preflight": read_rc(orch, "preflight"),
        "chatmock_service": read_rc(orch, "chatmock_service"),
        "qwen_vllm": read_rc(orch, "qwen_vllm"),
        "gpt55": read_rc(orch, "gpt55"),
        "gptCodexS": read_rc(orch, "gptCodexS"),
        "gpt41_gate": read_rc(orch, "gpt41_gate"),
        "qwen": read_rc(orch, "qwen"),
        "mistral4": read_rc(orch, "mistral4"),
        "llama33": read_rc(orch, "llama33"),
    }

    engine_groups = [
        ("gpt55", "gpt5_5"),
        ("gptCodexS", "gpt5_3_codexS"),
        ("gpt4.1", "gpt4_1"),
        ("qwen", "qwen3_6"),
        ("mistral4", "mistral4"),
        ("llama", "llama3_3"),
    ]
    items: list[dict[str, Any]] = []
    for group, engine in engine_groups:
        score_report = find_score_report(repo, args.expt_id, engine)
        if score_report is None:
            continue
        payload = load_json(score_report)
        for raw in payload.get("items", []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["engine_group"] = group
            item["source_score_report"] = str(score_report)
            items.append(item)

    preflight_path = orch / "preflight_status.json"
    preflight = load_json(preflight_path) if preflight_path.is_file() else {}
    blockers = []
    for blocker_path in sorted((expt / "blockers").glob("*.json")):
        try:
            blockers.append({"path": str(blocker_path), "payload": load_json(blocker_path)})
        except Exception as exc:
            blockers.append({"path": str(blocker_path), "error": str(exc)})
    quota = []
    for path in sorted(expt.glob("**/*.quota_wait.json")):
        try:
            quota.append({"path": str(path), "payload": load_json(path)})
        except Exception as exc:
            quota.append({"path": str(path), "error": str(exc)})

    if rc_status.get("preflight") not in (None, 0):
        status = "blocked_preflight"
    elif any(value not in (None, 0) for value in rc_status.values()):
        status = "blocked_runtime"
    elif all(value == 0 for value in rc_status.values()):
        status = "completed"
    else:
        status = "running"

    status_counts = Counter(str(item.get("score_status") or item.get("route_status") or "unknown") for item in items)
    payload = {
        "time": now,
        "experiment": "7",
        "stage": "fqan_formal_score_report_corrected_current",
        "top_expt_id": args.expt_id,
        "status": status,
        "rc_status": rc_status,
        "items": items,
        "status_counts": dict(status_counts),
        "processed_input_paths": preflight.get("processed_input_paths", {}),
        "preflight_matched_count": preflight.get("matched_count"),
        "preflight_expected_matched_count": preflight.get("expected_matched_count"),
        "preflight_backfilled_input_paths": preflight.get("backfilled_input_paths", {}),
        "blocker_audits": blockers,
        "gpt_quota_checkpoints": quota,
        "llama_full_started": (orch / "answer_llama3_3.full_started").is_file(),
        "note": "Corrected current status; no model rerun, output clearing, tmux kill, or service kill was performed.",
    }

    out_json = expt / "score_report.corrected_current.json"
    out_md = expt / "score_report.corrected_current.md"
    targets = [(out_json, out_md)]
    if args.write_main:
        targets.append((expt / "score_report.json", expt / "score_report.md"))

    lines = [
        "# Experiment 7 FQAN Formal Corrected Current Status",
        f"- top_expt_id: {args.expt_id}",
        f"- time: {now}",
        f"- status: {status}",
        f"- preflight_matched_count: {payload['preflight_matched_count']}/{payload['preflight_expected_matched_count']}",
        f"- llama_full_started: {payload['llama_full_started']}",
        f"- blocker_audits: {len(blockers)}",
        f"- gpt_quota_checkpoints: {len(quota)}",
        "",
        "## RC Status",
    ]
    lines.extend(f"- {key}: {value}" for key, value in rc_status.items())
    lines.extend(["", "## Score Counts"])
    lines.extend(f"- {key}: {value}" for key, value in sorted(status_counts.items()))
    lines.extend(["", "## Blockers"])
    if blockers:
        for audit in blockers:
            audit_payload = audit.get("payload") or {}
            lines.append(f"- {audit_payload.get('engine', 'unknown')}: {audit_payload.get('reason', audit_payload.get('status', 'blocked'))}")
    else:
        lines.append("- none")

    for json_path, md_path in targets:
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_path = None
    if args.write_log:
        safe_now = now.replace("-", "").replace(":", "")
        log_path = workspace / "src" / "log" / f"{safe_now}_experiment7_fqan_corrected_current_status.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index_path = workspace / "src" / "log" / "index.json"
        try:
            index = load_json(index_path) if index_path.is_file() else {"entries": []}
        except Exception:
            index = {"entries": []}
        rel = str(log_path.relative_to(workspace))
        entry = {
            "time": now,
            "path": rel,
            "repo": str(repo),
            "kind": "experiment7_fqan_corrected_current_status",
            "status": status,
            "summary": "Corrected Experiment 7 current status using gptCodexS naming and blocked_runtime rc semantics.",
            "tags": ["experiment_7", "finqa", "ea", "fqan", "gptCodexS", "blocked_runtime"],
        }
        index["entries"] = [item for item in index.setdefault("entries", []) if item.get("path") != rel] + [entry]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "items": len(items), "out_json": str(out_json), "log_path": str(log_path) if log_path else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
