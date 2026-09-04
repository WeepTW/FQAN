#!/usr/bin/env python3
"""Record validated Experiment 6 fixed-v2 phase completion in indexed docs/log."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
LOG_ROOT = WORKSPACE_ROOT / "src" / "log"
INDEX_PATH = LOG_ROOT / "index.json"


class ProtocolError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validated_phase(
    report: Mapping[str, Any], phase: str
) -> tuple[str, str, list[str]]:
    if report.get("protocol") != "narrative2-fixed-python-v2":
        raise ProtocolError("unexpected evaluation protocol")
    if phase == "part1":
        required = {
            "status": "development_partial_no_ranking",
            "completedOfficialCases": 9,
            "evaluatedCaseRuns": 90,
            "formalPredictions": 7650,
            "rankingPublished": False,
        }
        status = "part1_completed_fixed_v2_evaluated_no_ranking"
        summary = (
            "Experiment 6 Part 1 completed 9 retriever cases, 90 independent "
            "case-runs, and 7,650 formal predictions; fixed-v2 evaluation "
            "completed and the full-matrix ranking remains withheld."
        )
        tags = ["part1", "9_cases", "90_runs", "7650_predictions", "no_ranking"]
    else:
        required = {
            "status": "completed",
            "completedOfficialCases": 54,
            "evaluatedCaseRuns": 540,
            "formalPredictions": 45900,
            "rankingPublished": True,
        }
        status = "completed_54_cases_fixed_v2_ranking_published"
        summary = (
            "Experiment 6 completed all 54 official cases, 540 independent "
            "case-runs, and 45,900 formal predictions; fixed-v2 evaluation "
            "passed its completion gate and published the full ranking."
        )
        tags = ["full_matrix", "54_cases", "540_runs", "45900_predictions"]
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ProtocolError(
                f"{phase} report mismatch for {key}: "
                f"{report.get(key)!r} != {expected!r}"
            )
    if report.get("blockers"):
        raise ProtocolError(f"{phase} report contains blockers")
    return status, summary, tags


def record(output_root: Path, phase: str) -> dict[str, Any]:
    output_root = output_root.resolve()
    report_path = output_root / "evaluation_fixed_v2" / "evaluation_report.json"
    if not report_path.is_file():
        raise ProtocolError(f"missing evaluation report: {report_path}")
    report = read_json(report_path)
    status, summary, phase_tags = validated_phase(report, phase)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    kind = f"experiment6_narrative2_fixed_v2_{phase}"
    relative_output = str(output_root.relative_to(REPO_ROOT))
    record_value = {
        "time": now.isoformat().replace("+00:00", "Z"),
        "repo": str(REPO_ROOT),
        "kind": kind,
        "status": status,
        "summary": summary,
        "output_root": relative_output,
        "evaluation_report": str(report_path.relative_to(WORKSPACE_ROOT)),
        "evaluation_report_sha256": sha256_file(report_path),
        "coverage": {
            "official_cases": report["completedOfficialCases"],
            "case_runs": report["evaluatedCaseRuns"],
            "formal_predictions": report["formalPredictions"],
        },
        "ranking_published": report["rankingPublished"],
        "tags": ["experiment_6", "narrative2", "fixed_v2", *phase_tags],
    }
    log_path = LOG_ROOT / f"{timestamp}_{kind}.json"
    lock_path = LOG_ROOT / ".index.lock"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index = read_json(INDEX_PATH)
        entries = index.get("entries")
        if not isinstance(entries, list):
            raise ProtocolError("indexed docs/log/index.json entries is not a list")
        duplicate = next(
            (
                item for item in entries
                if item.get("kind") == kind
                and item.get("output_root") == relative_output
                and item.get("evaluation_report_sha256")
                == record_value["evaluation_report_sha256"]
            ),
            None,
        )
        if duplicate is not None:
            return duplicate
        write_json_atomic(log_path, record_value)
        index_entry = {
            key: record_value[key]
            for key in ("time", "repo", "kind", "status", "summary")
        }
        index_entry.update({
            "path": str(log_path.relative_to(WORKSPACE_ROOT)),
            "output_root": relative_output,
            "evaluation_report_sha256": record_value[
                "evaluation_report_sha256"
            ],
            "tags": record_value["tags"],
        })
        entries.append(index_entry)
        write_json_atomic(INDEX_PATH, index)
    return record_value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("part1", "full"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        value = record(**vars(parse_args(argv)))
    except (OSError, ValueError, KeyError, ProtocolError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
