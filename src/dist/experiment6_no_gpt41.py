#!/usr/bin/env python3
"""Canonical 34-case start/resume/evaluate interface for Experiment 6."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_narrative2_hybrid_v4_no_gpt41 as evaluator
import experiment6_no_gpt41_finalize as finalizer
import experiment6_v4 as legacy


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
EXPERIMENT_ROOT = REPO_ROOT / "Experiment"
DEFAULT_CONFIG = (
    REPO_ROOT / "config" / "experiment6_narrative2_hybrid_v4_no_gpt41.json"
)
CURRENT_POINTER = EXPERIMENT_ROOT / "experiment_6_no_gpt41_current.json"
CONTRACT_NAME = "experiment6_no_gpt41_contract.json"
COMPLETED_STATUSES = {"completed", "completed_with_format_errors"}


class ProtocolError(RuntimeError):
    """Raised when a root cannot safely start, resume, or evaluate."""


def load_config(path: Path) -> dict[str, Any]:
    value = legacy.read_json(path)
    if not isinstance(value, dict) or value.get("schemaVersion") != 4:
        raise ProtocolError("Experiment 6 config must have schemaVersion=4")
    if int(value.get("expectedOfficialCases", -1)) != 34:
        raise ProtocolError("Experiment 6 must contain exactly 34 official cases")
    if int(value.get("expectedFormalPredictions", -1)) != 28900:
        raise ProtocolError("Experiment 6 must expect exactly 28,900 predictions")
    if "gpt4_1" not in {str(item) for item in value.get("excludedSourceIds", [])}:
        raise ProtocolError("Experiment 6 must explicitly exclude gpt4_1")
    return value


def effective_generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base_path = legacy.workspace_path(config["baseGenerationConfig"])
    value = legacy.read_json(base_path)
    excluded = {str(item) for item in config["excludedSourceIds"]}
    parts: list[dict[str, Any]] = []
    for raw_part in value["parts"]:
        part = dict(raw_part)
        if "models" in part:
            part["models"] = [
                dict(model)
                for model in part["models"]
                if str(model["sourceId"]) not in excluded
            ]
        if "cases" in part:
            part["cases"] = [
                dict(case)
                for case in part["cases"]
                if str(case["sourceId"]) not in excluded
            ]
        count = (
            len(part.get("models", [])) * len(part.get("promptModes", []))
            if "models" in part
            else len(part.get("cases", []))
        )
        if count:
            parts.append(part)
    value["schemaVersion"] = 4
    # The active generation runner currently recognizes this v4 transport
    # protocol; the exact 34-case scope is carried by the frozen parts and
    # v4Contract below rather than inferred from this legacy label.
    value["protocol"] = "experiment6-narrative2-generation-v4-38case"
    value["parts"] = parts
    value["controls"] = []
    value["expectedOfficialCases"] = int(config["expectedOfficialCases"])
    value["expectedDiagnosticCases"] = 0
    value["expectedPartCounts"] = dict(config["expectedPartCounts"])
    value["sourceWorkbook"] = dict(config["sourceWorkbook"])
    value["v4Contract"] = {
        "excludedSourceIds": sorted(excluded),
        "expectedFormalPredictions": int(config["expectedFormalPredictions"]),
        "evaluationProtocol": evaluator.PROTOCOL,
        "bindingAlignment": "fixed-index-no-reorder",
        "failurePolicy": "rejected-zero",
    }
    retriever = dict(value.get("retriever") or {})
    retriever["retry"] = dict(config["retrieverRetry"])
    value["retriever"] = retriever
    direct = dict(value.get("directBinding") or {})
    tokenizers = dict(direct.get("tokenizers") or {})
    tokenizer_roles = dict(direct.get("tokenizerRoles") or {})
    for source_id in excluded:
        tokenizers.pop(source_id, None)
        tokenizer_roles.pop(source_id, None)
    preflight = config["directTokenPreflight"]
    for source_id in preflight["sourceIds"]:
        tokenizers[str(source_id)] = str(preflight["proxyTokenizer"])
        tokenizer_roles[str(source_id)] = str(preflight["role"])
    direct["tokenizers"] = tokenizers
    direct["tokenizerRoles"] = tokenizer_roles
    value["directBinding"] = direct
    value["runtimeRoutes"] = {
        source_id: route
        for source_id, route in value.get("runtimeRoutes", {}).items()
        if str(source_id) not in excluded
    }
    part_counts: dict[str, int] = {}
    official = 0
    for part in parts:
        count = (
            len(part["models"]) * len(part["promptModes"])
            if "models" in part
            else len(part["cases"])
        )
        part_counts[str(part["part"])] = count
        official += count
    if part_counts != dict(config["expectedPartCounts"]) or official != 34:
        raise ProtocolError(f"effective matrix mismatch: {part_counts}, total={official}")
    if len(value["runs"]) != 10 or int(value["expectedRows"]) != 85:
        raise ProtocolError("effective matrix must use ten runs and 85 rows")
    outputs = evaluator.expected_output_sources(value, excluded)
    if len(outputs) != 34 or "gpt4_1" in set(outputs.values()):
        raise ProtocolError("effective output matrix is not exact no-GPT-4.1 scope")
    return value


def build_contract(
    config_path: Path,
    config: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    source = legacy.contract(config_path, config, generation)
    files = dict(source["files"])
    tracked = {
        "orchestrator": Path(__file__).resolve(),
        "finalizer": Path(finalizer.__file__).resolve(),
    }
    for name, path in tracked.items():
        files[name] = {"path": str(path), "sha256": legacy.sha256_file(path)}
    body = {
        "protocol": str(config["protocol"]),
        "expectedRows": int(config["expectedRows"]),
        "expectedRuns": int(config["expectedRuns"]),
        "expectedOfficialCases": int(config["expectedOfficialCases"]),
        "expectedFormalPredictions": int(config["expectedFormalPredictions"]),
        "excludedSourceIds": list(config["excludedSourceIds"]),
        "effectiveGenerationConfigSha256": legacy.sha256_json(generation),
        "files": files,
    }
    body["compatibilityFingerprint"] = legacy.sha256_json(body)
    return body


def prepare_root(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    *,
    create: bool,
    root_role: str = "formal",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if create:
        if root.exists():
            raise ProtocolError(f"new output root already exists: {root}")
        root.mkdir(parents=True)
    elif not root.is_dir():
        raise ProtocolError(f"output root missing: {root}")
    generation = effective_generation_config(config)
    current_contract = build_contract(config_path, config, generation)
    contract_path = root / CONTRACT_NAME
    effective_path = root / "effective_generation_config.json"
    if create:
        legacy.write_json(effective_path, generation)
        legacy.write_json(contract_path, {
            **current_contract,
            "createdAt": legacy.utc_now(),
            "rootRole": root_role,
        })
        builder = Path(__file__).resolve().parent / "build_experiment6_judge_examples_v4.py"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(builder),
                "--config",
                str(config_path),
                "--output-dir",
                str(root / "judge_examples"),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
        if completed.returncode:
            raise ProtocolError("judge example bundle build failed")
    else:
        if not contract_path.is_file() or not effective_path.is_file():
            raise ProtocolError("resume root has no no-GPT-4.1 contract/effective config")
        stored = legacy.read_json(contract_path)
        if stored.get("rootRole") != root_role:
            raise ProtocolError("root role mismatch")
        if stored.get("compatibilityFingerprint") != current_contract["compatibilityFingerprint"]:
            raise ProtocolError("resume compatibility fingerprint mismatch; start a new root")
        if legacy.sha256_json(legacy.read_json(effective_path)) != current_contract["effectiveGenerationConfigSha256"]:
            raise ProtocolError("effective generation config changed inside resume root")
    return generation, current_contract


def root_status(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_paths = sorted((root / "manifests").glob("*.json"))
    manifests: list[dict[str, Any]] = []
    for path in manifest_paths:
        try:
            manifests.append(legacy.read_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    statuses = Counter(str(item.get("status") or "unknown") for item in manifests)
    completed = sum(item.get("status") in COMPLETED_STATUSES for item in manifests)
    prediction_rows = 0
    for item in manifests:
        files = item.get("files") if isinstance(item.get("files"), dict) else {}
        path = Path(str(files.get("predictions") or ""))
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                prediction_rows += sum(bool(line.strip()) for line in handle)
    expected_runs = int(config["expectedOfficialCases"]) * int(config["expectedRuns"])
    expected_predictions = int(config["expectedFormalPredictions"])
    return {
        "time": legacy.utc_now(),
        "protocol": evaluator.PROTOCOL,
        "outputRoot": str(root),
        "expectedCaseRuns": expected_runs,
        "reportedCaseRuns": len(manifests),
        "completedCaseRuns": completed,
        "statusCounts": dict(statuses),
        "formalPredictionRows": prediction_rows,
        "expectedFormalPredictions": expected_predictions,
        "generationComplete": (
            len(manifests) == expected_runs
            and completed == expected_runs
            and prediction_rows == expected_predictions
        ),
        "evaluationComplete": (root / "experiment6_results.json").is_file(),
    }


def expected_route_ids(generation: Mapping[str, Any]) -> set[str]:
    return {
        str(item["sourceId"])
        for part in generation["parts"]
        for item in part.get("models", []) + part.get("cases", [])
    }


def run_preflight_and_smoke(root: Path, effective_path: Path) -> dict[str, Any]:
    runner = Path(__file__).resolve().parent / "run_experiment6_narrative2_generation.py"
    legacy.run_checked([
        sys.executable,
        "-B",
        str(runner),
        "--config",
        str(effective_path),
        "--output-root",
        str(root),
        "--preflight-only",
    ])
    generation = legacy.read_json(effective_path)
    routes = expected_route_ids(generation)
    smoke_root = root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-B",
        str(runner),
        "--config",
        str(effective_path),
        "--output-root",
        str(smoke_root),
        "--smoke-only",
        "--no-resume",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    manifests = [
        legacy.read_json(path)
        for path in sorted((smoke_root / "manifests").glob("*.json"))
    ]
    manifest_sources = {str(item.get("sourceId")) for item in manifests}
    if len(manifests) != len(routes) or manifest_sources != routes:
        raise ProtocolError("smoke route coverage mismatch")
    passed = sorted(
        str(item["sourceId"])
        for item in manifests
        if item.get("status") == "completed"
    )
    blocked = [
        {
            "outputId": item.get("outputId"),
            "sourceId": item.get("sourceId"),
            "status": item.get("status"),
            "error": item.get("error"),
        }
        for item in manifests
        if item.get("status") != "completed"
    ]
    if completed.returncode not in {0, 2}:
        raise ProtocolError(f"smoke command failed: {completed.returncode}")
    status = {
        "time": legacy.utc_now(),
        "status": "passed" if not blocked else "partial_runtime_blocked",
        "separateFromFormalRuns": True,
        "expectedRoutes": len(routes),
        "passedRoutes": len(passed),
        "passedSourceIds": passed,
        "blockedRoutes": blocked,
    }
    legacy.write_json(smoke_root / "smoke_status.json", status)
    return status


def run_finalizer(root: Path, config_path: Path) -> None:
    legacy.run_checked([
        sys.executable,
        "-B",
        str(Path(finalizer.__file__).resolve()),
        "--once",
        "--output-root",
        str(root),
        "--config",
        str(config_path),
        "--evaluation-attempts",
        "3",
    ])


def run_report(root: Path, config_path: Path) -> None:
    legacy.run_checked([
        sys.executable,
        "-B",
        str(Path(evaluator.__file__).resolve()),
        "--config",
        str(config_path),
        "--output-root",
        str(root),
        "--report-only",
    ])


def latest_incomplete(config: Mapping[str, Any]) -> Path | None:
    prefix = str(config["rootPrefix"])
    formal_name = re.compile(rf"{re.escape(prefix)}\d{{8}}T\d{{6}}Z")
    for path in sorted(EXPERIMENT_ROOT.glob(prefix + "*"), reverse=True):
        contract_path = path / CONTRACT_NAME
        if not (
            path.is_dir()
            and formal_name.fullmatch(path.name)
            and contract_path.is_file()
        ):
            continue
        try:
            stored = legacy.read_json(contract_path)
        except (OSError, json.JSONDecodeError):
            continue
        if stored.get("rootRole") == "formal" and not root_status(path, config)["generationComplete"]:
            return path
    return None


def write_current(root: Path, status: str) -> None:
    legacy.write_json(CURRENT_POINTER, {
        "time": legacy.utc_now(),
        "protocol": evaluator.PROTOCOL,
        "outputRoot": str(root),
        "status": status,
    })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["auto", "start", "resume", "status", "evaluate", "report", "preflight", "smoke"],
    )
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    try:
        config = load_config(config_path)
        action = args.action
        if action in {"start", "resume"}:
            if action == "start":
                root = (
                    args.root.resolve()
                    if args.root
                    else EXPERIMENT_ROOT / f"{config['rootPrefix']}{legacy.utc_stamp()}"
                )
                prepare_root(root, config_path, config, create=True)
            else:
                if not args.root:
                    raise ProtocolError("resume requires an explicit output root")
                root = args.root.resolve()
                prepare_root(root, config_path, config, create=False)
            write_current(root, "preflight")
            smoke = run_preflight_and_smoke(root, root / "effective_generation_config.json")
            write_current(root, "running")
            legacy.run_generation(
                root,
                root / "effective_generation_config.json",
                smoke["passedSourceIds"],
                config["executionGroups"],
            )
            if smoke["blockedRoutes"]:
                write_current(root, "runtime_blocked_waiting_for_routes")
                return 2
            write_current(root, "evaluating")
            run_finalizer(root, config_path)
            write_current(root, "completed")
        elif action == "auto":
            root = args.root.resolve() if args.root else latest_incomplete(config)
            if root is None:
                return main(["start", "--config", str(config_path)])
            return main(["resume", str(root), "--config", str(config_path)])
        elif action in {"status", "evaluate", "report"}:
            root = args.root.resolve() if args.root else None
            if root is None and CURRENT_POINTER.is_file():
                root = Path(str(legacy.read_json(CURRENT_POINTER)["outputRoot"]))
            if root is None:
                raise ProtocolError(f"{action} requires an output root")
            if action == "status":
                print(json.dumps(root_status(root, config), ensure_ascii=False, indent=2))
                return 0
            prepare_root(root, config_path, config, create=False)
            if action == "evaluate":
                run_finalizer(root, config_path)
            else:
                run_report(root, config_path)
        elif action in {"preflight", "smoke"}:
            root = (
                args.root.resolve()
                if args.root
                else Path("/tmp") / f"experiment6_no_gpt41_{action}_{legacy.utc_stamp()}"
            )
            prepare_root(root, config_path, config, create=True, root_role="diagnostic")
            runner = Path(__file__).resolve().parent / "run_experiment6_narrative2_generation.py"
            command = [
                sys.executable,
                "-B",
                str(runner),
                "--config",
                str(root / "effective_generation_config.json"),
                "--output-root",
                str(root),
                "--preflight-only" if action == "preflight" else "--smoke-only",
            ]
            if action == "smoke":
                command.append("--no-resume")
            legacy.run_checked(command)
        else:
            raise AssertionError(action)
        print(json.dumps(root_status(root, config), ensure_ascii=False, indent=2))
        return 0
    except (ProtocolError, legacy.ProtocolError, finalizer.FinalizerError) as error:
        print(json.dumps({
            "time": legacy.utc_now(),
            "protocol": "experiment6-narrative2-hybrid-v4-no-gpt41",
            "status": "blocked",
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
