#!/usr/bin/env python3
"""Canonical start/resume/evaluate interface for Experiment 6 hybrid-v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment6_paths import PATHS


REPO_ROOT = PATHS.repo
WORKSPACE_ROOT = PATHS.workspace
DIST_ROOT = PATHS.dist
EXPERIMENT_ROOT = REPO_ROOT / "Experiment"
DEFAULT_CONFIG = REPO_ROOT / "config" / "experiment6_narrative2_hybrid_v4.json"
CURRENT_POINTER = EXPERIMENT_ROOT / "experiment_6_v4_current.json"


class ProtocolError(RuntimeError):
    """Raised when a root cannot safely start or resume."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def workspace_path(raw: str | Mapping[str, Any]) -> Path:
    if isinstance(raw, Mapping):
        return PATHS.resolve_locator(raw)
    path = Path(raw)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schemaVersion") != 4:
        raise ProtocolError("Experiment 6 v4 config must have schemaVersion=4")
    if int(value.get("expectedOfficialCases", -1)) != 38:
        raise ProtocolError("Experiment 6 v4 must contain exactly 38 official cases")
    if int(value.get("expectedFormalPredictions", -1)) != 32300:
        raise ProtocolError("Experiment 6 v4 must expect exactly 32,300 predictions")
    return value


def effective_generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base_path = workspace_path(config["baseGenerationConfig"])
    value = read_json(base_path)
    excluded = set(str(item) for item in config["excludedSourceIds"])
    parts = []
    for part in value["parts"]:
        current = dict(part)
        if int(part["part"]) == 2:
            current["models"] = [
                dict(model)
                for model in part["models"]
                if str(model["sourceId"]) not in excluded
            ]
        elif int(part["part"]) == 3:
            current["cases"] = [
                dict(case)
                for case in part["cases"]
                if str(case["sourceId"]) not in excluded
            ]
        parts.append(current)
    value["schemaVersion"] = 4
    value["protocol"] = "experiment6-narrative2-generation-v4-38case"
    value["parts"] = parts
    value["controls"] = []
    value["expectedOfficialCases"] = 38
    value["expectedDiagnosticCases"] = 0
    value["expectedPartCounts"] = {"1": 9, "2": 12, "3": 13, "4": 4}
    value["sourceWorkbook"] = dict(config["sourceWorkbook"])
    value["v4Contract"] = {
        "excludedSourceIds": sorted(excluded),
        "expectedFormalPredictions": 32300,
        "evaluationProtocol": "narrative2-hybrid-v4",
        "bindingAlignment": "fixed-index-no-reorder",
        "failurePolicy": "rejected-zero",
    }
    retriever = dict(value.get("retriever") or {})
    retriever["retry"] = dict(config["retrieverRetry"])
    value["retriever"] = retriever
    direct = dict(value.get("directBinding") or {})
    tokenizers = dict(direct.get("tokenizers") or {})
    for source_id in excluded:
        tokenizers.pop(source_id, None)
    direct_token_preflight = config["directTokenPreflight"]
    tokenizer_roles = dict(direct.get("tokenizerRoles") or {})
    for source_id in direct_token_preflight["sourceIds"]:
        tokenizers[str(source_id)] = str(
            direct_token_preflight["proxyTokenizer"]
        )
        tokenizer_roles[str(source_id)] = str(direct_token_preflight["role"])
    direct["tokenizers"] = tokenizers
    direct["tokenizerRoles"] = tokenizer_roles
    value["directBinding"] = direct
    routes = dict(value.get("runtimeRoutes") or {})
    for source_id in excluded:
        routes.pop(source_id, None)
    value["runtimeRoutes"] = routes

    part_counts = {}
    official = 0
    for part in parts:
        if "models" in part:
            count = len(part["models"]) * len(part["promptModes"])
        else:
            count = len(part["cases"])
        part_counts[str(part["part"])] = count
        official += count
    if part_counts != {"1": 9, "2": 12, "3": 13, "4": 4} or official != 38:
        raise ProtocolError(f"effective matrix mismatch: {part_counts}, total={official}")
    if len(value["runs"]) != 10 or int(value["expectedRows"]) != 85:
        raise ProtocolError("effective matrix must use ten runs and 85 rows")
    return value


def contract(
    config_path: Path,
    config: Mapping[str, Any],
    generation: Mapping[str, Any],
    *,
    require_external_runtime: bool = True,
) -> dict[str, Any]:
    tracked_paths = {
        "overlayConfig": config_path,
        "baseGenerationConfig": workspace_path(config["baseGenerationConfig"]),
        "sourceRegistry": workspace_path(generation["sourceRegistry"]),
        "baseEvaluationConfig": workspace_path(config["baseEvaluationConfig"]),
        "generationRunner": workspace_path(config["generationRunner"]),
        "evaluationRunner": workspace_path(config["evaluationRunner"]),
        "orchestrator": Path(__file__).resolve(),
        "judgeBuilder": DIST_ROOT / "build_experiment6_judge_examples_v4.py",
        "legacyGenerationRunner": DIST_ROOT / "run_experiment6_binding_generation.py",
        "generatorRuntime": REPO_ROOT / "new_full_finqa_run.py",
        "resultOrganization": REPO_ROOT / "result_organization.py",
        "retrieverJsonSchema": REPO_ROOT / "retriever_json_schema.py",
        "retrieverLmfe": REPO_ROOT / "retriever_lmfe.py",
        "retrieverRowCheckpoint": REPO_ROOT / "retriever_row_checkpoint.py",
        "seq2seqRetriever": REPO_ROOT / ".external/FINDER/Retriever Codes" / "seq2seq_retriever.py",
        "flanRetriever": REPO_ROOT / ".external/FINDER/Retriever Codes" / "Flan" / "lora_flan_large_finqa_rel_fact.py",
        "mistralRetriever": REPO_ROOT / ".external/FINDER/Retriever Codes" / "Mistral" / "mistral_inference.py",
        "t5gemma2Retriever": REPO_ROOT / ".external/FINDER/Retriever Codes" / "t5gemma-2" / "t5gemma-2_train.py",
        "inferenceWorkbook": workspace_path(config["inferenceWorkbook"]),
        "sourceWorkbook": workspace_path(config["sourceWorkbook"]),
        "judgeWorkbook": workspace_path(config["judgeExamples"]),
        "gold": workspace_path(config["evaluation"]["goldPath"]),
    }
    if require_external_runtime:
        tracked_paths.update({
            "chatmockLauncher": DIST_ROOT / "start_chatmock_server.sh",
            "chatmockServer": REPO_ROOT / ".external" / "ChatMock" / "chatmock.py",
            "chatmockRoutes": REPO_ROOT / ".external" / "ChatMock" / "chatmock" / "routes_openai.py",
        })
    hashes = {}
    for name, path in tracked_paths.items():
        if not path.is_file():
            raise ProtocolError(f"contract file missing: {path}")
        hashes[name] = {"path": str(path), "sha256": sha256_file(path)}
    expected_hashes = {
        "sourceRegistry": generation["sourceRegistry"]["sha256"],
        "inferenceWorkbook": config["inferenceWorkbook"]["sha256"],
        "sourceWorkbook": config["sourceWorkbook"]["sha256"],
        "judgeWorkbook": config["judgeExamples"]["sha256"],
        "gold": config["evaluation"]["goldSha256"],
    }
    for name, expected in expected_hashes.items():
        actual = hashes[name]["sha256"]
        if actual != expected:
            raise ProtocolError(f"{name} SHA mismatch: {actual} != {expected}")
    body = {
        "protocol": config["protocol"],
        "expectedRows": 85,
        "expectedRuns": 10,
        "expectedOfficialCases": 38,
        "expectedFormalPredictions": 32300,
        "excludedSourceIds": config["excludedSourceIds"],
        "effectiveGenerationConfigSha256": sha256_json(generation),
        "files": hashes,
    }
    body["compatibilityFingerprint"] = sha256_json(body)
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
    current_contract = contract(
        config_path,
        config,
        generation,
        require_external_runtime=root_role == "formal",
    )
    contract_path = root / "experiment6_v4_contract.json"
    effective_path = root / "effective_generation_config.json"
    if create:
        write_json(effective_path, generation)
        write_json(contract_path, {
            **current_contract,
            "createdAt": utc_now(),
            "rootRole": root_role,
        })
        builder = DIST_ROOT / "build_experiment6_judge_examples_v4.py"
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
            raise ProtocolError("resume root has no v4 contract/effective config")
        stored = read_json(contract_path)
        if stored.get("rootRole") != root_role:
            raise ProtocolError(
                f"root role mismatch: stored={stored.get('rootRole')!r} "
                f"requested={root_role!r}"
            )
        if stored.get("compatibilityFingerprint") != current_contract["compatibilityFingerprint"]:
            raise ProtocolError(
                "resume compatibility fingerprint mismatch; start a new root"
            )
        if sha256_json(read_json(effective_path)) != current_contract["effectiveGenerationConfigSha256"]:
            raise ProtocolError("effective generation config changed inside resume root")
    return generation, current_contract


def root_status(root: Path) -> dict[str, Any]:
    manifest_paths = sorted((root / "manifests").glob("*.json"))
    manifests = []
    for path in manifest_paths:
        try:
            manifests.append(read_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    statuses = Counter(str(item.get("status") or "unknown") for item in manifests)
    completed_statuses = {"completed", "completed_with_format_errors"}
    completed = sum(item.get("status") in completed_statuses for item in manifests)
    prediction_rows = 0
    for item in manifests:
        files = item.get("files") if isinstance(item.get("files"), dict) else {}
        prediction_path = Path(str(files.get("predictions") or ""))
        if prediction_path.is_file():
            with prediction_path.open(encoding="utf-8") as handle:
                prediction_rows += sum(bool(line.strip()) for line in handle)
    return {
        "time": utc_now(),
        "protocol": "experiment6-narrative2-hybrid-v4",
        "outputRoot": str(root),
        "expectedCaseRuns": 380,
        "reportedCaseRuns": len(manifests),
        "completedCaseRuns": completed,
        "statusCounts": dict(statuses),
        "formalPredictionRows": prediction_rows,
        "expectedFormalPredictions": 32300,
        "generationComplete": (
            len(manifests) == 380
            and completed == 380
            and prediction_rows == 32300
        ),
        "evaluationComplete": (root / "experiment6_results.json").is_file(),
    }


def run_checked(command: list[str]) -> None:
    print(json.dumps({"time": utc_now(), "command": command}, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        raise ProtocolError(
            f"command failed with return code {completed.returncode}: {command}"
        )


def run_preflight_and_smoke(
    root: Path, config_path: Path, effective_path: Path
) -> dict[str, Any]:
    runner = DIST_ROOT / "run_experiment6_narrative2_generation.py"
    run_checked([
        sys.executable,
        "-B",
        str(runner),
        "--config",
        str(effective_path),
        "--output-root",
        str(root),
        "--preflight-only",
    ])
    smoke_root = root / "smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_command = [
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
    print(json.dumps({"time": utc_now(), "command": smoke_command}, ensure_ascii=False), flush=True)
    completed = subprocess.run(smoke_command, cwd=REPO_ROOT, check=False)
    manifests = [
        read_json(path) for path in sorted((smoke_root / "manifests").glob("*.json"))
    ]
    if len(manifests) != 14 or len({item.get("sourceId") for item in manifests}) != 14:
        raise ProtocolError(
            "smoke route coverage mismatch: "
            f"manifests={len(manifests)} uniqueSources="
            f"{len({item.get('sourceId') for item in manifests})}"
        )
    passed = sorted(
        str(item["sourceId"]) for item in manifests if item.get("status") == "completed"
    )
    blocked = [
        {
            "outputId": item.get("outputId"),
            "sourceId": item.get("sourceId"),
            "status": item.get("status"),
            "error": item.get("error"),
            "formatComplianceRate": item.get("formatComplianceRate"),
        }
        for item in manifests
        if item.get("status") != "completed"
    ]
    if completed.returncode not in {0, 2}:
        raise ProtocolError(
            f"smoke command failed with unexpected return code {completed.returncode}"
        )
    status = {
        "time": utc_now(),
        "status": "passed" if not blocked else "partial_runtime_blocked",
        "separateFromFormalRuns": True,
        "configSha256": sha256_file(effective_path),
        "expectedRoutes": 14,
        "passedRoutes": len(passed),
        "passedSourceIds": passed,
        "blockedRoutes": blocked,
    }
    write_json(smoke_root / "smoke_status.json", status)
    return status


def run_generation(
    root: Path,
    effective_path: Path,
    source_ids: Sequence[str] | None = None,
    execution_groups: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    runner = DIST_ROOT / "run_experiment6_narrative2_generation.py"
    base_command = [
        sys.executable,
        "-B",
        str(runner),
        "--config",
        str(effective_path),
        "--output-root",
        str(root),
    ]
    selected = set(source_ids or ())
    groups: list[tuple[str, list[str]]] = []
    covered: set[str] = set()
    for raw_group in execution_groups or ():
        name = str(raw_group["name"])
        group_sources = [
            str(source_id)
            for source_id in raw_group["sourceIds"]
            if str(source_id) in selected
        ]
        overlap = covered & set(group_sources)
        if overlap:
            raise ProtocolError(
                f"execution source groups overlap: {sorted(overlap)}"
            )
        covered.update(group_sources)
        if group_sources:
            groups.append((name, group_sources))
    if covered != selected:
        raise ProtocolError(
            "execution source groups do not cover selected routes: "
            f"missing={sorted(selected - covered)} extra={sorted(covered - selected)}"
        )
    if not groups:
        raise ProtocolError("no executable source group passed smoke")

    runtime_dir = root / "runtime" / "generation_workers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    workers: list[tuple[str, list[str], list[str], Any, Any]] = []
    worker_report: dict[str, Any] = {
        "time": utc_now(),
        "status": "running",
        "policy": "disjoint-source-groups-with-shared-gpu-locks",
        "workers": {},
    }
    for name, group_sources in groups:
        command = list(base_command)
        for source_id in group_sources:
            command.extend(["--source-id", source_id])
        log_path = runtime_dir / f"{name}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(json.dumps({
            "time": utc_now(),
            "event": "generation_worker_started",
            "worker": name,
            "sourceIds": group_sources,
            "command": command,
            "log": str(log_path),
        }, ensure_ascii=False), flush=True)
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((name, group_sources, command, process, log_handle))
        worker_report["workers"][name] = {
            "sourceIds": group_sources,
            "pid": process.pid,
            "log": str(log_path),
            "status": "running",
        }
    write_json(runtime_dir / "workers.json", worker_report)

    failed: list[dict[str, Any]] = []
    for name, group_sources, command, process, log_handle in workers:
        return_code = process.wait()
        log_handle.close()
        worker_report["workers"][name]["returnCode"] = return_code
        worker_report["workers"][name]["status"] = (
            "completed" if return_code == 0 else "blocked"
        )
        if return_code:
            failed.append({
                "worker": name,
                "sourceIds": group_sources,
                "returnCode": return_code,
                "command": command,
            })
        worker_report["time"] = utc_now()
        worker_report["status"] = "blocked" if failed else "running"
        write_json(runtime_dir / "workers.json", worker_report)
    worker_report["time"] = utc_now()
    worker_report["status"] = "blocked" if failed else "completed"
    worker_report["failedWorkers"] = failed
    write_json(runtime_dir / "workers.json", worker_report)
    if failed:
        raise ProtocolError(f"generation workers failed: {failed}")


def run_evaluation(root: Path, config_path: Path, *, report_only: bool) -> None:
    evaluator = DIST_ROOT / "evaluate_narrative2_hybrid_v4.py"
    command = [
        sys.executable,
        "-B",
        str(evaluator),
        "--config",
        str(config_path),
        "--output-root",
        str(root),
    ]
    if report_only:
        command.append("--report-only")
    run_checked(command)


def latest_incomplete(config: Mapping[str, Any]) -> Path | None:
    prefix = str(config["rootPrefix"])
    formal_name = re.compile(rf"{re.escape(prefix)}\d{{8}}T\d{{6}}Z")
    for path in sorted(EXPERIMENT_ROOT.glob(prefix + "*"), reverse=True):
        contract_path = path / "experiment6_v4_contract.json"
        if not (
            path.is_dir()
            and formal_name.fullmatch(path.name)
            and contract_path.is_file()
        ):
            continue
        try:
            stored = read_json(contract_path)
        except (OSError, json.JSONDecodeError):
            continue
        if stored.get("rootRole") == "formal" and not root_status(path)["generationComplete"]:
            return path
    return None


def write_current(root: Path, status: str) -> None:
    write_json(CURRENT_POINTER, {
        "time": utc_now(),
        "protocol": "experiment6-narrative2-hybrid-v4",
        "outputRoot": str(root),
        "status": status,
    })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "auto", "start", "resume", "status", "evaluate", "report",
            "preflight", "public-preflight", "smoke",
        ],
    )
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    try:
        action = args.action
        if action == "start":
            root = (
                args.root.resolve()
                if args.root
                else EXPERIMENT_ROOT / f"{config['rootPrefix']}{utc_stamp()}"
            )
            prepare_root(root, config_path, config, create=True, root_role="formal")
            write_current(root, "preflight")
            smoke_status = run_preflight_and_smoke(
                root, config_path, root / "effective_generation_config.json"
            )
            write_current(root, "running")
            run_generation(
                root,
                root / "effective_generation_config.json",
                smoke_status["passedSourceIds"],
                config["executionGroups"],
            )
            if smoke_status["blockedRoutes"]:
                write_current(root, "runtime_blocked_waiting_for_routes")
                print(json.dumps({
                    "time": utc_now(),
                    "status": "partial_generation_no_ranking",
                    "outputRoot": str(root),
                    "blockedRoutes": smoke_status["blockedRoutes"],
                    "progress": root_status(root),
                }, ensure_ascii=False, indent=2))
                return 2
            write_current(root, "evaluating")
            run_evaluation(root, config_path, report_only=False)
            write_current(root, "completed")
        elif action == "resume":
            if not args.root:
                raise ProtocolError("resume requires an explicit output root")
            root = args.root.resolve()
            prepare_root(root, config_path, config, create=False, root_role="formal")
            smoke_status = run_preflight_and_smoke(
                root, config_path, root / "effective_generation_config.json"
            )
            write_current(root, "running")
            run_generation(
                root,
                root / "effective_generation_config.json",
                smoke_status["passedSourceIds"],
                config["executionGroups"],
            )
            if smoke_status["blockedRoutes"]:
                write_current(root, "runtime_blocked_waiting_for_routes")
                print(json.dumps({
                    "time": utc_now(),
                    "status": "partial_generation_no_ranking",
                    "outputRoot": str(root),
                    "blockedRoutes": smoke_status["blockedRoutes"],
                    "progress": root_status(root),
                }, ensure_ascii=False, indent=2))
                return 2
            write_current(root, "evaluating")
            run_evaluation(root, config_path, report_only=False)
            write_current(root, "completed")
        elif action == "auto":
            root = args.root.resolve() if args.root else latest_incomplete(config)
            if root is None:
                return main(["start", "--config", str(config_path)])
            return main(["resume", str(root), "--config", str(config_path)])
        elif action in {"status", "evaluate", "report"}:
            root = args.root.resolve() if args.root else None
            if root is None and CURRENT_POINTER.is_file():
                root = Path(str(read_json(CURRENT_POINTER)["outputRoot"]))
            if root is None:
                raise ProtocolError(f"{action} requires an output root")
            if action == "status":
                print(json.dumps(root_status(root), ensure_ascii=False, indent=2))
                return 0
            prepare_root(root, config_path, config, create=False, root_role="formal")
            run_evaluation(root, config_path, report_only=action == "report")
        elif action in {"preflight", "public-preflight", "smoke"}:
            root = args.root.resolve() if args.root else Path("/tmp") / f"experiment6_v4_{action}_{utc_stamp()}"
            generation, _ = prepare_root(
                root,
                config_path,
                config,
                create=True,
                root_role="diagnostic",
            )
            if action == "public-preflight":
                print(json.dumps({
                    "time": utc_now(),
                    "status": "public_preflight_passed",
                    "outputRoot": str(root),
                    "officialCases": generation["expectedOfficialCases"],
                    "runs": len(generation["runs"]),
                    "rows": generation["expectedRows"],
                    "runtimeArtifactsChecked": False,
                }, ensure_ascii=False, indent=2))
                return 0
            runner = DIST_ROOT / "run_experiment6_narrative2_generation.py"
            command = [
                sys.executable,
                "-B",
                str(runner),
                "--config",
                str(root / "effective_generation_config.json"),
                "--output-root",
                str(root),
            ]
            command.append(
                "--preflight-only" if action == "preflight" else "--smoke-only"
            )
            if action == "smoke":
                command.append("--no-resume")
            run_checked(command)
        else:
            raise AssertionError(action)
        root_value = locals().get("root")
        if isinstance(root_value, Path):
            print(json.dumps(root_status(root_value), ensure_ascii=False, indent=2))
        return 0
    except ProtocolError as error:
        print(json.dumps({
            "time": utc_now(),
            "protocol": config["protocol"],
            "status": "blocked",
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
