#!/usr/bin/env python3
"""Finalize the fresh 6_mistral_d rerun through repaired-v4 and v6.1.

The finalizer is intentionally single-shot.  It validates the completed
generation root, replaces only 6_mistral_d in the frozen 34-case diagnostic
view, applies the existing lossless content-repair pipeline, and runs the
CPU-only v6.1 evaluator with an explicit four-thread contract.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
CASE_ID = "6_mistral_d"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ROUTE = "adapter-converter"
CONVERTER_MODEL = "gpt-5.5"
REASONING_EFFORT = "medium"
EXPECTED_RUNS = tuple(range(1, 11))
EXPECTED_SEEDS = tuple(range(2026073101, 2026073111))
EXPECTED_ROWS = 85
ADAPTER_CONFIG_SHA256 = (
    "2bc001098ecf7b9d6209de40a112a545d9d7f02c5b9b8c2d7116cbbeb8d0672a"
)
ADAPTER_WEIGHTS_SHA256 = (
    "2d413ac81bba543e3a2cd83dcaf072ff7f7fb0485ad08eb6f0a01102f9c7377c"
)
UNIFIED_PROTOCOL = "experiment6-binding-materialization-v2-unified34"
THREAD_ENVIRONMENTS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class FinalizationError(RuntimeError):
    """Raised when a provenance or completion invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(
                isinstance(value, dict),
                f"expected JSON object: {path}:{line_number}",
            )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return resolved.name


def sanitize_text(value: str) -> str:
    return value.replace(str(WORKSPACE_ROOT), "$FQAN_ROOT")


def command_for_audit(command: Sequence[str]) -> list[str]:
    return [sanitize_text(str(value)) for value in command]


def validate_hashed_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files") or {}
    hashes = manifest.get("hashes") or {}
    for name in (
        "predictions",
        "rawResponse",
        "prompts",
        "runtime",
        "formatReport",
        "converterRawResponses",
        "retrieverCandidates",
        "stage1Raw",
    ):
        path_value = files.get(name)
        expected = hashes.get(name)
        require(path_value and expected, f"manifest lacks {name}")
        path = Path(str(path_value))
        if not path.is_absolute():
            path = root / path
        require(path.is_file(), f"artifact missing: {path}")
        require(sha256_file(path) == expected, f"artifact SHA mismatch: {path}")


def validate_generation(root: Path) -> dict[str, Any]:
    preflight = read_json(root / "preflight.json")
    matrix = preflight.get("matrix") or {}
    require(preflight.get("status") == "passed", "generation preflight failed")
    require(matrix.get("selectedFormalPredictions") == 850, "selected count is not 850")
    selected = matrix.get("selectedCases") or []
    require(
        len(selected) == 1
        and selected[0].get("output_id") == CASE_ID
        and selected[0].get("route") == ROUTE,
        "preflight selected case/route mismatch",
    )
    require(matrix.get("runs") == list(EXPECTED_RUNS), "preflight runs mismatch")
    require(
        (preflight.get("input") or {}).get("rows") == EXPECTED_ROWS,
        "preflight row count mismatch",
    )
    require(
        (preflight.get("tokens") or {})
        .get("families", {})
        .get("mistral", {})
        .get("truncationAllowed")
        is False,
        "truncation policy mismatch",
    )

    manifest_paths = sorted((root / "manifests").glob(f"{CASE_ID}__run_*.json"))
    require(len(manifest_paths) == len(EXPECTED_RUNS), "expected 10 run manifests")
    fingerprints: set[str] = set()
    total_rows = 0
    runtime_seconds = 0.0
    for expected_run, expected_seed, path in zip(
        EXPECTED_RUNS, EXPECTED_SEEDS, manifest_paths
    ):
        manifest = read_json(path)
        require(manifest.get("outputId") == CASE_ID, "case mismatch")
        require(manifest.get("run") == expected_run, "run order mismatch")
        require(manifest.get("seed") == expected_seed, "seed mismatch")
        require(
            manifest.get("status") in {"completed", "completed_with_format_errors"},
            f"incomplete generation manifest: {path}",
        )
        require(manifest.get("declaredRoute") == ROUTE, "declared route mismatch")
        require(manifest.get("effectiveRoute") == ROUTE, "effective route mismatch")
        require(manifest.get("actualModel") == MODEL_ID, "model identity mismatch")
        require(manifest.get("converterModel") == CONVERTER_MODEL, "converter mismatch")
        require(
            manifest.get("reasoningEffort") == REASONING_EFFORT,
            "reasoning effort mismatch",
        )
        require(manifest.get("runtimeBlockedRows") == 0, "runtime-blocked rows present")
        require(manifest.get("expectedRows") == EXPECTED_ROWS, "expectedRows mismatch")
        resolved = manifest.get("resolvedSource") or {}
        adapter = resolved.get("adapter") or {}
        require(
            (adapter.get("config") or {}).get("sha256") == ADAPTER_CONFIG_SHA256,
            "adapter config SHA mismatch",
        )
        require(
            (adapter.get("weights") or {}).get("sha256") == ADAPTER_WEIGHTS_SHA256,
            "adapter weights SHA mismatch",
        )
        validate_hashed_files(root, manifest)
        predictions_path = Path(str((manifest.get("files") or {})["predictions"]))
        predictions = read_jsonl(predictions_path)
        require(len(predictions) == EXPECTED_ROWS, "prediction row count mismatch")
        sources = [str(row.get("Source") or row.get("source") or "") for row in predictions]
        require(len(set(sources)) == EXPECTED_ROWS and "" not in sources, "Source coverage mismatch")
        total_rows += len(predictions)
        runtime_seconds += float(manifest.get("runtimeSeconds") or 0.0)
        fingerprint = str(manifest.get("compatibilityFingerprint") or "")
        require(bool(fingerprint), "missing compatibility fingerprint")
        fingerprints.add(fingerprint)

    require(len(fingerprints) == 1, "multiple compatibility fingerprints")
    require(total_rows == 850, "aggregate generation row count mismatch")
    protocol = str(read_json(manifest_paths[0]).get("protocol") or "")
    require(bool(protocol), "missing generation protocol")
    return {
        "status": "passed",
        "protocol": protocol,
        "compatibilityFingerprint": next(iter(fingerprints)),
        "runs": len(manifest_paths),
        "rows": total_rows,
        "runtimeSeconds": runtime_seconds,
        "goldAccessedDuringGeneration": False,
    }


def build_unified_config(
    base: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    groups = config.get("sourceGroups") or []
    corrected = next(group for group in groups if group.get("name") == "corrected12")
    historical = next(group for group in groups if group.get("name") == "historical22")
    historical["name"] = "historical21"
    historical["caseIds"] = [
        case_id for case_id in historical["caseIds"] if case_id != CASE_ID
    ]
    require(len(historical["caseIds"]) == 21, "historical case split mismatch")
    fresh = {
        "name": "fresh_mistral_d",
        "sourceProtocol": generation["protocol"],
        "sourceCompatibilityFingerprint": generation["compatibilityFingerprint"],
        # Adapter-converter generation records strict predictions and preserves
        # malformed converter output for the downstream relaxed/repaired stages.
        # Unlike corrected12 direct-binding output, it does not produce a
        # row-for-row nonformal repair sidecar at generation time.
        "requireRepairCoverage": False,
        "caseIds": [CASE_ID],
    }
    config["sourceGroups"] = [corrected, historical, fresh]
    require(
        sum(len(group["caseIds"]) for group in config["sourceGroups"]) == 34,
        "unified case count mismatch",
    )
    return config


def run_checked(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    command_log: list[dict[str, Any]],
) -> None:
    started = utc_now()
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=dict(environment),
        check=False,
        text=True,
        capture_output=True,
    )
    command_log.append(
        {
            "command": command_for_audit(command),
            "startedAt": started,
            "finishedAt": utc_now(),
            "returnCode": result.returncode,
            "stdout": sanitize_text(result.stdout[-4000:]),
            "stderr": sanitize_text(result.stderr[-4000:]),
        }
    )
    if result.returncode != 0:
        raise FinalizationError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr[-2000:]}"
        )


def run_or_validate_stage(
    output_root: Path,
    build_command: Sequence[str],
    validate_command: Sequence[str],
    *,
    environment: Mapping[str, str],
    command_log: list[dict[str, Any]],
) -> None:
    if not output_root.exists():
        run_checked(build_command, environment=environment, command_log=command_log)
    run_checked(validate_command, environment=environment, command_log=command_log)


def write_inventory(root: Path) -> Path:
    inventory = root / "sha256_inventory.tsv"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != inventory:
            rows.append(
                f"{sha256_file(path)}\t{path.stat().st_size}\t{path.relative_to(root)}"
            )
    inventory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return inventory


def write_completion_log(
    summary: Mapping[str, Any],
    markdown: str,
) -> tuple[Path, Path]:
    created_at = str(summary["createdAt"])
    stamp = created_at.replace("-", "").replace(":", "")
    log_root = WORKSPACE_ROOT / "docs" / "log"
    report_path = (
        log_root / f"{stamp}_experiment6_mistral_d_dynamic_finetuned_complete.md"
    )
    audit_path = (
        log_root / f"{stamp}_experiment6_mistral_d_dynamic_finetuned_complete.json"
    )
    report_path.write_text(markdown, encoding="utf-8")
    write_json(audit_path, summary)
    index_path = log_root / "index.json"
    lock_path = log_root / "index.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index = read_json(index_path)
        entries = index.get("entries")
        require(isinstance(entries, list), "docs/log index entries missing")
        report_logical = logical_path(report_path)
        require(
            not any(entry.get("report") == report_logical for entry in entries),
            "completion log already indexed",
        )
        fresh = summary["fresh"]
        entries.append(
            {
                "time": created_at,
                "repo": "$FQAN_ROOT",
                "kind": "experiment6_mistral_d_dynamic_finetuned_v61",
                "status": "complete_diagnostic_scoped_34_case_sensitivity",
                "summary": (
                    "Fresh 6_mistral_d replaced only its historical 34-case "
                    "entry after repaired-v4 content recovery and CPU-only "
                    f"v6.1 evaluation; Precision/Recall/F1="
                    f"{fresh['precision']}/{fresh['recall']}/{fresh['f1']}."
                ),
                "report": report_logical,
                "audit": logical_path(audit_path),
                "sha256": sha256_file(report_path),
                "auditSha256": sha256_file(audit_path),
                "bytes": report_path.stat().st_size,
                "tags": [
                    "experiment_6",
                    "mistral-7b",
                    "fine-tuned",
                    "dynamic-shot",
                    "gpt-5.5-medium",
                    "repaired-v4",
                    "v6.1.0",
                    "cpu-4-threads",
                    "diagnostic-only",
                    "claim-ineligible",
                ],
            }
        )
        write_json(index_path, index)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return report_path, audit_path


def metric_mean(case: Mapping[str, Any], name: str) -> float | None:
    value = ((case.get("aggregate") or {}).get("macro") or {}).get(name) or {}
    mean = value.get("mean")
    return None if mean is None else float(mean)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--corrected12-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.threads == 4, "evaluation thread contract is fixed at 4")

    generation_root = args.generation_root.resolve()
    corrected_root = args.corrected12_root.resolve()
    historical_root = args.historical_root.resolve()
    baseline_path = args.baseline_evaluation.resolve()
    for path in (generation_root, corrected_root, historical_root):
        require(path.is_dir(), f"source root missing: {path}")
    require(baseline_path.is_file(), f"baseline report missing: {baseline_path}")

    generation = validate_generation(generation_root)
    base_config_path = REPO_ROOT / "config" / "experiment6_binding_unified34_v2.json"
    unified_config = build_unified_config(read_json(base_config_path), generation)
    evaluation_state = generation_root / "evaluation"
    evaluation_state.mkdir(parents=True, exist_ok=True)
    unified_config_path = evaluation_state / "fresh_mistral_d_unified34_v2.json"
    write_json(unified_config_path, unified_config)

    output_parent = REPO_ROOT / "Experiment"
    unified_root = output_parent / f"experiment_6_mistral_d_dynamic_finetuned_unified34_v2_{args.stamp}"
    relaxed_root = output_parent / f"experiment_6_mistral_d_dynamic_finetuned_relaxed_v3_{args.stamp}"
    repaired_root = output_parent / f"experiment_6_mistral_d_dynamic_finetuned_repaired_v4_{args.stamp}"
    evaluation_root = output_parent / f"experiment_6_mistral_d_dynamic_finetuned_evaluation_v6_1_0_{args.stamp}"

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in THREAD_ENVIRONMENTS:
        environment[name] = str(args.threads)
    command_log: list[dict[str, Any]] = []
    python = sys.executable

    run_or_validate_stage(
        unified_root,
        [
            python,
            "-B",
            "dist/materialize_experiment6_bindings_v2.py",
            "--source",
            f"corrected12={corrected_root}",
            "--source",
            f"historical21={historical_root}",
            "--source",
            f"fresh_mistral_d={generation_root}",
            "--config",
            str(unified_config_path),
            "--output-root",
            str(unified_root),
        ],
        [
            python,
            "-B",
            "dist/validate_experiment6_bindings_v2.py",
            "--root",
            str(unified_root),
        ],
        environment=environment,
        command_log=command_log,
    )
    run_or_validate_stage(
        relaxed_root,
        [
            python,
            "-B",
            "dist/materialize_experiment6_bindings_relaxed_v3.py",
            "materialize",
            "--source-root",
            str(unified_root),
            "--output-root",
            str(relaxed_root),
        ],
        [
            python,
            "-B",
            "dist/materialize_experiment6_bindings_relaxed_v3.py",
            "validate",
            "--root",
            str(relaxed_root),
        ],
        environment=environment,
        command_log=command_log,
    )
    run_or_validate_stage(
        repaired_root,
        [
            python,
            "-B",
            "dist/materialize_experiment6_bindings_repaired_v4.py",
            "materialize",
            "--source-root",
            str(relaxed_root),
            "--output-root",
            str(repaired_root),
        ],
        [
            python,
            "-B",
            "dist/materialize_experiment6_bindings_repaired_v4.py",
            "validate",
            "--root",
            str(repaired_root),
        ],
        environment=environment,
        command_log=command_log,
    )

    report_path = evaluation_root / "evaluation_report.json"
    if not report_path.is_file():
        run_checked(
            [
                python,
                "-B",
                "dist/evaluate_experiment6_binding_candidates_v1.py",
                "--version",
                "v6.1.0",
                "--scope",
                "candidate34",
                "--candidate-root",
                str(repaired_root),
                "--evaluation-root",
                str(evaluation_root),
                "--config",
                str(REPO_ROOT / "config" / "experiment6_binding_repaired_v4_evaluation.json"),
            ],
            environment=environment,
            command_log=command_log,
        )
    require(report_path.is_file(), "evaluation report missing")
    run_checked(
        [
            python,
            "-B",
            "dist/build_experiment6_binding_candidate_score_tables.py",
            "--evaluation-report",
            str(report_path),
            "--evaluation-root",
            str(evaluation_root),
            "--source-registry",
            str(REPO_ROOT / "config" / "experiment6_source_registry.json"),
            "--output-dir",
            str(evaluation_root),
        ],
        environment=environment,
        command_log=command_log,
    )

    report = read_json(report_path)
    fresh_case = next(
        (case for case in report.get("cases", []) if case.get("outputId") == CASE_ID),
        None,
    )
    require(fresh_case is not None, "fresh case absent from evaluation")
    baseline = read_json(baseline_path)
    baseline_case = next(
        (case for case in baseline.get("cases", []) if case.get("outputId") == CASE_ID),
        None,
    )
    require(baseline_case is not None, "baseline case absent")
    summary = {
        "schemaVersion": 1,
        "protocol": "experiment6-mistral-d-fresh-v6.1-summary-v1",
        "status": "complete",
        "createdAt": utc_now(),
        "diagnosticOnly": True,
        "claimEligible": False,
        "caseId": CASE_ID,
        "generation": generation,
        "runtime": {
            "threads": args.threads,
            "cudaVisibleDevices": "",
            "chatMockUsedDuringEvaluation": False,
        },
        "old": {
            "precision": metric_mean(baseline_case, "precision"),
            "recall": metric_mean(baseline_case, "recall"),
            "f1": metric_mean(baseline_case, "f1"),
        },
        "fresh": {
            "precision": metric_mean(fresh_case, "precision"),
            "recall": metric_mean(fresh_case, "recall"),
            "f1": metric_mean(fresh_case, "f1"),
            "fields": (fresh_case.get("aggregate") or {}).get("fields"),
            "coverage": [item.get("coverage") for item in fresh_case.get("runResults", [])],
        },
        "outputs": {
            "generationRoot": logical_path(generation_root),
            "unifiedRoot": logical_path(unified_root),
            "relaxedRoot": logical_path(relaxed_root),
            "repairedRoot": logical_path(repaired_root),
            "evaluationRoot": logical_path(evaluation_root),
        },
        "commands": command_log,
    }
    summary_path = evaluation_root / "6_mistral_d_evaluation_summary.json"
    bindings_source = repaired_root / "bindings.jsonl"
    bindings_target = evaluation_root / "bindings.jsonl"
    require(bindings_source.is_file(), "repaired bindings missing")
    if not bindings_target.exists():
        os.link(bindings_source, bindings_target)
    require(
        sha256_file(bindings_source) == sha256_file(bindings_target),
        "bindings link content mismatch",
    )

    def display(value: float | None) -> str:
        return "NA" if value is None else f"{value:.6f}"

    markdown = [
        "## Material Passport",
        "",
        "- Schema: ARS Material Passport 9",
        "- Verification status: VERIFIED",
        f"- Completed: {summary['createdAt']}",
        "- Text judge: disabled",
        "- Evaluation runtime: CPU-only, 4 threads",
        "",
        "# Experiment 6: fresh 6_mistral_d repaired-v4 / v6.1",
        "",
        "> DIAGNOSTIC ONLY — OFFICIAL=false — CLAIM-ELIGIBLE=false",
        "",
        "## Result",
        "",
        "| Version | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
        (
            f"| old repaired34 | {display(summary['old']['precision'])} | "
            f"{display(summary['old']['recall'])} | {display(summary['old']['f1'])} |"
        ),
        (
            f"| fresh 6_mistral_d | {display(summary['fresh']['precision'])} | "
            f"{display(summary['fresh']['recall'])} | {display(summary['fresh']['f1'])} |"
        ),
        "",
        "The merged 34-case view replaces only 6_mistral_d; all other cases retain frozen provenance.",
        "",
    ]
    summary["outputs"]["inventory"] = logical_path(
        evaluation_root / "sha256_inventory.tsv"
    )
    markdown_text = "\n".join(markdown)
    completion_stamp = str(summary["createdAt"]).replace("-", "").replace(":", "")
    summary["outputs"]["logReport"] = (
        f"docs/log/{completion_stamp}_experiment6_mistral_d_dynamic_finetuned_complete.md"
    )
    summary["outputs"]["logAudit"] = (
        f"docs/log/{completion_stamp}_experiment6_mistral_d_dynamic_finetuned_complete.json"
    )
    write_json(summary_path, summary)
    (evaluation_root / "6_mistral_d_evaluation_summary.md").write_text(
        markdown_text, encoding="utf-8"
    )
    inventory = write_inventory(evaluation_root)
    require(inventory.is_file(), "evaluation inventory missing")
    write_completion_log(summary, markdown_text)
    print(
        json.dumps(
            {
                "status": "complete",
                "caseId": CASE_ID,
                "precision": summary["fresh"]["precision"],
                "recall": summary["fresh"]["recall"],
                "f1": summary["fresh"]["f1"],
                "evaluationRoot": logical_path(evaluation_root),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "blocked", "error": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
