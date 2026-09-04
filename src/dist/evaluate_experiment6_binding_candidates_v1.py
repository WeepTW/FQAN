#!/usr/bin/env python3
"""Evaluate materialized Experiment 6 Binding candidates diagnostically.

This entry point never changes or relabels source manifests.  It accepts only
the frozen, claim-ineligible Binding-candidate protocol and reuses the field
scoring primitives of reference-aligned v6.0.1, v6.0.2, or v6.1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import evaluate_narrative2_reference_aligned_v6 as v601
import evaluate_narrative2_reference_aligned_v6_0_2 as v602
import evaluate_narrative2_reference_aligned_v6_1 as v610
from validate_experiment6_binding_candidates import validate_output as validate_candidate_v1
from validate_experiment6_bindings_v2 import validate_output as validate_candidate_v2
from materialize_experiment6_bindings_relaxed_v3 import validate_output as validate_candidate_v3
from materialize_experiment6_bindings_repaired_v4 import validate_output as validate_candidate_v4


REPO_ROOT = DIST.parent
WORKSPACE_ROOT = REPO_ROOT.parent
PROTOCOL = "experiment6-binding-candidate-evaluation-v1"
SCORERS = {"v6.0.1": v601, "v6.0.2": v602, "v6.1.0": v610}
DEFAULT_SCORING_CONFIGS = {
    "v6.1.0": "config/experiment6_narrative2_evaluation_v6_1.json",
}
EXPECTED_CANDIDATE_PROTOCOL = "experiment6-binding-candidate-materialization-v1"
UNIFIED_CANDIDATE_PROTOCOL = "experiment6-binding-materialization-v2-unified34"
RELAXED_CANDIDATE_PROTOCOL = "experiment6-binding-materialization-relaxed-v3-unified34"
REPAIRED_CANDIDATE_PROTOCOL = "experiment6-binding-materialization-repaired-v4-unified34"
MISTRAL_CHAT_PROJECTION_PROTOCOL = "experiment6-mistral-chat-repaired-projection-v1"
SUPPORTED_CANDIDATE_PROTOCOLS = {
    EXPECTED_CANDIDATE_PROTOCOL,
    UNIFIED_CANDIDATE_PROTOCOL,
    RELAXED_CANDIDATE_PROTOCOL,
    REPAIRED_CANDIDATE_PROTOCOL,
    MISTRAL_CHAT_PROJECTION_PROTOCOL,
}
THREAD_ENVIRONMENTS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class CandidateEvaluationError(RuntimeError):
    """Raised when a diagnostic evaluation invariant is violated."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateEvaluationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CandidateEvaluationError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return resolved.name


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateEvaluationError(message)


def validate_runtime() -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    thread_values = {name: os.environ.get(name) for name in THREAD_ENVIRONMENTS}
    require(all(value == "4" for value in thread_values.values()), "all CPU thread limits must equal 4")
    return {
        "condaEnvironment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cudaVisibleDevices": "",
        "threadLimits": thread_values,
        "textJudge": "disabled",
        "chatMockUsed": False,
    }


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error": type(error).__name__, "processes": []}
    if result.returncode != 0:
        return {"status": "unavailable", "error": result.stderr.strip(), "processes": []}
    processes = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        processes.append({"pid": int(parts[0]), "processName": Path(parts[1]).name, "usedMemoryMiB": int(parts[2])})
    return {"status": "available", "processes": sorted(processes, key=lambda item: (item["pid"], item["usedMemoryMiB"]))}


def assert_no_new_gpu_process(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before.get("status") != "available" or after.get("status") != "available":
        return
    before_pids = {int(item["pid"]) for item in before.get("processes", [])}
    new_processes = [item for item in after.get("processes", []) if int(item["pid"]) not in before_pids]
    require(not new_processes, f"new GPU process appeared during CPU-only evaluation: {new_processes}")


def validate_wrapper_config(config: Mapping[str, Any]) -> None:
    require(config.get("protocolId") == PROTOCOL, "candidate evaluation protocol mismatch")
    require(config.get("acceptedCandidateProtocol") in SUPPORTED_CANDIDATE_PROTOCOLS, "candidate input protocol mismatch")
    require(config.get("claimEligible") is False, "candidate evaluation must remain claim-ineligible")
    require(config.get("selectionRole") == "diagnostic-descriptive-only", "selection role mismatch")
    require(config.get("textJudge") == "disabled", "Text judge must be disabled")
    configured = set(config.get("scoringVersions") or {})
    require({"v6.0.1", "v6.0.2"} <= configured, "scoring-version map mismatch")
    require(configured <= set(SCORERS), "unknown scoring version")


def validate_candidate_manifest(
    manifest: Mapping[str, Any],
    *,
    accepted_protocol: str = EXPECTED_CANDIDATE_PROTOCOL,
) -> None:
    require(accepted_protocol in SUPPORTED_CANDIDATE_PROTOCOLS, "unsupported candidate protocol")
    require(manifest.get("protocol") == accepted_protocol, "candidate run protocol mismatch")
    require(manifest.get("official") is False, "candidate manifest cannot be official")
    require(manifest.get("diagnosticOnly") is True, "candidate manifest must be diagnostic-only")
    require(manifest.get("claimEligible") is False, "candidate manifest cannot be claim-eligible")
    require(manifest.get("goldAccessed") is False, "candidate materialization accessed gold")
    expected_status = (
        "completed"
        if accepted_protocol == MISTRAL_CHAT_PROJECTION_PROTOCOL
        else "completed_diagnostic_binding_candidates"
    )
    require(manifest.get("status") == expected_status, "candidate run status mismatch")


def validate_mistral_chat_projection(candidate_root: Path) -> dict[str, Any]:
    dataset = read_json(candidate_root / "dataset_manifest.json")
    require(dataset.get("protocol") == MISTRAL_CHAT_PROJECTION_PROTOCOL, "Mistral projection protocol mismatch")
    require(dataset.get("status") == "complete", "Mistral projection is incomplete")
    require(dataset.get("official") is False, "Mistral projection cannot be official")
    require(dataset.get("diagnosticOnly") is True, "Mistral projection must be diagnostic-only")
    require(dataset.get("claimEligible") is False, "Mistral projection cannot be claim-eligible")
    require(dataset.get("goldAccessed") is False, "Mistral projection accessed gold")
    counts = dataset.get("counts") or {}
    require(int(counts.get("cases", -1)) == 2, "Mistral projection case count mismatch")
    require(int(counts.get("caseRuns", -1)) == 20, "Mistral projection case-run count mismatch")
    require(int(counts.get("rows", -1)) == 1700, "Mistral projection row count mismatch")

    expected_cases = {"6_mistral_base_m", "6_mistral_base_d"}
    expected_runs = set(range(1, 11))
    grouped: dict[str, set[int]] = defaultdict(set)
    manifest_paths = sorted((candidate_root / "manifests").glob("*.json"))
    require(len(manifest_paths) == 20, "Mistral projection manifest count mismatch")
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        validate_candidate_manifest(manifest, accepted_protocol=MISTRAL_CHAT_PROJECTION_PROTOCOL)
        output_id = str(manifest.get("outputId") or "")
        run = int(manifest.get("run") or 0)
        grouped[output_id].add(run)
        prediction_path = resolve_prediction_path(v610, manifest, candidate_root)
        require(prediction_path.is_file(), f"Mistral projection prediction missing: {prediction_path}")
    require(set(grouped) == expected_cases, "Mistral projection case IDs mismatch")
    require(all(runs == expected_runs for runs in grouped.values()), "Mistral projection run coverage mismatch")

    rows = read_jsonl(candidate_root / "rows.jsonl")
    require(len(rows) == 1700, "Mistral projection rows.jsonl count mismatch")
    pairs = Counter((str(row.get("outputId")), int(row.get("run") or 0)) for row in rows)
    require(len(pairs) == 20 and all(value == 85 for value in pairs.values()), "Mistral projection row grouping mismatch")
    require((candidate_root / "sha256_inventory.tsv").is_file(), "Mistral projection inventory missing")
    return {
        "status": "passed",
        "protocol": MISTRAL_CHAT_PROJECTION_PROTOCOL,
        "cases": 2,
        "caseRuns": 20,
        "rows": 1700,
    }


def validate_candidate_root(candidate_root: Path, accepted_protocol: str) -> dict[str, Any]:
    if accepted_protocol == EXPECTED_CANDIDATE_PROTOCOL:
        return validate_candidate_v1(candidate_root)
    if accepted_protocol == UNIFIED_CANDIDATE_PROTOCOL:
        return validate_candidate_v2(candidate_root)
    if accepted_protocol == RELAXED_CANDIDATE_PROTOCOL:
        return validate_candidate_v3(candidate_root)
    if accepted_protocol == REPAIRED_CANDIDATE_PROTOCOL:
        return validate_candidate_v4(candidate_root)
    if accepted_protocol == MISTRAL_CHAT_PROJECTION_PROTOCOL:
        return validate_mistral_chat_projection(candidate_root)
    raise CandidateEvaluationError(f"unsupported candidate protocol: {accepted_protocol}")


def resolve_source_manifest(candidate_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    if candidate_manifest.get("protocol") == MISTRAL_CHAT_PROJECTION_PROTOCOL:
        path = Path(str(candidate_manifest.get("sourceGenerationManifest") or ""))
        expected_sha = candidate_manifest.get("sourceGenerationManifestSha256")
    else:
        source = candidate_manifest.get("source") or {}
        path = Path(str(source.get("manifest") or ""))
        expected_sha = source.get("manifestSha256")
    require(path.is_file(), f"candidate source manifest missing: {path}")
    require(sha256_file(path) == expected_sha, f"candidate source manifest SHA mismatch: {path}")
    return read_json(path), path


def resolve_prediction_path(scorer: Any, manifest: Mapping[str, Any], root: Path) -> Path:
    path = scorer.resolve_artifact(manifest, "predictions", root)
    require(path.is_file(), f"prediction file missing: {path}")
    require(sha256_file(path) == (manifest.get("hashes") or {}).get("predictions"), f"prediction SHA mismatch: {path}")
    return path


def load_entries(
    candidate_root: Path,
    base_root: Path | None,
    scope: str,
    scorer: Any,
    wrapper_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_protocol = str(wrapper_config["acceptedCandidateProtocol"])
    validation = validate_candidate_root(candidate_root, accepted_protocol)
    dataset = read_json(candidate_root / "dataset_manifest.json")
    validate_candidate_manifest({
        "protocol": dataset.get("protocol"),
        "official": dataset.get("official"),
        "diagnosticOnly": dataset.get("diagnosticOnly"),
        "claimEligible": dataset.get("claimEligible"),
        "goldAccessed": dataset.get("goldAccessed"),
        "status": "completed" if accepted_protocol == MISTRAL_CHAT_PROJECTION_PROTOCOL else "completed_diagnostic_binding_candidates",
    }, accepted_protocol=accepted_protocol)
    candidate_entries: list[dict[str, Any]] = []
    candidate_origin = (
        "candidate34"
        if scope == "candidate34"
        else "flan-long-context"
        if scope == "flan-long-context"
        else "mistral-base-md"
        if scope == "mistral-base-md"
        else "candidate12"
    )
    for manifest_path in sorted((candidate_root / "manifests").glob("*.json")):
        manifest = read_json(manifest_path)
        validate_candidate_manifest(manifest, accepted_protocol=accepted_protocol)
        source_manifest, source_manifest_path = resolve_source_manifest(manifest)
        candidate_entries.append({
            "origin": candidate_origin,
            "root": candidate_root,
            "manifest": manifest,
            "manifestPath": manifest_path,
            "sourceManifest": source_manifest,
            "sourceManifestPath": source_manifest_path,
        })
    candidate_ids = {str(entry["manifest"]["outputId"]) for entry in candidate_entries}
    require(len(candidate_ids) == int(wrapper_config["expectedCandidateCases"]), "candidate case count mismatch")

    entries = list(candidate_entries)
    if scope == "candidate-merged34":
        require(base_root is not None, "candidate-merged34 requires --base-root")
        base_manifest_paths = sorted((base_root / "manifests").glob("*.json"))
        base_entries = []
        all_base_ids = set()
        for manifest_path in base_manifest_paths:
            manifest = read_json(manifest_path)
            if manifest.get("official") is not True:
                continue
            output_id = str(manifest["outputId"])
            all_base_ids.add(output_id)
            if output_id in candidate_ids:
                continue
            base_entries.append({
                "origin": "historical22",
                "root": base_root,
                "manifest": manifest,
                "manifestPath": manifest_path,
                "sourceManifest": manifest,
                "sourceManifestPath": manifest_path,
            })
        require(len(all_base_ids) == int(wrapper_config["expectedMergedCases"]), "base root is not the frozen 34-case scope")
        entries.extend(base_entries)

    expected_cases = int(
        wrapper_config["expectedCandidateCases"]
        if scope in {"candidate12", "candidate34", "flan-long-context", "mistral-base-md"}
        else wrapper_config["expectedMergedCases"]
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    pairs = set()
    for entry in entries:
        manifest = entry["manifest"]
        pair = (str(manifest["outputId"]), int(manifest["run"]))
        require(pair not in pairs, f"duplicate case-run: {pair}")
        pairs.add(pair)
        grouped[pair[0]].append(pair[1])
    require(len(grouped) == expected_cases, f"scope case count mismatch: expected {expected_cases}, got {len(grouped)}")
    expected_runs = list(range(1, int(wrapper_config["expectedRuns"]) + 1))
    incomplete = {output_id: sorted(runs) for output_id, runs in grouped.items() if sorted(runs) != expected_runs}
    require(not incomplete, f"scope has incomplete runs: {incomplete}")
    require(len(entries) == expected_cases * len(expected_runs), "scope manifest count mismatch")

    for entry in entries:
        resolve_prediction_path(scorer, entry["manifest"], entry["root"])
    validation["datasetManifestSha256"] = sha256_file(candidate_root / "dataset_manifest.json")
    validation["inventorySha256"] = sha256_file(candidate_root / "sha256_inventory.tsv")
    return entries, validation


def entry_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    manifest = entry["manifest"]
    source_manifest = entry["sourceManifest"]
    return {
        "outputId": str(manifest["outputId"]),
        "origin": str(entry["origin"]),
        "sourceId": str(source_manifest.get("sourceId") or ""),
        "promptMode": str(source_manifest.get("promptMode") or ""),
        "inputType": str(source_manifest.get("inputType") or ""),
        "finishedAt": str(source_manifest.get("finishedAt") or ""),
        "route": str(source_manifest.get("route") or manifest.get("route") or ""),
    }


def collect_input_state(entries: Sequence[Mapping[str, Any]], scorer: Any) -> dict[str, Any]:
    artifacts = []
    for entry in sorted(entries, key=lambda item: (str(item["manifest"]["outputId"]), int(item["manifest"]["run"]))):
        manifest = entry["manifest"]
        manifest_path = entry["manifestPath"]
        prediction_path = resolve_prediction_path(scorer, manifest, entry["root"])
        artifacts.append({
            "outputId": str(manifest["outputId"]),
            "run": int(manifest["run"]),
            "origin": str(entry["origin"]),
            "manifestPath": logical_path(manifest_path),
            "manifestSha256": sha256_file(manifest_path),
            "predictionPath": logical_path(prediction_path),
            "predictionSha256": sha256_file(prediction_path),
        })
    hash_material = [
        {key: item[key] for key in ("outputId", "run", "origin", "manifestSha256", "predictionSha256")}
        for item in artifacts
    ]
    return {"artifacts": artifacts, "inputSetSha256": stable_sha256(hash_material)}


def aggregate_candidate_case(scorer: Any, output_id: str, runs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    case = scorer.aggregate_case(output_id, runs, "formal", config)
    case["mode"] = "diagnostic"
    case["claimEligible"] = False
    case["selection"]["role"] = "diagnostic-descriptive-only"
    return case


def mean_defined(values: Sequence[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else None


def sum_metric_counts(run_results: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter()
    for run in run_results:
        metric = run[key]
        counts.update({name: int(metric[name]) for name in ("tp", "fp", "fn")})
    return dict(counts)


def sum_primary_counts(scorer: Any, run_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = scorer.zero_counts()
    for run in run_results:
        scorer.add_counts(counts, run["primary"]["counts"])
    return scorer.metrics_from_counts(counts)


def sum_without_trend_counts(scorer: Any, run_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = [field for field in scorer.PRIMARY_FIELDS if field != "Trend"]
    counts = scorer.zero_counts(fields)
    for run in run_results:
        scorer.add_counts(counts, run["withoutTrendAblation"]["counts"])
    return scorer.metrics_from_counts(counts)


def aggregate_method_audit(run_results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    audits = [run.get("methodAudit") for run in run_results if run.get("methodAudit")]
    if not audits:
        return None
    counts = Counter()
    unmatched_gold = 0
    unmatched_prediction = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        counts.update({key: int(value) for key, value in (audit.get("counts") or {}).items()})
        unmatched_gold += int(audit.get("unmatchedGoldBindings", 0))
        unmatched_prediction += int(audit.get("unmatchedPredictionBindings", 0))
        for category, values in (audit.get("examples") or {}).items():
            remaining = max(0, 12 - len(examples[category]))
            examples[category].extend(list(values)[:remaining])
    return {
        "counts": dict(sorted(counts.items())),
        "examples": {key: value for key, value in sorted(examples.items())},
        "unmatchedGoldBindings": unmatched_gold,
        "unmatchedPredictionBindings": unmatched_prediction,
    }


def candidate_stats(candidate_root: Path) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_jsonl(candidate_root / "rows.jsonl"):
        counter = grouped[str(row["outputId"])]
        counter["rows"] += 1
        format_valid = row.get("schemaValid")
        if format_valid is None:
            format_valid = row.get("formatValid")
        if format_valid is True:
            counter["acceptedRows"] += 1
            if int(row.get("bindingCount") or 0) > 0:
                counter["acceptedRowsWithBindings"] += 1
            else:
                counter["acceptedEmptyBindingRows"] += 1
        else:
            counter["rejectedRows"] += 1
        counter[f"status:{row.get('candidateStatus')}"] += 1
    return {output_id: dict(sorted(values.items())) for output_id, values in sorted(grouped.items())}


def score_projection(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "outputId": case["outputId"],
        "runs": [
            {
                "run": run["run"],
                "primary": run["primary"],
                "bindingLevel": run["bindingLevel"],
                "withoutTrendAblation": run["withoutTrendAblation"],
                "coverage": run["coverage"],
                "trend": run["trend"],
            }
            for run in case["runResults"]
        ],
    }


def compare_sensitivity(cases: Sequence[Mapping[str, Any]], reference_path: Path | None) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    reference = read_json(reference_path)
    require(reference.get("protocol") == "experiment6-deterministic-repair-sensitivity-v1", "sensitivity reference protocol mismatch")
    expected = {str(case["outputId"]): score_projection(case) for case in reference.get("cases", [])}
    actual = {str(case["outputId"]): score_projection(case) for case in cases}
    require(actual == expected, "materialized candidate scores do not reproduce deterministic-repair sensitivity")
    return {
        "status": "exact-score-projection-match",
        "referencePath": logical_path(reference_path),
        "referenceSha256": sha256_file(reference_path),
        "referenceMacroF1Mean": reference.get("overall", {}).get("repairSensitivityMacroF1Mean"),
    }


def compare_formal(cases: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    reference = read_json(path)
    reference_cases = {str(case["outputId"]): case for case in reference.get("cases", [])}
    require(set(reference_cases) == {str(case["outputId"]) for case in cases}, "formal reference case set mismatch")
    differences = []
    for case in cases:
        output_id = str(case["outputId"])
        current = case["aggregate"]["macro"]["f1"]["mean"]
        baseline = reference_cases[output_id]["aggregate"]["macro"]["f1"]["mean"]
        differences.append({
            "outputId": output_id,
            "formalMacroF1Mean": baseline,
            "candidateMacroF1Mean": current,
            "difference": float(current) - float(baseline) if current is not None and baseline is not None else None,
        })
    return {
        "referencePath": logical_path(path),
        "referenceSha256": sha256_file(path),
        "referenceProtocol": reference.get("protocol"),
        "caseDifferences": differences,
        "formalCaseMeanMacroF1": mean_defined([item["formalMacroF1Mean"] for item in differences]),
    }


def method_metadata(
    scorer: Any,
    scoring_config_path: Path,
    scoring_config: Mapping[str, Any],
    wrapper_config_path: Path,
    wrapper_config: Mapping[str, Any],
) -> dict[str, Any]:
    base = scorer.method_metadata(scoring_config_path, scoring_config)
    wrapper_sha = sha256_file(Path(__file__).resolve())
    material = {
        "protocolId": PROTOCOL,
        "acceptedCandidateProtocol": wrapper_config["acceptedCandidateProtocol"],
        "claimEligible": False,
        "selectionRole": wrapper_config["selectionRole"],
        "textJudge": "disabled",
        "baseMethodSha256": base["methodSha256"],
    }
    compatibility = stable_sha256(material)
    return {
        "protocolId": PROTOCOL,
        "baseScoringProtocol": scorer.PROTOCOL,
        "baseMethodSha256": base["methodSha256"],
        "baseConfigSha256": base["configSha256"],
        "wrapperConfigPath": logical_path(wrapper_config_path),
        "wrapperConfigSha256": sha256_file(wrapper_config_path),
        "evaluatorPath": logical_path(Path(__file__)),
        "evaluatorSha256": wrapper_sha,
        "methodCompatibilitySha256": compatibility,
        "methodSha256": stable_sha256({"compatibility": compatibility, "evaluator": wrapper_sha}),
        "scoreRole": "diagnostic-only",
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    runtime = validate_runtime()
    gpu_before = gpu_snapshot()
    candidate_root = args.candidate_root.resolve()
    base_root = args.base_root.resolve() if args.base_root else None
    output_root = args.evaluation_root.resolve()
    wrapper_config_path = args.config.resolve()
    wrapper_config = read_json(wrapper_config_path)
    validate_wrapper_config(wrapper_config)
    scorer = SCORERS[args.version]
    scoring_config_locator = (wrapper_config.get("scoringVersions") or {}).get(
        args.version
    ) or DEFAULT_SCORING_CONFIGS.get(args.version)
    require(scoring_config_locator is not None, f"scoring config missing: {args.version}")
    scoring_config_path = (REPO_ROOT / scoring_config_locator).resolve()
    scoring_config = read_json(scoring_config_path)
    scorer.validate_config(scoring_config)
    if hasattr(scorer, "configure"):
        scorer.configure(scoring_config)
    require(int(scoring_config["expectedRows"]) == int(wrapper_config["expectedRows"]), "row contract mismatch")
    require(int(scoring_config["expectedRuns"]) == int(wrapper_config["expectedRuns"]), "run contract mismatch")

    entries, candidate_validation_before = load_entries(
        candidate_root, base_root, args.scope, scorer, wrapper_config
    )
    input_before = collect_input_state(entries, scorer)
    gold_path = (REPO_ROOT / scoring_config["goldPath"]).resolve()
    require(gold_path.is_file(), f"gold missing: {gold_path}")
    require(sha256_file(gold_path) == scoring_config["goldSha256"], "gold SHA mismatch")
    targets = read_json(gold_path).get("rows")
    require(isinstance(targets, list) and len(targets) == int(scoring_config["expectedRows"]), "gold row count mismatch")

    objects = scorer.ObjectMatcher(scoring_config["objectName"])
    trends = scorer.TrendClassifier(scoring_config["trend"], allow_model=True)
    run_results = []
    metadata_by_case: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda item: (str(item["manifest"]["outputId"]), int(item["manifest"]["run"]))):
        manifest = entry["manifest"]
        predictions, prediction_path = scorer.load_predictions(manifest, targets, entry["root"])
        summary, records = scorer.evaluate_rows(targets, predictions, objects, trends)
        output_id = str(manifest["outputId"])
        run = int(manifest["run"])
        run_result = {
            "outputId": output_id,
            "run": run,
            "seed": manifest.get("seed"),
            "inputOrigin": entry["origin"],
            "generationStatus": manifest.get("status"),
            "predictionPath": logical_path(prediction_path),
            "predictionSha256": sha256_file(prediction_path),
            **summary,
        }
        run_results.append(run_result)
        run_root = output_root / "cases" / output_id / f"run_{run:02d}"
        write_jsonl(run_root / "records.jsonl", records)
        write_json(run_root / "metrics.json", run_result)
        metadata = entry_metadata(entry)
        previous = metadata_by_case.setdefault(output_id, metadata)
        for key in ("origin", "sourceId", "promptMode", "inputType", "route"):
            require(previous[key] == metadata[key], f"inconsistent case metadata: {output_id} {key}")
        previous["finishedAt"] = max(previous["finishedAt"], metadata["finishedAt"])

    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in run_results:
        grouped_results[str(run["outputId"])].append(run)
    cases = [
        aggregate_candidate_case(scorer, output_id, grouped_results[output_id], scoring_config)
        for output_id in sorted(grouped_results)
    ]
    for case in cases:
        case["inputOrigin"] = metadata_by_case[str(case["outputId"])]["origin"]

    sensitivity = compare_sensitivity(cases, args.sensitivity_reference.resolve() if args.sensitivity_reference else None)
    formal_comparison = (
        compare_formal(cases, args.formal_reference.resolve())
        if args.formal_reference
        else None
    )
    primary_pooled = sum_primary_counts(scorer, run_results)
    without_trend_pooled = sum_without_trend_counts(scorer, run_results)
    binding_counts = sum_metric_counts(run_results, "bindingLevel")
    binding_pooled = scorer.metric(**binding_counts)
    coverage = Counter()
    trend_support = Counter()
    trend_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for run in run_results:
        coverage.update({key: int(value) for key, value in run["coverage"].items()})
        trend_support.update({key: int(value) for key, value in run["trend"]["support"].items()})
        for gold_class, columns in run["trend"]["confusionMatrix"].items():
            trend_confusion[gold_class].update({key: int(value) for key, value in columns.items()})
    trend_summary = {
        "support": dict(sorted(trend_support.items())),
        "confusionMatrix": {
            key: dict(sorted(value.items()))
            for key, value in sorted(trend_confusion.items())
        },
    }
    if hasattr(scorer, "trend_metrics_from_confusion"):
        trend_summary.update(
            scorer.trend_metrics_from_confusion(
                trend_summary["confusionMatrix"], trend_summary["support"]
            )
        )
    method_audit = aggregate_method_audit(run_results)
    case_mean = mean_defined([case["aggregate"]["macro"]["f1"]["mean"] for case in cases])
    without_trend_case_mean = mean_defined([
        mean_defined([run["withoutTrendAblation"]["macro"]["f1"] for run in case["runResults"]])
        for case in cases
    ])
    input_after = collect_input_state(entries, scorer)
    require(input_after == input_before, "source manifests or predictions changed during evaluation")
    candidate_validation_after = validate_candidate_root(
        candidate_root, str(wrapper_config["acceptedCandidateProtocol"])
    )
    require(candidate_validation_after == {key: value for key, value in candidate_validation_before.items() if key not in {"datasetManifestSha256", "inventorySha256"}}, "candidate validation changed during evaluation")
    gpu_after = gpu_snapshot()
    assert_no_new_gpu_process(gpu_before, gpu_after)

    total_rows = len(run_results) * int(scoring_config["expectedRows"])
    gold_bindings = int(coverage.get("gold_bindings", 0))
    matched_bindings = int(coverage.get("matched_bindings", 0))
    report = {
        "schemaVersion": 1,
        "protocol": PROTOCOL,
        "scoringProtocol": scorer.PROTOCOL,
        "status": "diagnostic_only",
        "mode": "diagnostic",
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "selectionEmitted": True,
        "selectionRole": "diagnostic-descriptive-only",
        "scope": args.scope,
        "scopeComplete": True,
        "experimentMatrixComplete": False,
        "text": {"status": "deferred", "score": None, "judgeUsed": False},
        "time": scorer.utc_now(),
        "method": method_metadata(scorer, scoring_config_path, scoring_config, wrapper_config_path, wrapper_config),
        "input": {
            "candidateProtocol": wrapper_config["acceptedCandidateProtocol"],
            "candidateRoot": logical_path(candidate_root),
            "baseRoot": logical_path(base_root) if base_root else None,
            "inputSetSha256Before": input_before["inputSetSha256"],
            "inputSetSha256After": input_after["inputSetSha256"],
            "artifacts": len(input_before["artifacts"]),
        },
        "inputProvenancePath": logical_path(output_root / "input_provenance.json"),
        "candidateValidation": {**candidate_validation_before, "root": logical_path(candidate_root)},
        "candidateStatsByCase": candidate_stats(candidate_root),
        "runtime": {**runtime, "gpuBefore": gpu_before, "gpuAfter": gpu_after, "newGpuProcess": False},
        "coverage": {
            **dict(sorted(coverage.items())),
            "evaluatedRows": total_rows,
            "formatInvalidRate": coverage.get("format_invalid_rows", 0) / total_rows if total_rows else None,
            "emptyOutputRate": coverage.get("empty_output_rows", 0) / total_rows if total_rows else None,
            "anchorMatchRate": matched_bindings / gold_bindings if gold_bindings else None,
        },
        "overall": {
            "caseMeanMacroF1": case_mean,
            "withoutTrendCaseMeanMacroF1": without_trend_case_mean,
            "lowScoreThreshold": float(wrapper_config["lowScoreThreshold"]),
            "lowScoreTriggered": args.scope == "candidate12" and case_mean is not None and case_mean < float(wrapper_config["lowScoreThreshold"]),
            "primaryPooled": primary_pooled,
            "withoutTrendPooled": without_trend_pooled,
            "bindingPooled": binding_pooled,
        },
        "trend": trend_summary,
        "methodAudit": method_audit,
        "formalComparison": formal_comparison,
        "sensitivityReproduction": sensitivity,
        "caseMetadata": [metadata_by_case[output_id] for output_id in sorted(metadata_by_case)],
        "completedCases": len(cases),
        "completedCaseRuns": len(run_results),
        "missingMatrixCaseIds": wrapper_config["missingMatrixCaseIds"],
        "cases": cases,
        "limitations": [
            "Materialized candidates are diagnostic and cannot replace formal predictions or rankings.",
            "Top-1 and top-3 are descriptive shared-run selections and are claim-ineligible.",
            "Text was not judged and remains NA.",
            (
                "This root contains only the selected diagnostic scope; "
                "it is not a formal Experiment 6 matrix."
                if args.scope in {"flan-long-context", "mistral-base-md"}
                else "Four GPT-4.1 cases are absent, so the 38-case experiment matrix is incomplete."
            ),
        ],
    }
    provenance = {
        "schemaVersion": 1,
        "protocol": PROTOCOL,
        "scope": args.scope,
        "claimEligible": False,
        **input_before,
    }
    write_json(output_root / "input_provenance.json", provenance)
    report["inputProvenanceSha256"] = sha256_file(output_root / "input_provenance.json")
    write_json(output_root / "evaluation_report.json", report)
    lines = [
        f"# Experiment 6 Binding candidates: {args.version} × {args.scope}",
        "",
        "- Status: diagnostic only; official=false; claimEligible=false.",
        f"- Cases/runs: {len(cases)}/{len(run_results)}; Text judge disabled.",
        f"- Five-field case-mean macro-F1: {case_mean}.",
        f"- Binding-level pooled TP/FP/FN: {binding_counts['tp']}/{binding_counts['fp']}/{binding_counts['fn']}.",
        f"- Input set SHA-256: `{input_before['inputSetSha256']}` (unchanged after evaluation).",
        "- Top-1/top-3 are descriptive shared-run summaries only.",
        (
            "- Selected diagnostic scope only; experimentMatrixComplete=false."
            if args.scope in {"flan-long-context", "mistral-base-md"}
            else "- Four GPT-4.1 cases remain excluded; experimentMatrixComplete=false."
        ),
    ]
    (output_root / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=tuple(SCORERS), required=True)
    parser.add_argument(
        "--scope",
        choices=("candidate12", "candidate34", "candidate-merged34", "flan-long-context", "mistral-base-md"),
        required=True,
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--formal-reference", type=Path)
    parser.add_argument("--sensitivity-reference", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "experiment6_binding_candidate_evaluation_v1.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build(args)
    except (
        CandidateEvaluationError,
        v601.ProtocolError,
        v602.ProtocolError,
        v610.ProtocolError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "status": report["status"],
        "scope": report["scope"],
        "scoringProtocol": report["scoringProtocol"],
        "claimEligible": report["claimEligible"],
        "completedCases": report["completedCases"],
        "completedCaseRuns": report["completedCaseRuns"],
        "caseMeanMacroF1": report["overall"]["caseMeanMacroF1"],
        "lowScoreTriggered": report["overall"]["lowScoreTriggered"],
        "outputRoot": logical_path(args.evaluation_root),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
