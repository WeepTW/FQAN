#!/usr/bin/env python3
"""Audit complete Mistral base m/d v4 generation and six-field evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiment6_mistral_chat_binding_repair import repair_unique_binding

CASES = {"6_mistral_base_m", "6_mistral_base_d"}
RUNS = set(range(1, 11))
SEEDS = {run: 2026073100 + run for run in RUNS}
FIELDS = {"ObjectName", "Trend", "Num", "Text", "Position", "DataName"}
PROMPT_MODES = {"6_mistral_base_m": "many-shot", "6_mistral_base_d": "dynamic-shot"}
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CHAT_TEMPLATE_SHA256 = "e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826"


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AuditError(f"expected JSON object: {path}:{number}")
        rows.append(value)
    return rows


def resolve_file(root: Path, manifest: Mapping[str, Any], name: str) -> Path:
    raw = str((manifest.get("files") or {}).get(name) or "")
    path = Path(raw)
    if not path.is_file():
        path = root / "cases" / str(manifest["outputId"]) / f"run_{int(manifest['run']):02d}" / Path(raw).name
    if not path.is_file():
        raise AuditError(f"missing {name}: {path}")
    expected = str((manifest.get("hashes") or {}).get(name) or "")
    if not expected or sha256_file(path) != expected:
        raise AuditError(f"hash mismatch for {name}: {path}")
    return path


def event_details(path: Path) -> dict[str, str]:
    return {
        str(row["event"]): str(row["detail"])
        for row in read_jsonl(path)
        if row.get("event") and row.get("detail")
    }


def write_inventory(path: Path, files: Iterable[Path]) -> None:
    rows = sorted((str(file.resolve()), file.stat().st_size, sha256_file(file)) for file in set(files))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)


def verified_judge_bundle(binding_root: Path) -> list[Path]:
    judge_root = binding_root / "judge_examples"
    manifest_path = judge_root / "manifest.json"
    manifest = read_json(manifest_path)
    validation = manifest.get("validation") or {}
    if validation.get("status") != "passed" or validation.get("canonicalRows") != 26 or validation.get("canonicalBindings") != 55:
        raise AuditError(f"judge example validation mismatch: {validation}")
    files = manifest.get("files") or {}
    specs = [files.get("canonicalExamples"), files.get("repairManifest")]
    specs.extend((files.get("promptPrefixes") or {}).get(field) for field in ("ObjectName", "Trend", "Text"))
    verified = [manifest_path]
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise AuditError("judge example file specification missing")
        path = Path(str(spec.get("path") or ""))
        if not path.is_file():
            path = judge_root / path.name
        if not path.is_file() or sha256_file(path) != spec.get("sha256"):
            raise AuditError(f"judge example artifact mismatch: {path}")
        verified.append(path)
    return verified


def audit(generation_root: Path, events_path: Path, output: Path) -> dict[str, Any]:
    generation_root = generation_root.resolve()
    events_path = events_path.resolve()
    details = event_details(events_path)
    if not details.get("binding_materialization_complete") or not details.get("finalizer_complete"):
        raise AuditError("finalizer completion events are missing")
    binding_root = Path(details.get("binding_materialization_complete", ""))
    final_root = Path(details.get("finalizer_complete", ""))
    if not binding_root.is_dir() or not final_root.is_dir():
        raise AuditError("finalizer event paths are absent or invalid")

    manifest_paths = sorted((generation_root / "manifests").glob("*.json"))
    if len(manifest_paths) != 20:
        raise AuditError(f"expected 20 manifests, got {len(manifest_paths)}")
    coverage: dict[str, set[int]] = defaultdict(set)
    fingerprints: set[str] = set()
    prompt_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    prompt_echo_rows = 0
    inventory_files: list[Path] = [events_path]
    device_rows: dict[str, int] = defaultdict(int)
    format_totals = {"accepted": 0, "rejected": 0}
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        case = str(manifest.get("outputId") or "")
        run = int(manifest.get("run", 0))
        coverage[case].add(run)
        if case not in CASES or run not in RUNS or manifest.get("seed") != SEEDS[run]:
            raise AuditError(f"identity mismatch: {manifest_path}")
        if not str(manifest.get("status") or "").startswith("completed"):
            raise AuditError(f"non-complete manifest: {manifest_path}")
        if manifest.get("expectedRows") != 85 or manifest.get("runtimeBlockedRows") != 0:
            raise AuditError(f"row/blocker mismatch: {manifest_path}")
        if int(manifest.get("acceptedRows", 0)) + int(manifest.get("rejectedRows", 0)) != 85:
            raise AuditError(f"accepted/rejected mismatch: {manifest_path}")
        if manifest.get("declaredRoute") != "direct-binding" or manifest.get("effectiveRoute") != "direct-binding":
            raise AuditError(f"route mismatch: {manifest_path}")
        if manifest.get("adapter") is not None or manifest.get("converterModel") is not None:
            raise AuditError(f"adapter/converter used: {manifest_path}")
        if manifest.get("actualModel") != MODEL or manifest.get("promptMode") != PROMPT_MODES[case]:
            raise AuditError(f"model/prompt-mode mismatch: {manifest_path}")
        fingerprints.add(str(manifest.get("compatibilityFingerprint") or ""))
        format_totals["accepted"] += int(manifest.get("acceptedRows", 0))
        format_totals["rejected"] += int(manifest.get("rejectedRows", 0))
        inventory_files.append(manifest_path)
        for name in ("predictions", "rawResponse", "prompts", "runtime", "formatReport", "nonformalRepair", "stage1Raw"):
            inventory_files.append(resolve_file(generation_root, manifest, name))
        prompts = read_jsonl(resolve_file(generation_root, manifest, "prompts"))
        if len(prompts) != 85 or len({str(row.get("source") or "") for row in prompts}) != 85:
            raise AuditError(f"prompt Source coverage mismatch: {manifest_path}")
        prompt_by_source: dict[str, str] = {}
        for prompt_row in prompts:
            source = str(prompt_row.get("source") or "")
            prompt = str(prompt_row.get("directPrompt") or "")
            prompt_sha = str(prompt_row.get("directPromptSha256") or "")
            if not source or prompt_sha != sha256_text(prompt):
                raise AuditError(f"prompt hash mismatch: {manifest_path} {source}")
            if any(marker in prompt for marker in ('"targetBindings"', '"Binding_Result"', '"gold_targets"')):
                raise AuditError(f"gold marker in generation prompt: {manifest_path} {source}")
            prompt_by_source[source] = prompt_sha
        predictions = read_jsonl(resolve_file(generation_root, manifest, "predictions"))
        if len(predictions) != 85 or len({str(row.get("source") or "") for row in predictions}) != 85:
            raise AuditError(f"Source coverage mismatch: {manifest_path}")
        for row in predictions:
            source = str(row.get("source") or "")
            prompt_sha = str(row.get("promptSha256") or "")
            if prompt_by_source.get(source) != prompt_sha:
                raise AuditError(f"prediction prompt identity mismatch: {manifest_path} {source}")
            prompt_hashes[(case, source)].add(prompt_sha)
            if repair_unique_binding(str(row.get("rawResponse") or "")).get("reason") == "prompt_echo_guard":
                prompt_echo_rows += 1
        runtime = read_json(resolve_file(generation_root, manifest, "runtime"))
        stages = runtime.get("stages") or []
        if len(stages) != 1 or not isinstance(stages[0], dict):
            raise AuditError(f"unexpected runtime stages: {manifest_path}")
        raw = stages[0].get("raw") or {}
        if raw.get("execution_device") != "cuda" or raw.get("use_adapter") is not False:
            raise AuditError(f"device/adapter runtime mismatch: {manifest_path}")
        if raw.get("converter_used") is not False or raw.get("generation_cache_used") is not False:
            raise AuditError(f"converter/cache runtime mismatch: {manifest_path}")
        if raw.get("chat_template_applied") is not True or raw.get("structured_output") != "off":
            raise AuditError(f"chat/structured runtime mismatch: {manifest_path}")
        if raw.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256:
            raise AuditError(f"chat-template identity mismatch: {manifest_path}")
        if raw.get("max_input_tokens") != 8192 or raw.get("context_window") != 12288:
            raise AuditError(f"context contract mismatch: {manifest_path}")
        if raw.get("batch_size_effective") not in {1, 3, 6}:
            raise AuditError(f"unexpected batch-size fallback: {manifest_path}")
        device_rows[str(raw.get("cuda_visible_devices"))] += 85
    if set(coverage) != CASES or any(runs != RUNS for runs in coverage.values()):
        raise AuditError(f"case/run coverage mismatch: {dict(coverage)}")
    if len(fingerprints) != 1 or "" in fingerprints:
        raise AuditError(f"fingerprint mismatch: {sorted(fingerprints)}")
    if prompt_echo_rows:
        raise AuditError(f"prompt echo rows remain: {prompt_echo_rows}")
    unstable_prompts = [identity for identity, hashes in prompt_hashes.items() if len(hashes) != 1]
    if len(prompt_hashes) != 170 or unstable_prompts:
        raise AuditError(
            f"cross-run prompt identity mismatch: identities={len(prompt_hashes)} "
            f"unstable={unstable_prompts[:5]}"
        )

    dataset_path = binding_root / "dataset_manifest.json"
    dataset = read_json(dataset_path)
    if (
        dataset.get("protocol") != "experiment6-mistral-chat-repaired-projection-v1"
        or dataset.get("status") != "complete"
        or dataset.get("official") is not False
        or dataset.get("diagnosticOnly") is not True
        or dataset.get("claimEligible") is not False
        or dataset.get("goldAccessed") is not False
    ):
        raise AuditError("binding projection identity/safety mismatch")
    counts = dataset.get("counts") or {}
    if counts.get("cases") != 2 or counts.get("caseRuns") != 20 or counts.get("rows") != 1700:
        raise AuditError(f"binding projection coverage mismatch: {counts}")
    inventory_files.extend([dataset_path, binding_root / "sha256_inventory.tsv"])
    inventory_files.extend(verified_judge_bundle(binding_root))

    final_report_path = final_root / "evaluation_report.json"
    final_markdown_path = final_root / "evaluation_report.md"
    mean_path = final_root / "experiment_6_v6_欄位分數_mean.md"
    final = read_json(final_report_path)
    if final.get("protocol") != "experiment6-reference-aligned-v6.1-with-semantic-text-v1" or final.get("status") != "completed":
        raise AuditError("unexpected final evaluation protocol/status")
    inputs = final.get("inputs") or {}
    component_paths: dict[str, Path] = {}
    for name, path_key, hash_key in (
        ("v610", "v610Report", "v610ReportSha256"),
        ("semantic", "semanticTextReport", "semanticTextReportSha256"),
    ):
        path = Path(str(inputs.get(path_key) or ""))
        if not path.is_file() or sha256_file(path) != inputs.get(hash_key):
            raise AuditError(f"final component provenance mismatch: {name} {path}")
        component_paths[name] = path
    v610 = read_json(component_paths["v610"])
    validation = v610.get("candidateValidation") or {}
    if (
        v610.get("protocol") != "experiment6-binding-candidate-evaluation-v1"
        or v610.get("scoringProtocol") != "experiment6-reference-aligned-v6.1.0"
        or v610.get("scope") != "mistral-base-md"
        or v610.get("diagnosticOnly") is not True
        or v610.get("claimEligible") is not False
        or validation.get("status") != "passed"
        or validation.get("cases") != 2
        or validation.get("caseRuns") != 20
        or validation.get("rows") != 1700
        or validation.get("datasetManifestSha256") != sha256_file(dataset_path)
        or validation.get("inventorySha256") != sha256_file(binding_root / "sha256_inventory.tsv")
    ):
        raise AuditError("v6.1 repaired-projection provenance mismatch")
    semantic = read_json(component_paths["semantic"])
    semantic_judge = semantic.get("judge") or {}
    if (
        semantic.get("protocol") != "narrative2-reference-aligned-hybrid-v5.1"
        or semantic.get("status") != "completed"
        or semantic.get("completedCases") != 2
        or semantic.get("completedCaseRuns") != 20
        or semantic.get("formalPredictions") != 1700
        or semantic_judge.get("model") != "gpt-5.5"
        or semantic_judge.get("reasoningEffort") != "medium"
        or float(semantic_judge.get("minimumConfidence", -1)) != 0.8
        or semantic_judge.get("disabled") is not False
    ):
        raise AuditError("semantic Text component provenance mismatch")
    inventory_files.extend(component_paths.values())
    judge = final.get("judge") or {}
    if judge.get("model") != "gpt-5.5" or judge.get("reasoningEffort") != "medium" or float(judge.get("minimumConfidence", -1)) != 0.8 or judge.get("disabled") is not False:
        raise AuditError(f"judge identity/confidence mismatch: {judge}")
    cases = final.get("cases") or []
    if {str(item.get("outputId")) for item in cases} != CASES:
        raise AuditError("final case set mismatch")
    scores: dict[str, Any] = {}
    for item in cases:
        if item.get("runs") != 10 or set(item.get("fields") or {}) != FIELDS:
            raise AuditError(f"six-field/run mismatch: {item.get('outputId')}")
        for field, metrics in item["fields"].items():
            if set(metrics) != {"precision", "recall", "f1"}:
                raise AuditError(f"metric keys mismatch: {item.get('outputId')} {field}")
        scores[str(item["outputId"])] = {"fields": item["fields"], "macro": item["macro"]}
    inventory_files.extend([final_report_path, final_markdown_path, mean_path])

    report = {
        "status": "complete",
        "protocol": "experiment6-mistral-base-md-v4-completion-audit-v1",
        "generationRoot": str(generation_root),
        "bindingRoot": str(binding_root.resolve()),
        "evaluationRoot": str(final_root.resolve()),
        "coverage": {"cases": 2, "caseRuns": 20, "rows": 1700, "sourcesPerRun": 85},
        "seeds": [SEEDS[run] for run in sorted(RUNS)],
        "compatibilityFingerprint": next(iter(fingerprints)),
        "promptEchoRows": 0,
        "promptIdentity": {
            "caseSources": len(prompt_hashes),
            "runsPerCaseSource": 10,
            "stable": True,
            "goldMarkers": 0,
        },
        "runtimeBlockedRows": 0,
        "formatRows": format_totals,
        "deviceRows": dict(sorted(device_rows.items())),
        "route": "direct-binding",
        "adapter": None,
        "converter": None,
        "generationCache": False,
        "judge": {"model": "gpt-5.5", "reasoningEffort": "medium", "minimumConfidence": 0.8},
        "scores": scores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_inventory(output.with_name("sha256_inventory.tsv"), [*inventory_files, output])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--finalizer-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.generation_root, args.finalizer_events, args.output)
    except (AuditError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
