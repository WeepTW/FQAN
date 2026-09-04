#!/usr/bin/env python3
"""Verify complete local-retriever coverage for Experiment 6 narrative2 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_SOURCES = {
    "flan": {"finqa_flan_z", "finqa_flan_m", "finqa_flan_d", "flan_t5_large"},
    "mistral": {
        "finqa_mistral_z", "finqa_mistral_m", "finqa_mistral_d", "mistral_v0_3"
    },
    "t5gemma2": {
        "finqa_t5gemma2_z", "finqa_t5gemma2_m", "finqa_t5gemma2_d",
        "t5gemma_2_1b_1b",
    },
}
EXPECTED_MODELS = {
    "flan": "google/flan-t5-large",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "t5gemma2": "google/t5gemma-2-1b-1b",
}
MODE_SUFFIX = {"zero-shot": "z", "many-shot": "m", "dynamic-shot": "d"}


class VerificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def optional_fmean(values: Sequence[float]) -> float | None:
    """Return a mean only when at least one verified value is available."""
    return statistics.fmean(values) if values else None


def format_optional(value: float | None, digits: int = 6) -> str:
    """Format verified metrics while preserving an explicit missing marker."""
    return "—" if value is None else f"{value:.{digits}f}"


def expand_retriever_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    retriever_sources = set().union(*FAMILY_SOURCES.values())
    for part in config["parts"]:
        part_number = int(part["part"])
        if part_number == 1:
            for item in part["cases"]:
                cases.append({
                    **item,
                    "part": part_number,
                    "route": part["route"],
                })
        elif part_number == 2:
            for model in part["models"]:
                if model["sourceId"] not in retriever_sources:
                    continue
                for mode in part["promptModes"]:
                    cases.append({
                        "outputId": f"{model['outputStem']}_{MODE_SUFFIX[mode]}",
                        "sourceId": model["sourceId"],
                        "promptMode": mode,
                        "part": part_number,
                        "route": part["route"],
                    })
        elif part_number == 3:
            for item in part["cases"]:
                if item["sourceId"] not in retriever_sources:
                    continue
                cases.append({
                    **item,
                    "promptMode": part["promptMode"],
                    "part": part_number,
                })
    output_ids = [item["outputId"] for item in cases]
    if len(cases) != 30 or len(set(output_ids)) != 30:
        raise VerificationError(
            f"retriever matrix must contain 30 unique cases, got {len(cases)}"
        )
    return cases


def family_for_source(source_id: str) -> str:
    for family, sources in FAMILY_SOURCES.items():
        if source_id in sources:
            return family
    raise VerificationError(f"unknown retriever source: {source_id}")


def verify(output_root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    cases = expand_retriever_cases(config)
    expected_rows = int(config["expectedRows"])
    runs = [int(run) for run in config["runs"]]
    failures: list[dict[str, Any]] = []
    family_stats: dict[str, dict[str, Any]] = {}
    case_stats: list[dict[str, Any]] = []
    family_accumulator: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        output_id = str(case["outputId"])
        source_id = str(case["sourceId"])
        family = family_for_source(source_id)
        prediction_hashes: list[str] = []
        candidate_hashes: list[str] = []
        format_rates: list[float] = []
        run_statuses: Counter[str] = Counter()
        runtime_seconds: list[float] = []
        execution_windows: list[tuple[str, str]] = []
        for run in runs:
            manifest_path = (
                output_root / "manifests" / f"{output_id}__run_{run:02d}.json"
            )
            if not manifest_path.is_file():
                failures.append({
                    "outputId": output_id,
                    "run": run,
                    "error": "manifest_missing",
                })
                continue
            try:
                manifest = read_json(manifest_path)
                expected_seed = int(config["seedBase"]) + run
                if int(manifest.get("run", -1)) != run:
                    raise VerificationError("manifest run mismatch")
                if int(manifest.get("seed", -1)) != expected_seed:
                    raise VerificationError("manifest seed mismatch")
                status = str(manifest.get("status"))
                started_at = manifest.get("startedAt")
                finished_at = manifest.get("finishedAt")
                if (
                    not isinstance(started_at, str)
                    or not isinstance(finished_at, str)
                    or started_at >= finished_at
                ):
                    raise VerificationError("invalid execution window")
                execution_windows.append((started_at, finished_at))
                run_statuses[status] += 1
                if status not in {"completed", "completed_with_format_errors"}:
                    raise VerificationError(f"bad status {status}")
                if manifest.get("expectedRows") != expected_rows:
                    raise VerificationError("expectedRows mismatch")
                if (
                    int(manifest.get("acceptedRows", -1))
                    + int(manifest.get("rejectedRows", -1))
                    != expected_rows
                ):
                    raise VerificationError("accepted+rejected mismatch")
                if int(manifest.get("runtimeBlockedRows", -1)) != 0:
                    raise VerificationError("runtimeBlockedRows is not zero")
                if manifest.get("sourceId") != source_id:
                    raise VerificationError("sourceId mismatch")
                if manifest.get("requestedModel") != source_id:
                    raise VerificationError("requestedModel mismatch")
                if manifest.get("actualModel") != EXPECTED_MODELS[family]:
                    raise VerificationError("actualModel mismatch")
                adapter_expected = source_id.startswith("finqa_")
                if bool(manifest.get("adapter")) != adapter_expected:
                    raise VerificationError("adapter provenance mismatch")
                if family == "mistral" and manifest.get("quantization") != "4bit-nf4":
                    raise VerificationError("Mistral quantization mismatch")
                if manifest.get("effectiveRoute") != "retriever-converter":
                    raise VerificationError("effectiveRoute mismatch")
                if (
                    manifest.get("converterModel") != "gpt-5.5"
                    or manifest.get("reasoningEffort") != "medium"
                ):
                    raise VerificationError("converter identity mismatch")

                files = manifest.get("files") or {}
                hashes = manifest.get("hashes") or {}
                required = (
                    "predictions", "prompts", "runtime", "formatReport",
                    "nonformalRepair", "converterRawResponses",
                    "retrieverCandidates",
                )
                for name in required:
                    raw_path = files.get(name)
                    if not raw_path:
                        raise VerificationError(f"missing file entry {name}")
                    path = Path(str(raw_path))
                    if not path.is_file():
                        raise VerificationError(f"missing artifact {name}")
                    if hashes.get(name) != sha256_file(path):
                        raise VerificationError(f"SHA mismatch {name}")

                runtime_payload = read_json(Path(files["runtime"]))
                if int(runtime_payload.get("seed", -1)) != int(manifest["seed"]):
                    raise VerificationError("runtime seed mismatch")
                stages = runtime_payload.get("stages") or []
                if len(stages) < 2:
                    raise VerificationError("retriever/converter runtime stages missing")
                retriever_raw = stages[0].get("raw") or {}
                if retriever_raw.get("structured_output") != "canonical":
                    raise VerificationError("structured_output is not canonical")
                if int(retriever_raw.get("max_new_tokens", -1)) != 128:
                    raise VerificationError("max_new_tokens is not 128")
                expected_cuda = str(
                    config["retriever"]["familyCudaVisibleDevices"][family]
                )
                if str(retriever_raw.get("cuda_visible_devices")) != expected_cuda:
                    raise VerificationError("family CUDA route mismatch")
                if family == "t5gemma2":
                    if int(retriever_raw.get("run_seed", -1)) != int(manifest["seed"]):
                        raise VerificationError("T5Gemma2 run seed was not applied")
                    if runtime_payload.get("seedSupport") != "transformers-set-seed-sampled":
                        raise VerificationError("T5Gemma2 seedSupport mismatch")

                predictions = read_jsonl(Path(files["predictions"]))
                prompts = read_jsonl(Path(files["prompts"]))
                candidates = read_jsonl(Path(files["retrieverCandidates"]))
                converter_rows = read_jsonl(Path(files["converterRawResponses"]))
                for name, values in (
                    ("predictions", predictions),
                    ("prompts", prompts),
                    ("retrieverCandidates", candidates),
                    ("converterRawResponses", converter_rows),
                ):
                    if len(values) != expected_rows:
                        raise VerificationError(
                            f"{name} rows={len(values)} expected={expected_rows}"
                        )
                if any(
                    str(item.get("status", "")).startswith("runtime_blocked")
                    for item in converter_rows
                ):
                    raise VerificationError("converter contains runtime-blocked rows")
                if [item.get("source") for item in predictions] != [
                    item.get("source") for item in prompts
                ]:
                    raise VerificationError("prediction/prompt Source order mismatch")

                expected_sources = [item.get("source") for item in prompts]
                for index, (prediction, candidate, converter) in enumerate(zip(
                    predictions, candidates, converter_rows
                )):
                    if (
                        prediction.get("index") != index
                        or prediction.get("source") != expected_sources[index]
                        or prediction.get("run") != run
                        or prediction.get("seed") != expected_seed
                        or prediction.get("requestedModel") != source_id
                        or prediction.get("actualModel") != EXPECTED_MODELS[family]
                    ):
                        raise VerificationError(
                            f"prediction provenance mismatch at row {index}"
                        )
                    if (
                        candidate.get("index") != index
                        or candidate.get("source") != expected_sources[index]
                        or candidate.get("run") != run
                        or candidate.get("seed") != expected_seed
                        or not isinstance(candidate.get("raw"), str)
                        or not isinstance(candidate.get("candidate"), str)
                        or candidate.get("candidateSha256")
                        != sha256_text(candidate["candidate"])
                    ):
                        raise VerificationError(
                            f"candidate provenance mismatch at row {index}"
                        )
                    if (
                        converter.get("index") != index
                        or converter.get("source") != expected_sources[index]
                        or converter.get("seed") != expected_seed + index
                        or converter.get("requestedModel") != "gpt-5.5"
                        or converter.get("actualModel") != "gpt-5.5"
                        or converter.get("reasoningEffort") != "medium"
                        or converter.get("candidateSha256")
                        != candidate.get("candidateSha256")
                    ):
                        raise VerificationError(
                            f"converter provenance mismatch at row {index}"
                        )

                prediction_hashes.append(str(hashes["predictions"]))
                candidate_hashes.append(str(hashes["retrieverCandidates"]))
                format_rates.append(float(manifest["formatComplianceRate"]))
                runtime_seconds.append(float(manifest["runtimeSeconds"]))
            except Exception as exc:
                failures.append({
                    "outputId": output_id,
                    "run": run,
                    "error": str(exc),
                    "manifest": str(manifest_path),
                })
        if (
            len(execution_windows) == len(runs)
            and len(set(execution_windows)) != len(execution_windows)
        ):
            failures.append({
                "outputId": output_id,
                "run": None,
                "error": "duplicate execution windows across runs",
            })
        case_report = {
            "outputId": output_id,
            "sourceId": source_id,
            "family": family,
            "expectedRuns": len(runs),
            "completedRuns": sum(run_statuses.values()),
            "statusCounts": dict(run_statuses),
            "uniquePredictionHashes": len(set(prediction_hashes)),
            "uniqueCandidateHashes": len(set(candidate_hashes)),
            "formatCompliance": {
                "mean": statistics.fmean(format_rates) if format_rates else None,
                "min": min(format_rates) if format_rates else None,
                "max": max(format_rates) if format_rates else None,
            },
            "runtimeSeconds": {
                "sum": sum(runtime_seconds),
                "mean": statistics.fmean(runtime_seconds)
                if runtime_seconds else None,
            },
        }
        case_stats.append(case_report)
        family_accumulator[family].append(case_report)

    for family, values in sorted(family_accumulator.items()):
        family_format_rates = [
            item["formatCompliance"]["mean"]
            for item in values
            if item["formatCompliance"]["mean"] is not None
        ]
        family_stats[family] = {
            "expectedCases": 10,
            "reportedCases": len(values),
            "completedRuns": sum(item["completedRuns"] for item in values),
            "expectedRuns": 100,
            "expectedPredictions": 8500,
            "formatComplianceMean": optional_fmean(family_format_rates),
        }

    report = {
        "time": utc_now(),
        "protocol": config["protocol"],
        "status": "completed" if not failures else "incomplete",
        "outputRoot": str(output_root),
        "configPath": str(config_path),
        "configSha256": sha256_file(config_path),
        "expectedCases": 30,
        "expectedCaseRuns": 300,
        "expectedPredictions": 25500,
        "families": family_stats,
        "cases": case_stats,
        "failures": failures,
        "rankingPublished": False,
    }
    report_path = output_root / "retriever_completion_report.json"
    atomic_write(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    lines = [
        "# Experiment 6 local retriever completion",
        "",
        f"- Status: `{report['status']}`",
        f"- Cases: {sum(item['completedRuns'] == 10 for item in case_stats)}/30",
        f"- Case-runs: {sum(item['completedRuns'] for item in case_stats)}/300",
        f"- Expected predictions: {report['expectedPredictions']}",
        "- Ranking published: false",
        "",
        "| Family | Cases | Runs | Predictions | Mean format compliance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family, item in family_stats.items():
        mean = item["formatComplianceMean"]
        lines.append(
            f"| {family} | {item['reportedCases']}/10 | "
            f"{item['completedRuns']}/100 | {item['expectedPredictions']} | "
            f"{format_optional(mean)} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures[:100]:
            lines.append(
                f"- {failure['outputId']} run {failure['run']}: "
                f"{failure['error']}"
            )
    atomic_write(
        output_root / "retriever_completion_report.md",
        "\n".join(lines) + "\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--print-output-ids",
        action="store_true",
        help="Print the config-derived 30 retriever output IDs and exit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "experiment6_narrative2_generation.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_output_ids:
        config = read_json(args.config.resolve())
        for case in expand_retriever_cases(config):
            print(case["outputId"])
        return 0
    report = verify(args.output_root.resolve(), args.config.resolve())
    print(json.dumps({
        "status": report["status"],
        "expectedCases": report["expectedCases"],
        "expectedCaseRuns": report["expectedCaseRuns"],
        "failures": len(report["failures"]),
    }, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
