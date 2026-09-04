#!/usr/bin/env python3
"""Independently validate and summarize completed Experiment 6 no-GPT-4.1 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FIELDS = ("ObjectName", "DataName", "Position", "Trend", "Num", "Text")
ALLOWED_GENERATION_STATUSES = {"completed", "completed_with_format_errors"}
TOLERANCE = 1e-12


class ValidationError(RuntimeError):
    """Raised when completed artifacts fail independent validation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl_rows(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"{path}:{line_number}: {error}") from error
            rows += 1
    return rows


def metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    if tp == fp == fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def require_close(label: str, reported: Any, expected: float) -> None:
    value = float(reported)
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE
    ):
        raise ValidationError(f"{label}: reported={value} expected={expected}")


def validate_summary(
    label: str,
    summary: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    top: Sequence[Mapping[str, Any]],
    field: str | None,
) -> None:
    def value(run: Mapping[str, Any], name: str) -> float:
        source = run["fields"][field] if field else run["macro"]
        return float(source[name])

    for name in ("precision", "recall", "f1"):
        values = [value(run, name) for run in runs]
        require_close(f"{label}.mean.{name}", summary["mean"][name], statistics.mean(values))
        require_close(
            f"{label}.sample_sd.{name}",
            summary["sample_sd"][name],
            statistics.stdev(values),
        )
        require_close(f"{label}.min.{name}", summary["min"][name], min(values))
        require_close(f"{label}.max.{name}", summary["max"][name], max(values))
        require_close(f"{label}.top1.{name}", summary["top1"][name], value(top[0], name))
        require_close(
            f"{label}.top3.{name}",
            summary["top3"][name],
            statistics.mean(value(run, name) for run in top),
        )
    if int(summary["top1"]["run_id"]) != int(top[0]["run"]):
        raise ValidationError(f"{label}: top1 run mismatch")
    if [int(value) for value in summary["top3"]["run_ids"]] != [
        int(run["run"]) for run in top
    ]:
        raise ValidationError(f"{label}: top3 runs mismatch")
    if field is not None:
        counts = {
            name: sum(int(run["fields"][field][name]) for run in runs)
            for name in ("tp", "fp", "fn")
        }
        expected = metric(**counts)
        for name, value_expected in expected.items():
            require_close(
                f"{label}.pooled_micro.{name}",
                summary["pooled_micro"][name],
                float(value_expected),
            )


def validate_data(config: Mapping[str, Any], expected_rows: int) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[2]
    declared = {
        "sourceWorkbook": config["sourceWorkbook"],
        "inferenceWorkbook": config["inferenceWorkbook"],
        "goldTargets": {
            "path": config["evaluation"]["goldPath"],
            "sha256": config["evaluation"]["goldSha256"],
        },
    }
    artifacts: dict[str, Any] = {}
    for name, item in declared.items():
        path = workspace_root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValidationError(f"declared data SHA mismatch: {name}: {path}")
        artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    gold = read_json(Path(artifacts["goldTargets"]["path"]))
    rows = gold.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValidationError("gold row count mismatch")
    sources = [str(row.get("source") or "") for row in rows]
    if not all(sources) or len(set(sources)) != expected_rows:
        raise ValidationError("gold Source coverage is empty or non-unique")
    bindings = 0
    rows_without_bindings = 0
    for row in rows:
        target_bindings = row.get("targetBindings")
        if not isinstance(target_bindings, list):
            raise ValidationError("gold targetBindings is not an array")
        rows_without_bindings += int(not target_bindings)
        for binding in target_bindings:
            if not isinstance(binding, dict) or set(binding) != set(FIELDS):
                raise ValidationError("gold binding is not exact six-field schema")
            bindings += 1
    return {
        "rows": len(rows),
        "uniqueSources": len(set(sources)),
        "bindings": bindings,
        "rowsWithoutBindings": rows_without_bindings,
        "artifacts": artifacts,
    }


def validate_manifests(
    output_root: Path,
    expected_ids: set[str],
    expected_runs: int,
    expected_rows: int,
    excluded_source_ids: set[str],
    required_chatmock_model: str,
    required_reasoning_effort: str,
) -> dict[str, Any]:
    manifests: dict[tuple[str, int], dict[str, Any]] = {}
    status_counts: dict[str, int] = {}
    prediction_rows = 0
    format_error_runs = 0
    for path in sorted((output_root / "manifests").glob("*.json")):
        item = read_json(path)
        output_id = str(item.get("outputId"))
        run = int(item.get("run", -1))
        key = (output_id, run)
        if output_id not in expected_ids or key in manifests:
            raise ValidationError(f"manifest outside scope or duplicate: {path}")
        if run not in range(1, expected_runs + 1):
            raise ValidationError(f"invalid run ID: {path}")
        if not item.get("official"):
            raise ValidationError(f"non-official manifest: {path}")
        if str(item.get("sourceId")) in excluded_source_ids:
            raise ValidationError(f"excluded source manifest: {path}")
        if item.get("reasoningEffort") != required_reasoning_effort:
            raise ValidationError(f"reasoning effort mismatch: {path}")
        uses_required_chatmock = (
            item.get("actualModel") == required_chatmock_model
            or item.get("converterModel") == required_chatmock_model
        )
        if not uses_required_chatmock:
            raise ValidationError(f"required ChatMock model absent: {path}")
        status = str(item.get("status"))
        if status not in ALLOWED_GENERATION_STATUSES:
            raise ValidationError(f"nonfinal generation status: {path}: {status}")
        status_counts[status] = status_counts.get(status, 0) + 1
        format_error_runs += int(status == "completed_with_format_errors")
        if int(item.get("runtimeBlockedRows") or 0) != 0:
            raise ValidationError(f"runtime-blocked rows present: {path}")
        accounted = sum(
            int(item.get(name) or 0)
            for name in ("acceptedRows", "rejectedRows", "runtimeBlockedRows")
        )
        if int(item.get("expectedRows", -1)) != expected_rows or accounted != expected_rows:
            raise ValidationError(f"row accounting mismatch: {path}")
        prediction_path = Path(str(item["files"]["predictions"]))
        if not prediction_path.is_file():
            raise ValidationError(f"missing predictions: {prediction_path}")
        if sha256_file(prediction_path) != item["hashes"]["predictions"]:
            raise ValidationError(f"prediction SHA mismatch: {prediction_path}")
        rows = jsonl_rows(prediction_path)
        if rows != expected_rows:
            raise ValidationError(f"prediction row mismatch: {prediction_path}: {rows}")
        prediction_rows += rows
        manifests[key] = item
    expected_keys = {
        (output_id, run)
        for output_id in expected_ids
        for run in range(1, expected_runs + 1)
    }
    if set(manifests) != expected_keys:
        raise ValidationError("manifest case/run coverage is not exact")
    return {
        "caseRuns": len(manifests),
        "predictionRows": prediction_rows,
        "statusCounts": status_counts,
        "formatErrorCaseRuns": format_error_runs,
        "manifests": manifests,
    }


def validate_judge(
    output_root: Path, expected_runs: int, minimum_confidence: float
) -> dict[str, Any]:
    evaluation_root = output_root / "evaluation_v4_no_gpt41"
    checkpoint_path = evaluation_root / "judge_checkpoint.jsonl"
    checkpoints: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        with checkpoint_path.open(encoding="utf-8") as handle:
            checkpoints = [json.loads(line) for line in handle if line.strip()]
    for item in checkpoints:
        if item.get("status") != "completed":
            raise ValidationError("judge checkpoint contains noncompleted call")
        if item.get("responseModel") != "gpt-5.5":
            raise ValidationError("judge response model is not gpt-5.5")
        if item.get("reasoningEffort") != "medium":
            raise ValidationError("judge reasoning effort is not medium")
        raw = str(item.get("rawResponse") or "")
        if not raw or sha256_text(raw) != item.get("rawResponseSha256"):
            raise ValidationError("judge raw response SHA mismatch")
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"decisions"}:
            raise ValidationError("judge raw response is not strict decisions JSON")
        for decision in payload["decisions"]:
            if not isinstance(decision, dict) or set(decision) != {
                "decisionId", "equivalent", "matchedPairs", "confidence",
                "evidenceSpan", "reasonCode",
            }:
                raise ValidationError("judge raw decision schema mismatch")
        for decision in item.get("validatedDecisions", []):
            if decision.get("accepted") and (
                not decision.get("equivalent")
                or float(decision.get("confidence", -1)) < minimum_confidence
                or not str(decision.get("evidenceSpan") or "").strip()
            ):
                raise ValidationError("accepted judge decision violates confidence/evidence gate")
        version = str(item.get("validationVersion") or "")
        if not any(f"-{field}-" in version for field in ("ObjectName", "Trend", "Text")):
            raise ValidationError("judge checkpoint lacks field-specific validation version")
    audit_population = audit_sampled = audit_disagreements = third_calls = 0
    audit_files = sorted((evaluation_root / "cases").glob("*/run_*/semantic_audit.json"))
    expected_audit_files = len(list((evaluation_root / "cases").glob("*/run_*")))
    if len(audit_files) != expected_audit_files:
        raise ValidationError("semantic audit file coverage mismatch")
    for path in audit_files:
        audit = read_json(path)
        summary = audit["summary"]
        records = audit["records"]
        population = int(summary["population"])
        sampled = int(summary["sampled"])
        expected_sampled = max(1, math.ceil(population * 0.1)) if population else 0
        if sampled != expected_sampled or len(records) != sampled:
            raise ValidationError(f"10% semantic audit mismatch: {path}")
        disagreements = 0
        for record in records:
            primary = record["primary"]
            swapped = record["swapped"]
            primary_signature = (
                bool(primary.get("accepted")),
                tuple(sorted(
                    (pair["goldIndex"], pair["predictionIndex"])
                    for pair in primary.get("matchedPairs", [])
                )),
            )
            swapped_signature = (
                bool(swapped.get("accepted")),
                tuple(sorted(
                    (pair["goldIndex"], pair["predictionIndex"])
                    for pair in swapped.get("matchedPairs", [])
                )),
            )
            agreement = primary_signature == swapped_signature
            if agreement != bool(record["agreement"]):
                raise ValidationError(f"semantic audit agreement mismatch: {path}")
            if bool(primary.get("abSwapped")) == bool(swapped.get("abSwapped")):
                raise ValidationError(f"semantic audit did not swap A/B: {path}")
            disagreements += int(not agreement)
        thirds = sum(record.get("thirdAdjudication") is not None for record in records)
        agreements = sampled - disagreements
        if (
            disagreements != int(summary["disagreements"])
            or agreements != int(summary["agreements"])
            or thirds != disagreements
            or thirds != int(summary["thirdAdjudications"])
        ):
            raise ValidationError(f"third adjudication mismatch: {path}")
        expected_rate = agreements / sampled if sampled else None
        if expected_rate is None:
            if summary.get("agreementRate") is not None:
                raise ValidationError(f"empty audit agreement rate mismatch: {path}")
        else:
            require_close(
                f"{path}.agreementRate", summary["agreementRate"], expected_rate
            )
        audit_population += population
        audit_sampled += sampled
        audit_disagreements += disagreements
        third_calls += thirds
    return {
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
        "checkpointCalls": len(checkpoints),
        "semanticAuditFiles": len(audit_files),
        "semanticDecisionPopulation": audit_population,
        "semanticAuditSampled": audit_sampled,
        "semanticAuditDisagreements": audit_disagreements,
        "thirdAdjudications": third_calls,
        "expectedRunsPerCase": expected_runs,
    }


def validate_results(
    output_root: Path,
    results: Sequence[Mapping[str, Any]],
    expected_ids: set[str],
    expected_runs: int,
) -> list[dict[str, Any]]:
    if len(results) != len(expected_ids):
        raise ValidationError("result case count mismatch")
    if {str(item["model"]["output_id"]) for item in results} != expected_ids:
        raise ValidationError("result output IDs mismatch")
    scorecard: list[dict[str, Any]] = []
    for item in results:
        output_id = str(item["model"]["output_id"])
        scores = item["scores"]
        if set(scores["fields"]) != set(FIELDS):
            raise ValidationError(f"{output_id}: field order/set mismatch")
        runs = sorted(scores["runs"], key=lambda run: int(run["run"]))
        if [int(run["run"]) for run in runs] != list(range(1, expected_runs + 1)):
            raise ValidationError(f"{output_id}: run coverage mismatch")
        for run in runs:
            for field in FIELDS:
                reported = run["fields"][field]
                expected = metric(
                    int(reported["tp"]), int(reported["fp"]), int(reported["fn"])
                )
                for name in ("precision", "recall", "f1"):
                    require_close(
                        f"{output_id}.run{run['run']}.{field}.{name}",
                        reported[name],
                        float(expected[name]),
                    )
            for name in ("precision", "recall", "f1"):
                require_close(
                    f"{output_id}.run{run['run']}.macro.{name}",
                    run["macro"][name],
                    statistics.mean(float(run["fields"][field][name]) for field in FIELDS),
                )
        top = sorted(
            runs,
            key=lambda run: (
                -float(run["macro"]["f1"]),
                -float(run["macro"]["precision"]),
                -float(run["macro"]["recall"]),
                int(run["run"]),
            ),
        )[:3]
        selection = scores["selection"]
        if int(selection["top1_run"]) != int(top[0]["run"]) or [
            int(value) for value in selection["top3_runs"]
        ] != [int(run["run"]) for run in top]:
            raise ValidationError(f"{output_id}: common run selection mismatch")
        validate_summary(f"{output_id}.overall", scores["overall"], runs, top, None)
        for field in FIELDS:
            validate_summary(
                f"{output_id}.{field}", scores["fields"][field], runs, top, field
            )
        row: dict[str, Any] = {
            "output_id": output_id,
            "requested_model": item["model"]["requested"],
            "part": item["prompt"]["part"],
            "prompt_mode": item["prompt"]["mode"],
            "overall_mean_precision": scores["overall"]["mean"]["precision"],
            "overall_mean_recall": scores["overall"]["mean"]["recall"],
            "overall_mean_f1": scores["overall"]["mean"]["f1"],
            "top1_run": scores["selection"]["top1_run"],
            "top1_f1": scores["overall"]["top1"]["f1"],
            "top3_runs": json.dumps(scores["selection"]["top3_runs"]),
            "top3_f1": scores["overall"]["top3"]["f1"],
            "format_compliance_mean": scores["format_compliance_rate"]["mean"],
        }
        for field in FIELDS:
            for name in ("precision", "recall", "f1"):
                row[f"{field}_mean_{name}"] = scores["fields"][field]["mean"][name]
        scorecard.append(row)
    return sorted(scorecard, key=lambda row: (-float(row["overall_mean_f1"]), row["output_id"]))


def write_scorecard(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValidationError("cannot write an empty scorecard")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: Mapping[str, Any], scorecard: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Experiment 6 no-GPT-4.1 方法與結果獨立驗證",
        "",
        f"- status: `{report['status']}`",
        f"- confidence: `{report['confidence']}`",
        f"- case-runs / predictions: {report['coverage']['caseRuns']} / {report['coverage']['predictionRows']}",
        f"- judge checkpoints: {report['judge']['checkpointCalls']}",
        f"- semantic audit: {report['judge']['semanticAuditSampled']} / {report['judge']['semanticDecisionPopulation']}",
        "",
        "## 六欄正式比較",
        "",
        "- ObjectName：NFKC、trim、大小寫與空白正規化後一對一；未決 mention 由盲化 GPT-5.5 medium 判同一實體或明確共指。",
        "- DataName：string，trim＋lowercase 後硬比對。",
        "- Position：typed JSON 硬比對；陣列順序和值不可變，object key 順序可不同。",
        "- Trend：exact/版本化 alias；未決者由 judge 同核方向、期間、基準與範圍。",
        "- Num：僅 allowed absent 或有限 JSON number array；一對一 isclose(rel=abs=1e-9)。",
        "- Text：僅首尾空白、大小寫作 deterministic exact；未決者核完整命題、數字、時間、範圍、基準與否定。",
        "",
        "## 排名摘要",
        "",
        "| rank | output_id | mean macro P | mean macro R | mean macro F1 | top1 run | top1 F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(scorecard, start=1):
        lines.append(
            f"| {rank} | {row['output_id']} | {float(row['overall_mean_precision']):.6f} | "
            f"{float(row['overall_mean_recall']):.6f} | {float(row['overall_mean_f1']):.6f} | "
            f"{row['top1_run']} | {float(row['top1_f1']):.6f} |"
        )
    lines.extend([
        "",
        "完整逐 case 六欄 P/R/F1、top-3 與 format compliance 見 `experiment6_case_scorecard.tsv`；逐 run 與 pooled counts 見正式 TSV。",
        "",
    ])
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    finalizer_status_path = output_root / "runtime" / "no_gpt41_finalizer_status.json"
    finalizer_status = read_json(finalizer_status_path)
    if finalizer_status.get("status") != "completed":
        raise ValidationError("finalizer status is not completed")
    expected_ids = {str(value) for value in finalizer_status["expectedOutputIds"]}
    if len(expected_ids) != int(config["expectedOfficialCases"]):
        raise ValidationError("finalizer expected output count mismatch")
    normalize_id = lambda value: "".join(
        character for character in str(value).casefold() if character.isalnum()
    )
    excluded = {normalize_id(value) for value in config["excludedSourceIds"]}
    if any(any(value in normalize_id(output_id) for value in excluded) for output_id in expected_ids):
        raise ValidationError("excluded model appears in expected output IDs")
    expected_runs = int(config["expectedRuns"])
    expected_rows = int(config["expectedRows"])
    required_chatmock_model = str(config["evaluation"]["judge"]["model"])
    required_reasoning_effort = str(
        config["evaluation"]["judge"]["reasoningEffort"]
    )
    data = validate_data(config, expected_rows)
    coverage = validate_manifests(
        output_root,
        expected_ids,
        expected_runs,
        expected_rows,
        {str(value) for value in config["excludedSourceIds"]},
        required_chatmock_model,
        required_reasoning_effort,
    )
    if coverage["predictionRows"] != int(config["expectedFormalPredictions"]):
        raise ValidationError("formal prediction count mismatch")
    results_path = output_root / "experiment6_results.json"
    results = read_json(results_path)
    scorecard = validate_results(output_root, results, expected_ids, expected_runs)
    judge = validate_judge(
        output_root,
        expected_runs,
        float(config["evaluation"]["judge"]["minimumConfidence"]),
    )
    per_run_path = output_root / "experiment6_per_run_scores.tsv"
    field_summary_path = output_root / "experiment6_field_summary.tsv"
    if sum(1 for _ in per_run_path.open(encoding="utf-8")) - 1 != len(expected_ids) * expected_runs:
        raise ValidationError("per-run TSV coverage mismatch")
    if sum(1 for _ in field_summary_path.open(encoding="utf-8")) - 1 != len(expected_ids) * len(FIELDS):
        raise ValidationError("field-summary TSV coverage mismatch")
    evaluation_root = output_root / "evaluation_v4_no_gpt41"
    scorecard_path = output_root / "experiment6_case_scorecard.tsv"
    unit_test_path = output_root / "runtime" / "evaluator_unit_tests.log"
    if not unit_test_path.is_file() or "Ran 43 tests" not in unit_test_path.read_text(
        encoding="utf-8"
    ):
        raise ValidationError("saved 43-test evaluator log is missing or incomplete")
    write_scorecard(scorecard_path, scorecard)
    report = {
        "time": utc_now(),
        "protocol": "experiment6-no-gpt41-independent-validation-v1",
        "status": "passed",
        "confidence": "high_with_llm_judge_variability_caveat",
        "coverage": {key: value for key, value in coverage.items() if key != "manifests"},
        "data": data,
        "judge": judge,
        "method": {
            "fields": list(FIELDS),
            "hardFields": ["DataName", "Position"],
            "semanticFields": ["ObjectName", "Trend", "Text"],
            "numericField": "Num",
            "recomputedFromCounts": True,
            "commonTopRunsVerified": True,
            "rejectedRowsIncluded": True,
            "runtimeBlockedRows": 0,
        },
        "artifacts": {
            "results": str(results_path),
            "resultsSha256": sha256_file(results_path),
            "perRunScores": str(per_run_path),
            "perRunScoresSha256": sha256_file(per_run_path),
            "fieldSummary": str(field_summary_path),
            "fieldSummarySha256": sha256_file(field_summary_path),
            "caseScorecard": str(scorecard_path),
            "caseScorecardSha256": sha256_file(scorecard_path),
            "config": str(config_path),
            "configSha256": sha256_file(config_path),
            "validator": str(Path(__file__).resolve()),
            "validatorSha256": sha256_file(Path(__file__).resolve()),
            "evaluatorUnitTests": str(unit_test_path),
            "evaluatorUnitTestsSha256": sha256_file(unit_test_path),
        },
        "caveats": [
            "ObjectName, Trend, and Text unresolved cases use a stochastic LLM judge; 10% swapped-order re-judging and third adjudication quantify but do not eliminate this variance.",
            "completed_with_format_errors are model-quality outcomes scored by rejected-zero, not runtime failures.",
        ],
    }
    report_path = evaluation_root / "method_validation_report.json"
    markdown_path = evaluation_root / "method_validation_report.md"
    write_json(report_path, report)
    markdown_path.write_text(markdown_report(report, scorecard), encoding="utf-8")
    return {**report, "report": str(report_path), "markdown": str(markdown_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = build(parse_args(argv))
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
