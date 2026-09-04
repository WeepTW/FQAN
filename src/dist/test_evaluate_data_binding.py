#!/usr/bin/env python3
"""Smoke tests for Experiment 6 data-binding evaluator scoring policy."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import evaluate_data_binding as evaluator


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_score_from_counts_empty_set_is_perfect() -> None:
    assert evaluator.score_from_counts(0, 0, 0)["precision"] == 1.0
    assert evaluator.score_from_counts(0, 0, 0)["recall"] == 1.0
    assert evaluator.score_from_counts(0, 0, 0)["f1"] == 1.0
    assert evaluator.score_from_counts(0, 0, 3)["f1"] == 0.0
    assert evaluator.score_from_counts(0, 2, 0)["f1"] == 0.0
    assert evaluator.score_from_counts(0, 2, 3)["f1"] == 0.0


def test_empty_result_matches_empty_prediction() -> None:
    gold = evaluator.extract_rows([{"case_id": "empty", "result": []}], strict=True)
    pred = evaluator.extract_rows([{"case_id": "empty", "result": []}], strict=False)
    metrics = evaluator.metrics_from_extracted(gold, pred, list(evaluator.TYPE_NAMES))
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_malformed_prediction_is_scored_as_empty_prediction() -> None:
    gold = evaluator.extract_rows(
        [{"case_id": "a", "result": [{"ObjectName": ["Revenue"], "Trend": "increase", "Num": ["3%"]}]}],
        strict=True,
    )
    pred = evaluator.extract_rows([{"case_id": "a", "result": "not-json"}], strict=False)
    metrics = evaluator.metrics_from_extracted(gold, pred, list(evaluator.TYPE_NAMES))
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 3
    assert metrics["f1"] == 0.0


def test_duplicate_prediction_case_id_blocks_formal_status() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        gold_jsonl = tmp / "gold.jsonl"
        pred_jsonl = tmp / "pred.jsonl"
        metrics_json = tmp / "out" / "metrics.json"
        status_json = tmp / "out" / "status.json"
        write_jsonl(gold_jsonl, [{"case_id": "a", "result": []}, {"case_id": "b", "result": []}])
        write_jsonl(pred_jsonl, [{"case_id": "a", "result": []}, {"case_id": "a", "result": []}])
        payload, exit_code = evaluator.build_status(
            SimpleNamespace(
                experiment_id="case_duplicate",
                source_id="unit",
                narrative_route="unit",
                gold_jsonl=gold_jsonl,
                pred_jsonl=pred_jsonl,
                metrics_json=metrics_json,
                status_json=status_json,
                vocabulary_types=list(evaluator.TYPE_NAMES),
                require_data=True,
                allow_controlled_predictions=False,
            )
        )
        metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
        assert exit_code == 2
        assert payload["status"] == "runtime_blocked"
        assert payload["metrics"] is None
        assert payload["failure_category"] == "invalid_binding_eval_input"
        assert "duplicate case_id" in " ".join(payload["blockers"])
        assert metrics_payload["status"] == "not_scored"


def test_existing_empty_jsonl_blocks_when_require_data() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        gold_jsonl = tmp / "gold.jsonl"
        pred_jsonl = tmp / "pred.jsonl"
        metrics_json = tmp / "out" / "metrics.json"
        status_json = tmp / "out" / "status.json"
        gold_jsonl.write_text("", encoding="utf-8")
        pred_jsonl.write_text("", encoding="utf-8")
        payload, exit_code = evaluator.build_status(
            SimpleNamespace(
                experiment_id="empty_files",
                source_id="unit",
                narrative_route="unit",
                gold_jsonl=gold_jsonl,
                pred_jsonl=pred_jsonl,
                metrics_json=metrics_json,
                status_json=status_json,
                vocabulary_types=list(evaluator.TYPE_NAMES),
                require_data=True,
                allow_controlled_predictions=False,
            )
        )
        assert exit_code == 2
        assert payload["metrics"] is None
        assert "gold JSONL is empty" in payload["blockers"]
        assert "prediction JSONL is empty" in payload["blockers"]


def test_invalid_prediction_metadata_blocks_formal_status() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        gold_jsonl = tmp / "gold.jsonl"
        pred_jsonl = tmp / "pred.jsonl"
        metrics_json = tmp / "out" / "metrics.json"
        status_json = tmp / "out" / "status.json"
        write_jsonl(gold_jsonl, [{"case_id": "a", "result": []}])
        write_jsonl(pred_jsonl, [{"case_id": "a", "result": []}])
        pred_jsonl.with_suffix(pred_jsonl.suffix + ".metadata.json").write_text("{bad-json", encoding="utf-8")
        payload, exit_code = evaluator.build_status(
            SimpleNamespace(
                experiment_id="bad_metadata",
                source_id="unit",
                narrative_route="unit",
                gold_jsonl=gold_jsonl,
                pred_jsonl=pred_jsonl,
                metrics_json=metrics_json,
                status_json=status_json,
                vocabulary_types=list(evaluator.TYPE_NAMES),
                require_data=True,
                allow_controlled_predictions=False,
            )
        )
        assert exit_code == 2
        assert payload["metrics"] is None
        assert payload["failure_category"] == "invalid_binding_eval_input"
        assert "metadata is invalid JSON" in " ".join(payload["blockers"])


def test_run_aggregate_records_missing_run_without_crashing() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        run_ok = tmp / "run_01.jsonl"
        run_missing = tmp / "run_02.jsonl"
        write_jsonl(run_ok, [{"case_id": "empty", "result": []}])
        gold = evaluator.extract_rows([{"case_id": "empty", "result": []}], strict=True)
        aggregate = evaluator.run_aggregate_metrics(
            {"run_prediction_jsonls": [str(run_ok), str(run_missing)], "top_k": 3},
            gold,
            list(evaluator.TYPE_NAMES),
        )
        assert aggregate is not None
        assert aggregate["completed_runs"] == 1
        assert aggregate["run_errors"] == 1
        assert aggregate["official_average"]["f1"] == 1.0
        assert aggregate["official_average"]["denominator_runs"] == 1
        assert aggregate["penalized_average"]["f1"] == 0.5
        assert aggregate["penalized_average"]["denominator_runs"] == 2
        assert aggregate["top_3_best_runs"]["runs"][0]["status"] == "completed"


def main() -> None:
    tests = [
        test_score_from_counts_empty_set_is_perfect,
        test_empty_result_matches_empty_prediction,
        test_malformed_prediction_is_scored_as_empty_prediction,
        test_duplicate_prediction_case_id_blocks_formal_status,
        test_existing_empty_jsonl_blocks_when_require_data,
        test_invalid_prediction_metadata_blocks_formal_status,
        test_run_aggregate_records_missing_run_without_crashing,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} evaluate_data_binding tests")


if __name__ == "__main__":
    main()
