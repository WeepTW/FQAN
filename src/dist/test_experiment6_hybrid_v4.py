#!/usr/bin/env python3
"""Contract and data-quality tests for Experiment 6 hybrid-v4."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_experiment6_judge_examples_v4 as builder
import evaluate_narrative2_hybrid_v4 as evaluator
import experiment6_v4 as orchestrator


CONFIG_PATH = orchestrator.DEFAULT_CONFIG


def binding(
    *,
    object_name: object = None,
    data_name: object = "Revenue",
    position: object = None,
    trend: object = "increase",
    num: object = None,
    text: object = "Revenue increased to 2.",
) -> dict[str, object]:
    return {
        "ObjectName": ["Revenue"] if object_name is None else object_name,
        "DataName": data_name,
        "Position": (
            [{"Begin": [0, 1], "End": [1, 1]}]
            if position is None
            else position
        ),
        "Trend": trend,
        "Num": [2.0] if num is None else num,
        "Text": text,
    }


class Experiment6HybridV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = orchestrator.load_config(CONFIG_PATH)
        cls.generation = orchestrator.effective_generation_config(cls.config)

    def test_matrix_is_exact_9_12_13_4(self) -> None:
        self.assertEqual(self.generation["expectedOfficialCases"], 38)
        self.assertEqual(
            self.generation["expectedPartCounts"],
            {"1": 9, "2": 12, "3": 13, "4": 4},
        )
        excluded = set(self.config["excludedSourceIds"])
        source_ids = set()
        for part in self.generation["parts"]:
            for item in part.get("models", []) + part.get("cases", []):
                source_ids.add(item["sourceId"])
        self.assertTrue(excluded.isdisjoint(source_ids))
        self.assertEqual(self.generation["controls"], [])
        self.assertEqual(
            set(self.generation["directBinding"]["tokenizers"]),
            {"gpt5_5", "gpt4_1"},
        )
        self.assertTrue(all(
            role == "local-reproducible-proxy-no-truncation"
            for role in self.generation["directBinding"]["tokenizerRoles"].values()
        ))
        grouped_sources = [
            source_id
            for group in self.config["executionGroups"]
            for source_id in group["sourceIds"]
        ]
        self.assertEqual(len(grouped_sources), len(set(grouped_sources)))
        self.assertEqual(set(grouped_sources), source_ids)

    def test_formal_prediction_count(self) -> None:
        self.assertEqual(38 * 10 * 85, 32300)
        self.assertEqual(self.config["expectedFormalPredictions"], 32300)

    def test_judge_builder_produces_26_valid_rows_and_10_repair_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = builder.build(CONFIG_PATH, Path(temporary))
            self.assertEqual(manifest["validation"]["canonicalRows"], 26)
            self.assertEqual(manifest["validation"]["rowsWithRepairs"], 10)
            self.assertEqual(manifest["validation"]["canonicalBindings"], 55)
            examples = [
                json.loads(line)
                for line in (Path(temporary) / "canonical_examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(examples), 26)
            self.assertTrue(all(
                not builder.validate_binding(item, "test")
                for example in examples
                for item in example["canonicalBindings"]
            ))
            for field in ("ObjectName", "Trend", "Text"):
                prefix = (
                    Path(temporary) / f"judge_prompt_prefix_{field}.txt"
                ).read_text(encoding="utf-8")
                self.assertIn("resultText", prefix)

    def test_judge_uses_three_distinct_field_specific_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder.build(CONFIG_PATH, root / "judge_examples")
            _, report, prompts, versions = evaluator.configure_judge(
                root, self.config, root / "evaluation_v4"
            )
            expected = {"ObjectName", "Trend", "Text"}
            self.assertEqual(set(prompts), expected)
            self.assertEqual(set(versions), expected)
            self.assertEqual(len(set(prompts.values())), 3)
            self.assertEqual(len(set(versions.values())), 3)
            self.assertTrue(report["fieldSpecific"])
            self.assertEqual(set(report["fields"]), expected)

    def test_fixed_anchor_accepts_trim_and_case_only(self) -> None:
        gold = [binding(data_name="Revenue")]
        prediction = {
            "formatValid": True,
            "result": [binding(data_name=" revenue ")],
        }
        self.assertEqual(evaluator.row_gate(gold, prediction), [])

    def test_position_order_is_hard(self) -> None:
        gold_position = [
            {"Begin": [0, 1], "End": [0, 1]},
            {"Begin": [1, 1], "End": [1, 1]},
        ]
        prediction_position = list(reversed(gold_position))
        errors = evaluator.row_gate(
            [binding(position=gold_position)],
            {
                "formatValid": True,
                "result": [binding(position=prediction_position)],
            },
        )
        self.assertTrue(any("anchor_mismatch" in error for error in errors))

    def test_binding_reorder_is_rejected(self) -> None:
        left = binding(data_name="Revenue")
        right = binding(data_name="Profit", position=[{"Begin": [0, 2], "End": [0, 2]}])
        errors = evaluator.row_gate(
            [left, right],
            {"formatValid": True, "result": [right, left]},
        )
        self.assertTrue(any("anchor_mismatch" in error for error in errors))

    def test_numeric_is_finite_json_number_array_only(self) -> None:
        self.assertTrue(evaluator.numeric_equal([2, 1.0], [1, 2.0], 1e-9, 1e-9))
        self.assertFalse(evaluator.numeric_equal([2.0], ["2"], 1e-9, 1e-9))
        self.assertFalse(evaluator.numeric_equal([12.0], ["12%"], 1e-9, 1e-9))
        errors = evaluator.validate_prediction_binding(binding(num=["2"]), "binding")
        self.assertTrue(any("finite JSON number" in error for error in errors))

    def test_rejected_row_has_zero_tp_and_full_fp_fn(self) -> None:
        counts = evaluator.rejected_counts([binding(), binding()], [binding()])
        for field in evaluator.FIELDS:
            self.assertEqual(counts[field], {"tp": 0, "fp": 1, "fn": 2})

    def test_rejected_row_counts_only_parseable_prediction_fields(self) -> None:
        partial = {
            "DataName": "Revenue",
            "Position": "not-an-array",
            "Num": [1.0],
            "Text": 42,
        }
        counts = evaluator.rejected_counts([binding()], [partial, "bad binding"])
        self.assertEqual(counts["DataName"], {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(counts["Num"], {"tp": 0, "fp": 1, "fn": 1})
        for field in ("ObjectName", "Position", "Trend", "Text"):
            self.assertEqual(counts[field], {"tp": 0, "fp": 0, "fn": 1})

    def test_common_top_selection_reports_same_runs_for_every_field(self) -> None:
        runs = []
        for run_id in range(1, 11):
            score = run_id / 10
            runs.append({
                "run": run_id,
                "macro": {"precision": score, "recall": score, "f1": score},
                "fieldMetrics": {
                    field: {
                        "tp": run_id,
                        "fp": 10 - run_id,
                        "fn": 10 - run_id,
                        "precision": score,
                        "recall": score,
                        "f1": score,
                    }
                    for field in evaluator.FIELDS
                },
            })
        top = sorted(
            runs,
            key=lambda item: (
                -item["macro"]["f1"],
                -item["macro"]["precision"],
                -item["macro"]["recall"],
                item["run"],
            ),
        )[:3]
        self.assertEqual([item["run"] for item in top], [10, 9, 8])
        for field in evaluator.FIELDS:
            summary = evaluator.metric_summary(runs, field, top)
            self.assertEqual(summary["top1"]["run_id"], 10)
            self.assertEqual(summary["top3"]["run_ids"], [10, 9, 8])
            self.assertAlmostEqual(summary["mean"]["f1"], 0.55)
            self.assertEqual(summary["pooled_micro"]["tp"], 55)
            self.assertEqual(summary["pooled_micro"]["fp"], 45)

    def test_auto_discovery_excludes_diagnostic_roots(self) -> None:
        original_root = orchestrator.EXPERIMENT_ROOT
        try:
            with tempfile.TemporaryDirectory() as temporary:
                orchestrator.EXPERIMENT_ROOT = Path(temporary)
                prefix = self.config["rootPrefix"]
                diagnostic = Path(temporary) / f"{prefix}smoke_20260808T000000Z"
                formal = Path(temporary) / f"{prefix}20260808T000001Z"
                for path, role in ((diagnostic, "diagnostic"), (formal, "formal")):
                    path.mkdir()
                    (path / "experiment6_v4_contract.json").write_text(
                        json.dumps({"rootRole": role}), encoding="utf-8"
                    )
                self.assertEqual(
                    orchestrator.latest_incomplete(self.config), formal
                )
        finally:
            orchestrator.EXPERIMENT_ROOT = original_root

    def test_contract_fingerprint_changes_with_effective_matrix(self) -> None:
        first = orchestrator.contract(CONFIG_PATH, self.config, self.generation)
        changed = json.loads(json.dumps(self.generation))
        changed["seedBase"] += 1
        second = orchestrator.contract(CONFIG_PATH, self.config, changed)
        self.assertNotEqual(
            first["compatibilityFingerprint"],
            second["compatibilityFingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
