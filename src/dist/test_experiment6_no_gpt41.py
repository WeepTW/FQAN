#!/usr/bin/env python3
"""Scope-contract tests for the 34-case Experiment 6 evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_narrative2_hybrid_v4_no_gpt41 as evaluator
import experiment6_no_gpt41 as orchestrator
import experiment6_v4 as legacy_orchestrator


CONFIG_PATH = REPO_ROOT / "config" / "experiment6_narrative2_hybrid_v4_no_gpt41.json"
LEGACY_CONFIG_PATH = REPO_ROOT / "config" / "experiment6_narrative2_hybrid_v4.json"


class Experiment6NoGpt41Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = orchestrator.load_config(CONFIG_PATH)
        cls.snapshot = orchestrator.effective_generation_config(cls.config)
        legacy_config = legacy_orchestrator.load_config(LEGACY_CONFIG_PATH)
        cls.legacy_snapshot = legacy_orchestrator.effective_generation_config(
            legacy_config
        )
        cls.excluded = set(cls.config["excludedSourceIds"])
        cls.outputs = evaluator.expected_output_sources(cls.snapshot, cls.excluded)

    def test_native_and_legacy_snapshot_sizes_are_explicitly_allowed(self) -> None:
        self.assertEqual(self.snapshot["expectedOfficialCases"], 34)
        self.assertEqual(self.legacy_snapshot["expectedOfficialCases"], 38)
        self.assertEqual(
            self.config["generationSnapshotExpectedOfficialCases"], 38
        )
        self.assertEqual(
            set(self.config["generationSnapshotAllowedOfficialCases"]), {34, 38}
        )
        legacy_outputs = evaluator.expected_output_sources(
            self.legacy_snapshot, self.excluded
        )
        self.assertEqual(legacy_outputs, self.outputs)

    def test_locator_objects_resolve_without_string_casting(self) -> None:
        config, evaluation = evaluator.load_effective_config(CONFIG_PATH)
        self.assertEqual(config["expectedOfficialCases"], 34)
        self.assertTrue(
            evaluator.workspace_path(config["baseEvaluationConfig"]).is_file()
        )
        self.assertTrue(
            evaluator.workspace_path(evaluation["goldPath"]).is_file()
        )

    def test_historical_judge_path_relocates_by_basename_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relocated = root / "judge_examples" / "prefix.txt"
            relocated.parent.mkdir(parents=True)
            relocated.write_text("prompt", encoding="utf-8")
            resolved = evaluator.resolve_judge_manifest_path(
                root, {"path": "/tmp/previous/judge_examples/prefix.txt"}
            )
            self.assertEqual(resolved, relocated)

    def test_no_gpt41_matrix_is_exactly_9_12_13(self) -> None:
        self.assertEqual(self.config["expectedOfficialCases"], 34)
        self.assertEqual(self.config["expectedFormalPredictions"], 34 * 10 * 85)
        self.assertEqual(len(self.outputs), 34)
        self.assertNotIn("gpt4_1", set(self.outputs.values()))
        self.assertFalse(any("gpt4.1" in output_id for output_id in self.outputs))
        by_part = {1: 0, 2: 0, 3: 0}
        for part in self.snapshot["parts"]:
            count = 0
            for case in part.get("cases", []):
                count += case["sourceId"] not in self.excluded
            for model in part.get("models", []):
                if model["sourceId"] not in self.excluded:
                    count += len(part.get("promptModes", []))
            if count:
                by_part[int(part["part"])] = count
        self.assertEqual(by_part, {1: 9, 2: 12, 3: 13})
        self.assertEqual(orchestrator.expected_route_ids(self.snapshot), {
            "finqa_flan_z", "finqa_flan_m", "finqa_flan_d",
            "finqa_mistral_z", "finqa_mistral_m", "finqa_mistral_d",
            "finqa_t5gemma2_z", "finqa_t5gemma2_m", "finqa_t5gemma2_d",
            "mistral_v0_3", "flan_t5_large", "t5gemma_2_1b_1b", "gpt5_5",
        })

    def test_position_values_are_strict_but_data_name_is_trim_lowercased(self) -> None:
        gold = [{
            "ObjectName": ["Revenue"],
            "DataName": "Revenue",
            "Position": [{"Begin": ["A", 1], "End": ["A", 1]}],
            "Trend": "increase",
            "Num": [1.0],
            "Text": "Revenue increased.",
        }]
        prediction = {
            "formatValid": True,
            "result": [{
                **gold[0],
                "DataName": " revenue ",
                "Position": [{"Begin": ["a", 1], "End": ["A", 1]}],
            }],
        }
        self.assertTrue(any(
            "anchor_mismatch" in error
            for error in evaluator.row_gate(gold, prediction)
        ))
        prediction["result"][0]["Position"] = gold[0]["Position"]
        self.assertEqual(evaluator.row_gate(gold, prediction), [])
        self.assertFalse(evaluator.data_name_equal("Straße", "STRASSE"))
        self.assertFalse(evaluator.data_name_equal("Ｒevenue", "Revenue"))
        self.assertFalse(evaluator.data_name_equal("Net  Revenue", "Net Revenue"))
        self.assertTrue(evaluator.position_equal(
            [{"Begin": [1, 2], "End": [3, 4]}],
            [{"End": [3.0, 4], "Begin": [1, 2.0]}],
        ))
        self.assertFalse(evaluator.position_equal([1, 2], [2, 1]))
        self.assertFalse(evaluator.position_equal([True], [1]))

    def test_v4_object_and_text_exact_normalization_is_field_specific(self) -> None:
        gold = {
            "ObjectName": [" Net   Revenue "],
            "DataName": "Revenue",
            "Position": [{"Begin": [0, 1], "End": [0, 1]}],
            "Trend": "increase",
            "Num": [1.0],
            "Text": "Revenue increased.",
        }
        prediction = {
            **gold,
            "ObjectName": ["net revenue"],
            "Text": "  revenue INCREASED.  ",
        }
        plan = evaluator.build_semantic_plan("[]", gold, prediction, 0, {})
        self.assertEqual(plan["exactObjectPairs"], [(0, 0)])
        self.assertTrue(plan["textDeterministic"])
        internal_space = {**prediction, "Text": "Revenue  increased."}
        plan = evaluator.build_semantic_plan("[]", gold, internal_space, 0, {})
        self.assertFalse(plan["textDeterministic"])
        self.assertTrue(any(
            decision["field"] == "Text" for decision in plan["decisions"]
        ))
        unicode_fold = {**prediction, "Text": "Revenue STRASSE."}
        unicode_gold = {**gold, "Text": "Revenue Straße."}
        plan = evaluator.build_semantic_plan("[]", unicode_gold, unicode_fold, 0, {})
        self.assertFalse(plan["textDeterministic"])

    def test_v4_object_matching_is_one_to_one_and_trend_alias_is_versioned(self) -> None:
        gold = {
            "ObjectName": ["Revenue", "Revenue"],
            "DataName": "Revenue",
            "Position": [],
            "Trend": "increase",
            "Num": [],
            "Text": "Revenue increased.",
        }
        prediction = {**gold, "ObjectName": ["revenue"], "Trend": "rose"}
        aliases = {"increase": "increase", "rose": "increase", "decrease": "decrease"}
        plan = evaluator.build_semantic_plan("[]", gold, prediction, 0, aliases)
        self.assertEqual(len(plan["exactObjectPairs"]), 1)
        self.assertEqual(plan["exactObjectPairs"][0][1], 0)
        self.assertFalse(evaluator.object_field_pass(plan, {})[0])
        self.assertTrue(plan["trendDeterministic"])
        different = {**prediction, "Trend": "decrease"}
        plan = evaluator.build_semantic_plan("[]", gold, different, 0, aliases)
        self.assertIsNone(plan["trendDeterministic"])
        self.assertTrue(any(
            decision["field"] == "Trend" for decision in plan["decisions"]
        ))

    def test_v4_absent_contract_rejects_literal_null_and_missing_object(self) -> None:
        self.assertTrue(evaluator.is_absent(None))
        self.assertTrue(evaluator.is_absent(" NoNe "))
        self.assertTrue(evaluator.is_absent([None, " ", ["None"]]))
        self.assertFalse(evaluator.is_absent("null"))
        self.assertIsNone(evaluator.strict_numeric_array("null"))
        self.assertTrue(evaluator.numeric_equal([2, 1.0], [1, 2.0], 1e-9, 1e-9))
        self.assertFalse(evaluator.numeric_equal([2.0], ["2"], 1e-9, 1e-9))
        self.assertFalse(evaluator.numeric_equal([1], [True], 1e-9, 1e-9))
        self.assertFalse(evaluator.numeric_equal([1, 1], [1], 1e-9, 1e-9))
        self.assertFalse(evaluator.numeric_equal([float("nan")], [float("nan")], 1e-9, 1e-9))
        invalid = {
            "ObjectName": ["None"],
            "DataName": "Revenue",
            "Position": [],
            "Trend": None,
            "Num": [],
            "Text": "Revenue.",
        }
        self.assertTrue(any(
            "ObjectName" in error
            for error in evaluator.validate_prediction_binding(invalid, "prediction[0]")
        ))

    def test_native_contract_tracks_no_gpt41_orchestrator_and_finalizer(self) -> None:
        contract = orchestrator.build_contract(
            CONFIG_PATH, self.config, self.snapshot
        )
        self.assertEqual(contract["expectedOfficialCases"], 34)
        self.assertEqual(contract["expectedFormalPredictions"], 28900)
        self.assertEqual(
            Path(contract["files"]["orchestrator"]["path"]).name,
            "experiment6_no_gpt41.py",
        )
        self.assertEqual(
            Path(contract["files"]["finalizer"]["path"]).name,
            "experiment6_no_gpt41_finalize.py",
        )

    def test_scope_gate_rejects_gpt41_manifest(self) -> None:
        with self.assertRaisesRegex(evaluator.ProtocolError, "excluded-source"):
            evaluator.validate_manifest_scope(
                [{"outputId": "6_gpt4.1_z", "sourceId": "gpt4_1"}],
                self.outputs,
                self.excluded,
            )

    def test_scope_gate_rejects_mislabeled_source(self) -> None:
        with self.assertRaisesRegex(evaluator.ProtocolError, "outside the 34-case"):
            evaluator.validate_manifest_scope(
                [{"outputId": "6_flan_z", "sourceId": "mistral_v0_3"}],
                self.outputs,
                self.excluded,
            )

    def test_scope_gate_accepts_all_expected_output_source_pairs(self) -> None:
        evaluator.validate_manifest_scope(
            [
                {"outputId": output_id, "sourceId": source_id}
                for output_id, source_id in self.outputs.items()
            ],
            self.outputs,
            self.excluded,
        )

    def test_flat_score_tables_include_all_runs_fields_and_text(self) -> None:
        runs = []
        for run_id in range(1, 11):
            metrics = {
                field: {
                    "tp": run_id,
                    "fp": 10 - run_id,
                    "fn": 10 - run_id,
                    "precision": run_id / 10,
                    "recall": run_id / 10,
                    "f1": run_id / 10,
                }
                for field in evaluator.FIELDS
            }
            runs.append({
                "run": run_id,
                "macro": {
                    "precision": run_id / 10,
                    "recall": run_id / 10,
                    "f1": run_id / 10,
                },
                "fields": metrics,
                "fieldMetrics": metrics,
                "format_compliance_rate": 1.0,
                "accepted_rows": 85,
                "rejected_rows": 0,
                "overall_micro": {},
                "semantic_audit": {},
            })
        top = list(reversed(runs[-3:]))
        aggregate = {
            "model": {
                "output_id": "6_flan_z",
                "requested": "finqa_flan_z",
                "actual": ["google/flan-t5-large"],
            },
            "prompt": {
                "part": 1,
                "mode": "zero-shot",
                "route": "adapter-converter",
            },
            "runtime": {
                "runs": [
                    {
                        "run": run_id,
                        "seed": 100 + run_id,
                        "status": "completed",
                        "runtime_seconds": 1.0,
                    }
                    for run_id in range(1, 11)
                ]
            },
            "scores": {
                "runs": runs,
                "fields": {
                    field: evaluator.metric_summary(runs, field, top)
                    for field in evaluator.FIELDS
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = evaluator.write_flat_score_tables(
                Path(temporary), [aggregate]
            )
            per_run = Path(report["perRun"]["path"])
            summary = Path(report["fieldSummary"]["path"])
            self.assertEqual(len(per_run.read_text(encoding="utf-8").splitlines()), 11)
            self.assertEqual(len(summary.read_text(encoding="utf-8").splitlines()), 7)
            self.assertIn("Text_f1", per_run.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("\tText\t", "\n" + summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
