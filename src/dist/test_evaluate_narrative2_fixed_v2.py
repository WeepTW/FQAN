#!/usr/bin/env python3
"""Tests for the Experiment 6 narrative fixed-v2 wrapper."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_narrative2_fixed_v2 as evaluator


CONFIG_PATH = (
    evaluator.REPO_ROOT
    / "config"
    / "experiment6_narrative2_fixed_evaluation.json"
)
GENERATION_CONFIG_PATH = (
    evaluator.REPO_ROOT
    / "config"
    / "experiment6_narrative2_generation.json"
)


class Narrative2FixedV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = evaluator.load_config(CONFIG_PATH)
        cls.fixed = evaluator.load_module(
            evaluator.workspace_path(cls.config["referenceEvaluator"]),
            "fixed_v2_test_reference",
        )
        cls.legacy = evaluator.load_module(
            evaluator.SCRIPTS_ROOT / "evaluate_data_binding.py",
            "fixed_v2_test_legacy",
        )
        cls.targets = cls.fixed._extract_targets(
            cls.fixed.load_json(
                evaluator.workspace_path(cls.config["goldPath"])
            )
        )
        cls.vocabulary, _ = cls.fixed._load_vocabulary(
            evaluator.workspace_path(cls.config["trendVocabularyPath"]),
            evaluator.workspace_path(cls.config["goldPath"]),
        )
        cls.generation_config = evaluator.read_json(GENERATION_CONFIG_PATH)
        cls.cases = evaluator.expand_matrix(cls.generation_config)

    def create_run(
        self,
        root: Path,
        mutate=None,
        generation_status: str = "completed",
        runtime_blocked_rows: int = 0,
    ) -> tuple[dict, Path]:
        case = next(item for item in self.cases if item["outputId"] == "6_flan_z")
        rows = [
            {
                "source": target["source"],
                "run": 1,
                "result": copy.deepcopy(target["targetBindings"]),
            }
            for target in self.targets
        ]
        if mutate is not None:
            mutate(rows)
        run_dir = root / "cases" / case["outputId"] / "run_01"
        predictions = run_dir / "predictions.jsonl"
        evaluator.write_jsonl(predictions, rows)
        status = {
            "protocol": "experiment6-narrative2-full-v2",
            "outputId": case["outputId"],
            "run": 1,
            "status": generation_status,
            "requestedModel": "finqa_flan_z",
            "actualModel": "google/flan-t5-large",
            "adapter": "adapter",
            "runtimeProfile": "test",
            "quantization": None,
            "runtimeSeconds": 1.0,
            "runtimeBlockedRows": runtime_blocked_rows,
            "formatComplianceRate": 1.0,
            "hashes": {"predictions": evaluator.sha256_file(predictions)},
        }
        evaluator.write_json(run_dir / "status.json", status)
        return case, predictions

    def run_once(self, root: Path, case: dict) -> dict:
        return evaluator.evaluate_run(
            self.fixed,
            self.legacy,
            self.config,
            root,
            case,
            1,
            self.targets,
            self.vocabulary,
        )

    def test_matrix_is_exact_54_plus_4(self) -> None:
        official = [case for case in self.cases if case["official"]]
        controls = [case for case in self.cases if not case["official"]]
        self.assertEqual(len(official), 54)
        self.assertEqual(len(controls), 4)
        self.assertEqual(
            Counter(case["part"] for case in official),
            Counter({1: 9, 2: 24, 3: 17, 4: 4}),
        )

    def test_reference_comparison_semantics_remain_fixed(self) -> None:
        self.assertTrue(self.fixed.same_fixed(" CPI ", "cpi"))
        self.assertFalse(self.fixed.same_fixed("C P I", "CPI"))
        self.assertTrue(self.fixed.same_fixed(12, 12.0))
        self.assertFalse(self.fixed.same_fixed("12", 12))
        self.assertFalse(self.fixed.same_fixed([1, 2], [2, 1]))
        self.assertTrue(
            self.fixed.same_fixed({"b": 2, "a": 1}, {"a": 1, "b": 2})
        )

    def test_perfect_run_has_full_fixed_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, _ = self.create_run(root)
            result = self.run_once(root, case)
            metrics = result["metrics"]
            self.assertEqual(metrics["acceptedRows"], 85)
            self.assertEqual(metrics["rejectedRows"], 0)
            self.assertEqual(metrics["allFieldsCombined"]["tested"], 173 * 6)
            self.assertEqual(metrics["allFieldsCombined"]["f1"], 1.0)
            self.assertEqual(metrics["formatComplianceRate"], 1.0)

    def test_binding_count_rejection_scores_target_fields_zero(self) -> None:
        def mutate(rows):
            rows[0]["result"] = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, _ = self.create_run(root, mutate)
            metrics = self.run_once(root, case)["metrics"]
            self.assertEqual(metrics["acceptedRows"], 84)
            self.assertEqual(metrics["rejectedRows"], 1)
            self.assertEqual(metrics["formatComplianceRate"], 1.0)
            self.assertAlmostEqual(
                metrics["fixedProtocolAcceptanceRate"], 84 / 85
            )
            self.assertEqual(metrics["allFieldsCombined"]["tested"], 173 * 6)
            self.assertEqual(
                metrics["allFieldsCombined"]["passes"], 173 * 6 - 2 * 6
            )
            self.assertAlmostEqual(
                metrics["allFieldsCombined"]["f1"],
                (173 * 6 - 12) / (173 * 6),
            )

    def test_numeric_string_and_binding_reordering_are_rejected(self) -> None:
        def numeric_string(rows):
            rows[0]["result"][0]["Num"] = ["12"]

        def reorder(rows):
            rows[0]["result"].reverse()

        for mutation in (numeric_string, reorder):
            with self.subTest(mutation=mutation.__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    case, _ = self.create_run(root, mutation)
                    metrics = self.run_once(root, case)["metrics"]
                    self.assertEqual(metrics["acceptedRows"], 84)
                    self.assertLess(metrics["allFieldsCombined"]["f1"], 1.0)

    def test_position_order_mutation_reduces_score(self) -> None:
        def mutate(rows):
            first = rows[1]["result"][0]["Position"]
            first.append(copy.deepcopy(first[0]))
            first.reverse()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, _ = self.create_run(root, mutate)
            metrics = self.run_once(root, case)["metrics"]
            self.assertLess(metrics["allFieldsCombined"]["f1"], 1.0)

    def test_runtime_block_is_scored_but_withholds_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, _ = self.create_run(
                root,
                generation_status="runtime_blocked",
                runtime_blocked_rows=1,
            )
            result = self.run_once(root, case)
            self.assertEqual(
                result["metrics"]["status"], "runtime_blocked_scored_zero"
            )
            run_results = []
            for run_number in range(1, 11):
                item = copy.deepcopy(result)
                item["metrics"]["run"] = run_number
                item["generationStatus"]["run"] = run_number
                run_results.append(item)
            aggregate = evaluator.aggregate_case(
                self.fixed, self.config, root, case, run_results
            )
            self.assertEqual(
                aggregate["scores"]["completion_status"],
                "runtime_blocked_no_ranking",
            )
            self.assertEqual(
                aggregate["scores"]["all_10"]["allFieldsCombined"]["f1"][
                    "sampleSd"
                ],
                0.0,
            )
            self.assertEqual(len(aggregate["scores"]["top_3"]["runs"]), 3)

    def test_ten_run_threshold_contract(self) -> None:
        threshold = self.config["targetThreshold"]
        self.assertEqual(threshold["minimumPassesForTenRuns"], 9)
        self.assertGreater(9 / 10, threshold["rate"])
        self.assertFalse(8 / 10 > threshold["rate"])


if __name__ == "__main__":
    unittest.main()
