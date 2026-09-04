#!/usr/bin/env python3
"""Regression tests for diagnostic Experiment 6 Binding-candidate evaluation."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import build_experiment6_binding_candidate_score_tables as candidate_tables
import evaluate_experiment6_binding_candidates_v1 as candidate
import evaluate_narrative2_reference_aligned_v6 as v601
import evaluate_narrative2_reference_aligned_v6_0_2 as v602
import evaluate_narrative2_reference_aligned_v6_1 as v610


def valid_manifest() -> dict[str, object]:
    return {
        "protocol": candidate.EXPECTED_CANDIDATE_PROTOCOL,
        "official": False,
        "diagnosticOnly": True,
        "claimEligible": False,
        "goldAccessed": False,
        "status": "completed_diagnostic_binding_candidates",
    }


def binding() -> dict[str, object]:
    return {
        "ObjectName": ["United States"],
        "DataName": "Revenue",
        "Position": [{"Begin": [1, 1], "End": [1, 1]}],
        "Trend": "None",
        "Num": [12],
        "Text": "Revenue was 12.",
    }


def run_result(module, run: int, tp: int) -> dict[str, object]:
    counts = {
        field: {"tp": tp, "fp": 1 - tp, "fn": 1 - tp}
        for field in module.PRIMARY_FIELDS
    }
    without = {field: value for field, value in counts.items() if field != "Trend"}
    return {
        "outputId": "case_a",
        "run": run,
        "primary": module.metrics_from_counts(counts),
        "bindingLevel": module.metric(tp=tp, fp=1 - tp, fn=1 - tp),
        "withoutTrendAblation": module.metrics_from_counts(without),
        "coverage": {"gold_bindings": 1, "matched_bindings": tp, "empty_output_rows": 1 - tp},
        "trend": {"support": {label: 0 for label in module.TREND_CLASSES}, "confusionMatrix": {}},
    }


class CandidateContractTests(unittest.TestCase):
    def test_rejects_official_claim_eligible_or_gold_accessed_manifests(self) -> None:
        candidate.validate_candidate_manifest(valid_manifest())
        for key, value in (("official", True), ("claimEligible", True), ("goldAccessed", True)):
            manifest = valid_manifest()
            manifest[key] = value
            with self.assertRaises(candidate.CandidateEvaluationError):
                candidate.validate_candidate_manifest(manifest)

    def test_v602_changes_binding_fp_but_not_field_scores(self) -> None:
        config_601 = json.loads((DIST.parent / "config/experiment6_narrative2_evaluation_v6.json").read_text(encoding="utf-8"))
        config_602 = json.loads((DIST.parent / "config/experiment6_narrative2_evaluation_v6_0_2.json").read_text(encoding="utf-8"))
        target = {"source": "s1", "targetBindings": [binding()]}
        prediction = {"source": "s1", "formatValid": False, "result": [], "rawResponse": "not empty"}
        summary_601, _ = v601.evaluate_rows(
            [target], [prediction], v601.ObjectMatcher(config_601["objectName"]), v601.TrendClassifier(config_601["trend"], allow_model=False)
        )
        summary_602, _ = v602.evaluate_rows(
            [target], [prediction], v602.ObjectMatcher(config_602["objectName"]), v602.TrendClassifier(config_602["trend"], allow_model=False)
        )
        self.assertEqual(summary_601["primary"], summary_602["primary"])
        self.assertEqual(summary_601["bindingLevel"]["fp"], 0)
        self.assertEqual(summary_602["bindingLevel"]["fp"], 1)

    def test_v610_repairs_only_evidence_backed_content_mismatches(self) -> None:
        config = json.loads(
            (DIST.parent / "config/experiment6_narrative2_evaluation_v6_1.json").read_text(
                encoding="utf-8"
            )
        )
        v610.configure(config)
        gold = {
            "ObjectName": ["The Gilt crisis"],
            "DataName": "Robocalls",
            "Position": [{"Begin": [1, 1], "End": [1, 1]}],
            "Trend": "increase",
            "Num": [360],
            "Text": "Robocalls increased to 360 million per month.",
        }
        prediction = {
            "ObjectName": ["Gilt crisis associated with September 2022"],
            "DataName": "Robocalls",
            "Position": [{"Begin": [1, 1], "End": [1, 1]}],
            "Trend": "increased over the period",
            "Num": ["360 million per month"],
            "Text": "Robocalls increased to 360 million per month.",
        }
        summary, _ = v610.evaluate_rows(
            [{"source": "s1", "targetBindings": [gold]}],
            [{"source": "s1", "formatValid": True, "result": [prediction]}],
            v610.ObjectMatcher(config["objectName"]),
            v610.TrendClassifier(config["trend"], allow_model=False),
        )
        for field in v610.PRIMARY_FIELDS:
            self.assertEqual(
                summary["primary"]["counts"][field], {"tp": 1, "fp": 0, "fn": 0}
            )
        self.assertEqual(
            summary["methodAudit"]["counts"]["object_article_normalized"], 1
        )
        self.assertEqual(
            summary["methodAudit"]["counts"]["num_semantic_normalized"], 1
        )

        wrong_direction = copy.deepcopy(prediction)
        wrong_direction["Trend"] = "declined over the period"
        wrong_summary, _ = v610.evaluate_rows(
            [{"source": "s1", "targetBindings": [gold]}],
            [{"source": "s1", "formatValid": True, "result": [wrong_direction]}],
            v610.ObjectMatcher(config["objectName"]),
            v610.TrendClassifier(config["trend"], allow_model=False),
        )
        self.assertEqual(
            wrong_summary["primary"]["counts"]["Trend"],
            {"tp": 0, "fp": 1, "fn": 1},
        )
        self.assertEqual(wrong_summary["primary"]["fields"]["Trend"]["f1"], 0.0)
        self.assertTrue(
            v610.semantic_numeric_equal([3.64], [["3.64"]], "Fed liquidity")[0]
        )
        self.assertTrue(v610.semantic_numeric_equal([23], ["23rd"], "Rank")[0])
        self.assertFalse(v610.semantic_numeric_equal([100], [100.5], "Value", 0.0)[0])
        self.assertTrue(v610.semantic_numeric_equal([100], [100.5], "Value", 0.01)[0])

    def test_v610_method_evidence_is_hash_pinned(self) -> None:
        config = json.loads(
            (DIST.parent / "config/experiment6_narrative2_evaluation_v6_1.json").read_text(
                encoding="utf-8"
            )
        )
        v610.validate_config(config)
        config["methodEvidence"]["recommendationDocumentSha256"] = "0" * 64
        with self.assertRaises(v610.ProtocolError):
            v610.validate_config(config)

    def test_metrics_are_derived_only_from_tp_fp_fn(self) -> None:
        result = v610.metric(tp=3, fp=2, fn=1)
        self.assertEqual(result["precision"], 3 / 5)
        self.assertEqual(result["recall"], 3 / 4)
        self.assertEqual(result["f1"], 6 / 9)

    def test_shared_ranking_is_descriptive_and_common_to_all_fields(self) -> None:
        config = json.loads((DIST.parent / "config/experiment6_narrative2_evaluation_v6_0_2.json").read_text(encoding="utf-8"))
        runs = [run_result(v602, run, 1 if run in {3, 5, 7} else 0) for run in range(1, 11)]
        case = candidate.aggregate_candidate_case(v602, "case_a", runs, config)
        self.assertEqual(case["mode"], "diagnostic")
        self.assertFalse(case["claimEligible"])
        self.assertEqual(case["selection"]["role"], "diagnostic-descriptive-only")
        self.assertEqual(case["selection"]["sharedRunOrder"][:3], [3, 5, 7])
        self.assertEqual(case["selection"]["top3"]["runs"], [3, 5, 7])

    def test_runtime_requires_cpu_only_and_four_threads(self) -> None:
        original = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", *candidate.THREAD_ENVIRONMENTS)}
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            for key in candidate.THREAD_ENVIRONMENTS:
                os.environ[key] = "4"
            self.assertEqual(candidate.validate_runtime()["textJudge"], "disabled")
            os.environ["OMP_NUM_THREADS"] = "8"
            with self.assertRaises(candidate.CandidateEvaluationError):
                candidate.validate_runtime()
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_table_banner_column_order_and_text_na(self) -> None:
        config = json.loads((DIST.parent / "config/experiment6_narrative2_evaluation_v6_0_2.json").read_text(encoding="utf-8"))
        case = candidate.aggregate_candidate_case(
            v602,
            "case_a",
            [run_result(v602, run, 1) for run in range(1, 11)],
            config,
        )
        evaluation = {
            "scoringProtocol": v602.PROTOCOL,
            "scope": "candidate12",
            "time": "2026-08-14T00:00:00Z",
            "completedCases": 1,
            "protocol": candidate.PROTOCOL,
            "method": {"methodSha256": "a" * 64},
        }
        rows = [{
            "outputId": "case_a",
            "fineTunedMethod": "no-adaptor",
            "retrieverModel": "model",
            "inputPromptType": "original",
            "case": case,
        }]
        rendered = candidate_tables.render("top-1", rows, evaluation)
        self.assertIn("CLAIM-ELIGIBLE=false", rendered)
        self.assertIn("fine-tuned method (prompt-type or no-adaptor)", rendered)
        self.assertIn("**Subject** | **Trend** | **Num** | **Position** | **DataName** | **Text**", rendered)
        self.assertIn("NA (judge deferred)", rendered)
        self.assertIn("diagnostic", rendered.lower())
        self.assertIn("## Precision", rendered)
        self.assertIn("## Recall", rendered)
        self.assertIn("## F1", rendered)


if __name__ == "__main__":
    unittest.main()
