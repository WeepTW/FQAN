#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import evaluate_narrative2_reference_aligned_v6 as v601
import evaluate_narrative2_reference_aligned_v6_0_2 as v6


CONFIG = json.loads(
    (DIST.parent / "config" / "experiment6_narrative2_evaluation_v6_0_2.json").read_text(
        encoding="utf-8"
    )
)


def binding(**updates):
    value = {
        "ObjectName": ["United States"],
        "DataName": "Revenue – USD",
        "Position": [{"Begin": [1, 2], "End": [1, 2]}],
        "Trend": "None",
        "Num": [12.0],
        "Text": "United States revenue was 12.",
    }
    value.update(updates)
    return value


def row(target_bindings):
    return {"source": "S1", "targetBindings": target_bindings}


def prediction(result, format_valid=True, raw_response=None):
    value = {
        "source": "S1",
        "result": result,
        "formatValid": format_valid,
        "parserDiagnostic": {},
    }
    if raw_response is not None:
        value["rawResponse"] = raw_response
    return value


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.objects = v6.ObjectMatcher(CONFIG["objectName"])
        self.trends = v6.TrendClassifier(CONFIG["trend"], allow_model=False)

    def test_data_name_relaxes_format_only(self):
        self.assertTrue(v6.data_name_equal(" Revenue – USD ", "revenue -  usd"))
        self.assertTrue(v6.data_name_equal("ＡＢＣ", "abc"))
        self.assertFalse(v6.data_name_equal("Revenue (USD)", "Revenue"))

    def test_position_key_case_integer_string_and_order(self):
        gold = [{"Begin": [1, 2], "End": [3, 4]}]
        formatted = [{"end": ["3", "4"], "BEGIN": ["1", "2"]}]
        self.assertEqual(v6.canonical_position(gold), v6.canonical_position(formatted))
        self.assertNotEqual(
            v6.canonical_position(gold),
            v6.canonical_position([{"Begin": [2, 1], "End": [3, 4]}]),
        )
        self.assertIsNone(v6.canonical_position([{"Begin": [1, 2], "End": [3, 4], "x": 1}]))

    def test_num_extracts_only_num_field_and_rejects_extra_numbers(self):
        self.assertTrue(v6.numeric_equal([1234.0], ["$1,234"]))
        self.assertTrue(v6.numeric_equal([12.0], "12%"))
        self.assertTrue(v6.numeric_equal([3.5], ["SGD 3.5 billion"]))
        self.assertTrue(v6.numeric_equal([1.0, 2.0], ["2", "1"]))
        self.assertFalse(v6.numeric_equal([1.0], ["1 to 2"]))
        self.assertFalse(v6.numeric_equal([1.0], [True]))
        self.assertFalse(v6.numeric_equal([1.0], [float("nan")]))
        self.assertFalse(v6.numeric_equal([25.0], ["0.25%"]))

    def test_num_does_not_accept_conflicting_explicit_units(self):
        self.assertFalse(v6.numeric_equal(["1%"], ["1 percentage point"]))
        self.assertFalse(v6.numeric_equal(["USD 1"], ["EUR 1"]))

    def test_object_alias_boundary_and_one_to_one_mentions(self):
        self.assertTrue(
            self.objects.equal(
                ["United States", "European Union"],
                ["the U.S. economy", "EU"],
            )
        )
        self.assertFalse(
            self.objects.equal(
                ["United States", "European Union"],
                ["U.S."],
            )
        )
        self.assertFalse(self.objects.equal(["US"], ["Russia"]))

    def test_six_trend_classes_and_direction_none(self):
        expected = {
            "head & shoulders": "head_and_shoulders",
            "cup-and-handle": "cup_and_handle",
            "rounded bottom": "rounding_bottom",
            "double top": "double_top",
            "triple-top": "triple_top",
            "declined": "none",
            "None": "none",
        }
        for raw, label in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.trends.classify(raw)["class"], label)


class CountingTests(unittest.TestCase):
    def setUp(self):
        self.objects = v6.ObjectMatcher(CONFIG["objectName"])
        self.trends = v6.TrendClassifier(CONFIG["trend"], allow_model=False)

    def evaluate(self, gold, predicted, format_valid=True):
        return v6.evaluate_rows(
            [row(gold)], [prediction(predicted, format_valid)], self.objects, self.trends
        )

    def test_all_correct_is_five_field_and_binding_tp(self):
        gold = binding()
        predicted = binding(
            ObjectName=["U.S."],
            DataName=" revenue - usd ",
            Position=[{"begin": ["1", "2"], "END": [1, 2]}],
            Num=["$12"],
        )
        summary, _ = self.evaluate([gold], [predicted])
        self.assertEqual(summary["bindingLevel"]["tp"], 1)
        self.assertEqual(summary["primary"]["micro"]["tp"], 5)

    def test_present_invalid_or_wrong_is_fp_plus_fn(self):
        gold = binding()
        predicted = binding(Num=[True], ObjectName=["Canada"])
        summary, _ = self.evaluate([gold], [predicted])
        self.assertEqual(summary["primary"]["counts"]["Num"], {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(summary["primary"]["counts"]["ObjectName"], {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(summary["bindingLevel"], {
            "tp": 0, "fp": 1, "fn": 1,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        })

    def test_missing_field_is_fn_only(self):
        gold = binding()
        predicted = binding()
        predicted.pop("Num")
        summary, _ = self.evaluate([gold], [predicted])
        self.assertEqual(summary["primary"]["counts"]["Num"], {"tp": 0, "fp": 0, "fn": 1})

    def test_wrong_anchor_is_unmatched_gold_and_prediction(self):
        gold = binding()
        predicted = binding(DataName="Profit")
        summary, _ = self.evaluate([gold], [predicted])
        self.assertEqual(summary["bindingLevel"]["fp"], 1)
        self.assertEqual(summary["bindingLevel"]["fn"], 1)
        self.assertEqual(summary["primary"]["micro"]["fp"], 5)
        self.assertEqual(summary["primary"]["micro"]["fn"], 5)

    def test_unmatched_prediction_counts_only_present_fields(self):
        predicted = {"DataName": "Profit", "Position": [{"Begin": [9, 9], "End": [9, 9]}], "Num": [1]}
        summary, _ = self.evaluate([], [predicted])
        self.assertEqual(summary["primary"]["counts"]["DataName"]["fp"], 1)
        self.assertEqual(summary["primary"]["counts"]["Position"]["fp"], 1)
        self.assertEqual(summary["primary"]["counts"]["Num"]["fp"], 1)
        self.assertEqual(summary["primary"]["counts"]["Trend"]["fp"], 0)

    def test_format_invalid_nonempty_retains_gold_fn_and_binding_fp(self):
        summary, _ = self.evaluate([binding()], "not-json-binding", False)
        self.assertEqual(summary["bindingLevel"]["fp"], 1)
        self.assertEqual(summary["bindingLevel"]["fn"], 1)
        self.assertEqual(summary["primary"]["micro"]["fn"], 5)
        self.assertEqual(summary["coverage"]["format_invalid_rows"], 1)

    def test_empty_normalized_result_with_nonempty_raw_response_is_binding_fp(self):
        summary, _ = v6.evaluate_rows(
            [row([binding()])],
            [prediction([], False, "## Output {not-json}")],
            self.objects,
            self.trends,
        )
        self.assertEqual(summary["bindingLevel"]["fp"], 1)
        self.assertEqual(summary["bindingLevel"]["fn"], 1)
        self.assertEqual(summary["primary"]["micro"]["fp"], 0)
        self.assertEqual(summary["primary"]["micro"]["fn"], 5)
        self.assertEqual(summary["coverage"]["unparseable_nonempty_binding_fp"], 1)

    def test_whitespace_raw_response_does_not_create_binding_fp(self):
        summary, _ = v6.evaluate_rows(
            [row([binding()])],
            [prediction([], False, "  \n\t")],
            self.objects,
            self.trends,
        )
        self.assertEqual(summary["bindingLevel"]["fp"], 0)
        self.assertEqual(summary["bindingLevel"]["fn"], 1)
        self.assertNotIn("unparseable_nonempty_binding_fp", summary["coverage"])

    def test_nonempty_result_and_raw_response_count_only_one_binding_fp(self):
        summary, _ = v6.evaluate_rows(
            [row([binding()])],
            [prediction("not-json-binding", False, "also nonempty")],
            self.objects,
            self.trends,
        )
        self.assertEqual(summary["bindingLevel"]["fp"], 1)
        self.assertEqual(summary["coverage"]["unparseable_nonempty_binding_fp"], 1)

    def test_raw_response_fix_does_not_change_field_metrics(self):
        record = prediction([], False, "not-json")
        current, _ = v6.evaluate_rows(
            [row([binding()])], [record], self.objects, self.trends
        )
        baseline, _ = v601.evaluate_rows(
            [row([binding()])],
            [record],
            v601.ObjectMatcher(CONFIG["objectName"]),
            v601.TrendClassifier(CONFIG["trend"], allow_model=False),
        )
        self.assertEqual(current["primary"], baseline["primary"])

    def test_duplicate_anchor_reports_ambiguity_with_fixed_tie_break(self):
        first = binding(ObjectName=["United States"])
        second = binding(ObjectName=["European Union"])
        alignment = v6.align_bindings([first, second], [first, second])
        self.assertEqual(alignment["matches"], [
            {"goldIndex": 0, "predictionIndex": 0},
            {"goldIndex": 1, "predictionIndex": 1},
        ])
        self.assertEqual(len(alignment["ambiguity"]), 1)

    def test_trend_support_confusion_and_without_trend_ablation(self):
        gold = binding(Trend="declined")
        predicted = binding(Trend="increase")
        summary, _ = self.evaluate([gold], [predicted])
        self.assertEqual(summary["trend"]["support"]["none"], 1)
        self.assertEqual(summary["trend"]["confusionMatrix"]["none"]["none"], 1)
        self.assertEqual(summary["trend"]["goldOrdinaryDirectionCount"], 1)
        self.assertEqual(summary["withoutTrendAblation"]["micro"]["tp"], 4)


class AggregationTests(unittest.TestCase):
    @staticmethod
    def make_run(run_id, f1, precision, recall):
        counts = {field: {"tp": 1, "fp": 0, "fn": 0} for field in v6.PRIMARY_FIELDS}
        metrics = v6.metrics_from_counts(counts)
        metrics["macro"] = {"f1": f1, "precision": precision, "recall": recall}
        return {"run": run_id, "primary": metrics}

    def test_zero_denominator_is_na(self):
        self.assertIsNone(v6.metric(0, 0, 0)["precision"])
        self.assertIsNone(v6.metric(0, 0, 0)["recall"])
        self.assertIsNone(v6.metric(0, 0, 0)["f1"])

    def test_common_order_uses_one_contract(self):
        runs = [
            self.make_run(3, 0.8, 0.7, 0.9),
            self.make_run(2, 0.8, 0.8, 0.8),
            self.make_run(1, 0.7, 0.9, 0.9),
        ]
        self.assertEqual(v6.common_run_order(runs), [2, 3, 1])

    def test_diagnostic_omits_selection_formal_shares_top_runs(self):
        runs = [self.make_run(index, index / 10, index / 10, index / 10) for index in range(1, 11)]
        diagnostic = v6.aggregate_case("x", runs[:2], "diagnostic", CONFIG)
        formal = v6.aggregate_case("x", runs, "formal", CONFIG)
        self.assertIsNone(diagnostic["selection"])
        self.assertEqual(formal["selection"]["top3"]["runs"], [10, 9, 8])
        for field in v6.PRIMARY_FIELDS:
            self.assertIn(field, formal["selection"]["top1"]["fields"])

    def test_top3_propagates_na_instead_of_averaging_fewer_runs(self):
        runs = [self.make_run(index, index / 10, index / 10, index / 10) for index in range(1, 11)]
        runs[-1]["primary"]["fields"]["ObjectName"]["precision"] = None
        formal = v6.aggregate_case("x", runs, "formal", CONFIG)
        self.assertEqual(formal["selection"]["top3"]["runs"], [10, 9, 8])
        self.assertIsNone(formal["selection"]["top3"]["fields"]["ObjectName"]["precision"])
        self.assertEqual(v6.format_metric_value(None), "NA")
        self.assertEqual(v6.format_metric_value(0.125), "0.125000")

    def test_paired_bootstrap_and_text_normalization(self):
        result = v6.paired_bootstrap({1: 0.6, 2: 0.8}, {1: 0.5, 2: 0.5}, 1000, 7)
        self.assertAlmostEqual(result["meanDifference"], 0.2)
        self.assertEqual(v6.normalize_text_score(75, 100, 50), 50.0)
        self.assertIsNone(v6.normalize_text_score(75, 50, 50))
        self.assertIsNone(v6.normalize_text_score(math.nan, 100, 50))


class FakeJudge(v6.TextQualityJudge):
    def _request(self, messages):
        labels = [line.split(":", 1)[0].split()[-1] for line in messages[1]["content"].splitlines() if line.startswith("Text ")]
        return {"items": [
            {"label": label, "factual_consistency": 80, "natural_fluency": 90, "reason": "supported"}
            for label in labels
        ]}


class TextJudgeTests(unittest.TestCase):
    def test_blinded_randomized_labels_map_back_to_roles(self):
        judge = FakeJudge(CONFIG["textJudge"])
        result = judge.score("evidence", {"gold": "g", "prediction": "p", "flan_raw": "a"}, "seed")
        self.assertEqual(set(result["scores"]), {"gold", "prediction", "flan_raw"})
        self.assertTrue(all(item["factualConsistency"] == 80 for item in result["scores"].values()))


if __name__ == "__main__":
    unittest.main()
