#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import analyze_experiment6_binding_candidate_errors as analysis
import build_experiment6_binding_candidate_score_tables as candidate_tables


class MaximumWeightPairsTests(unittest.TestCase):
    def test_trend_only_pair_is_rejected(self) -> None:
        vectors = [[[[False, False, False, True, False][index] for index in range(5)]]]
        self.assertEqual(analysis.maximum_weight_pairs(vectors), [])

    def test_maximizes_non_trend_equal_fields_one_to_one(self) -> None:
        # Fields follow ObjectName, DataName, Position, Trend, Num.
        vectors = [
            [
                [True, True, False, True, False],
                [False, False, True, True, False],
            ],
            [
                [False, True, False, True, False],
                [True, False, True, True, True],
            ],
        ]
        self.assertEqual(analysis.maximum_weight_pairs(vectors), [(0, 0), (1, 1)])


class PairCategoryTests(unittest.TestCase):
    def test_categories(self) -> None:
        self.assertEqual(
            analysis.classify_pair(analysis.scorer.PRIMARY_FIELDS),
            "all_five_fields_equal",
        )
        self.assertEqual(
            analysis.classify_pair(["DataName", "Position", "Trend"]),
            "hard_anchor_equal_other_field_error",
        )
        self.assertEqual(
            analysis.classify_pair(["Position", "Trend"]),
            "partial_anchor_near_miss",
        )
        self.assertEqual(
            analysis.classify_pair(["Num", "Trend"]),
            "value_only_near_miss",
        )


class CandidateTableLabelTests(unittest.TestCase):
    def test_field_scores_are_labeled_as_anchored_end_to_end(self) -> None:
        rendered = candidate_tables.render(
            "mean",
            [],
            {
                "scoringProtocol": "experiment6-reference-aligned-v6.0.2",
                "scope": "candidate12",
                "time": "2026-08-15T00:00:00Z",
                "completedCases": 0,
                "protocol": "experiment6-binding-candidate-evaluation-v1",
                "method": {"methodSha256": "test"},
            },
        )
        self.assertIn("anchored end-to-end field F1", rendered)
        self.assertIn("binding identity anchor", rendered)


if __name__ == "__main__":
    unittest.main()
