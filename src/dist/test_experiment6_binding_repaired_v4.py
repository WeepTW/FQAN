#!/usr/bin/env python3
"""Regression tests for repaired-v4 Binding materialization and reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analyze_experiment6_binding_candidate_errors as analyzer
import evaluate_experiment6_binding_candidates_v1 as evaluator
import materialize_experiment6_bindings_repaired_v4 as repaired


class RepairedV4Tests(unittest.TestCase):
    def test_uppercase_begin_end_pseudo_binding(self) -> None:
        raw = (
            '[BEGIN] {ObjectName :[Country], DataName :""Government health '
            'spending per person, PPP USD""", Position : [{Begin:[0,0], '
            'End:[0,0]}], Trend:"None", Num:[], Text:"India spent $61."} [END]'
        )
        bindings, _, _, operations, marker = repaired.recover_explicit_answer(raw)
        self.assertEqual(marker, "BEGIN-END")
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0]["ObjectName"], ["Country"])
        self.assertEqual(bindings[0]["DataName"], "Government health spending per person, PPP USD")
        self.assertEqual(bindings[0]["Text"], "India spent $61.")
        self.assertIn("parse-explicit-pseudo-binding", operations)

    def test_output_marker_stops_before_examples(self) -> None:
        raw = (
            '## Output result: [[ObjectName:["WTI", "Real Price"], '
            'DataName="Date", Position:[{Begin:[0,0], End:[0,0]}], '
            'Trend:"None", Num:[], Text:"observed"]] ## Example [EXAMPLE 01] bad'
        )
        bindings, _, _, _, marker = repaired.recover_explicit_answer(raw)
        self.assertEqual(marker, "output-marker")
        self.assertEqual(len(bindings), 1)
        self.assertNotIn("EXAMPLE", bindings[0]["Text"])

    def test_prompt_expected_output_is_not_promoted(self) -> None:
        raw = 'Instructions. ## Expected output {"result":[{"ObjectName":["fake"]}]}'
        bindings, _, _, _, marker = repaired.recover_explicit_answer(raw)
        self.assertEqual(bindings, [])
        self.assertIsNone(marker)

    def test_auxiliary_fields_are_preserved(self) -> None:
        payload = {
            "result": [{"RetFact": ["fact"], "Reason": "binding reason"}],
            "reason": "row reason",
        }
        retfacts, reasons = repaired.auxiliary_fields(payload)
        self.assertEqual(retfacts, ["fact"])
        self.assertEqual(reasons, ["binding reason", "row reason"])

    def test_empty_and_gibberish_are_distinct(self) -> None:
        self.assertEqual(repaired.pure_gibberish_text(""), (False, None))
        gibberish, reason = repaired.pure_gibberish_text("[##_rowOC__CO_____ [ [ [")
        self.assertTrue(gibberish)
        self.assertEqual(reason, "known-degenerate-token-prefix")

    def test_candidate_stats_uses_format_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"outputId": "case", "formatValid": True, "bindingCount": 0, "candidateStatus": "empty"},
                {"outputId": "case", "formatValid": True, "bindingCount": 1, "candidateStatus": "binding"},
            ]
            (root / "rows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            stats = evaluator.candidate_stats(root)["case"]
        self.assertEqual(stats["acceptedRows"], 2)
        self.assertEqual(stats["acceptedEmptyBindingRows"], 1)
        self.assertEqual(stats["acceptedRowsWithBindings"], 1)
        self.assertEqual(stats.get("rejectedRows", 0), 0)

    def test_annotation_means_include_text_na(self) -> None:
        field = {
            name: {
                metric: {"mean": value}
                for metric, value in (("precision", 0.4), ("recall", 0.2), ("f1", 0.25))
            }
            for name in analyzer.scorer.PRIMARY_FIELDS
        }
        report = {"cases": [{"aggregate": {"fields": field}}]}
        means = analyzer.annotation_case_means(report)
        self.assertEqual(means["ObjectName"]["precision"], 0.4)
        self.assertIsNone(means["Text"]["f1"])

    def test_evaluator_supports_repaired_protocol(self) -> None:
        self.assertIn(repaired.PROTOCOL, evaluator.SUPPORTED_CANDIDATE_PROTOCOLS)

    def test_numeric_unit_cleaning_is_overlap_safe(self) -> None:
        for scorer in (evaluator.v601, evaluator.v602):
            parsed = scorer.parse_numeric_item("185 million")
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed.value, 185.0)
            self.assertEqual(parsed.scale, "million")


if __name__ == "__main__":
    unittest.main()
