#!/usr/bin/env python3
"""Regression tests for Experiment 6 v6 score-table reporting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from build_experiment6_v6_score_tables import (
    FIELD_MAP,
    REPORT_COLUMNS,
    generation_finished_at,
    relationship_summary,
    safe_correlation,
)


def make_case(score: float, matched: int, empty: int) -> dict[str, object]:
    return {
        "aggregate": {
            "macro": {"f1": {"mean": score}},
            "fields": {
                field: {"f1": {"mean": score}} for _, field in FIELD_MAP
            },
        },
        "runResults": [
            {
                "coverage": {
                    "empty_output_rows": empty,
                    "gold_bindings": 1,
                    "matched_bindings": matched,
                    "predicted_bindings": matched,
                }
            }
        ],
    }


class RelationshipSummaryTest(unittest.TestCase):
    def test_constant_or_short_vectors_return_na_correlation(self) -> None:
        self.assertIsNone(safe_correlation([0.0, 0.0], [0.0, 0.0]))
        self.assertIsNone(safe_correlation([0.0], [0.0]))
        self.assertIsNone(safe_correlation([0.0], [0.0, 1.0]))

    def test_report_column_order_and_manifest_finished_at(self) -> None:
        self.assertEqual(
            REPORT_COLUMNS,
            (
                "fine-tuned method (prompt-type or no-adaptor)",
                "retriever model",
                "input prompt-type",
                "**Subject**",
                "**Trend**",
                "**Num**",
                "**Position**",
                "**DataName**",
                "**Text**",
            ),
        )
        self.assertEqual(
            generation_finished_at(
                [{"finishedAt": "2026-08-12T00:00:00Z"}, {"finishedAt": "2026-08-14T01:02:03Z"}]
            ),
            "2026-08-14T01:02:03Z",
        )

    def test_empty_rate_uses_record_count_without_nonempty_coverage_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_root = root / "cases"
            for output_id, source in (("case_a", "s1"), ("case_b", "s2")):
                run_root = cases_root / output_id / "run_01"
                run_root.mkdir(parents=True)
                (run_root / "records.jsonl").write_text(
                    json.dumps({"source": source, "matchDetails": []}) + "\n",
                    encoding="utf-8",
                )
            gold_path = root / "gold.json"
            gold_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {"source": "s1", "targetBindings": []},
                            {"source": "s2", "targetBindings": []},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "outputId": "case_a",
                    "fineTunedMethod": "zero-shot",
                    "retrieverModel": "model-a",
                    "inputPromptType": "zero-shot",
                    "case": make_case(0.0, matched=0, empty=1),
                },
                {
                    "outputId": "case_b",
                    "fineTunedMethod": "many-shot",
                    "retrieverModel": "model-b",
                    "inputPromptType": "many-shot",
                    "case": make_case(1.0, matched=1, empty=0),
                },
            ]
            # Serialized Counters omit zero-valued keys.
            del rows[1]["case"]["runResults"][0]["coverage"]["empty_output_rows"]

            summary = relationship_summary(rows, root, gold_path)

            self.assertAlmostEqual(
                summary["correlations"]["matchedBindingRateVsMacroF1"], 1.0
            )
            self.assertAlmostEqual(
                summary["correlations"]["emptyRowRateVsMacroF1"], -1.0
            )

    def test_all_zero_cases_are_reportable_with_na_correlations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output_id, source in (("case_a", "s1"), ("case_b", "s2")):
                run_root = root / "cases" / output_id / "run_01"
                run_root.mkdir(parents=True)
                (run_root / "records.jsonl").write_text(
                    json.dumps({"source": source, "matchDetails": []}) + "\n",
                    encoding="utf-8",
                )
            gold_path = root / "gold.json"
            gold_path.write_text(
                json.dumps({"rows": [
                    {"source": "s1", "targetBindings": []},
                    {"source": "s2", "targetBindings": []},
                ]}),
                encoding="utf-8",
            )
            rows = [
                {
                    "outputId": output_id,
                    "fineTunedMethod": "no-adaptor",
                    "retrieverModel": "model",
                    "inputPromptType": "original",
                    "case": make_case(0.0, matched=0, empty=1),
                }
                for output_id in ("case_a", "case_b")
            ]
            summary = relationship_summary(rows, root, gold_path)
            self.assertEqual(summary["macroF1AcrossCases"]["mean"], 0.0)
            self.assertIsNone(summary["correlations"]["matchedBindingRateVsMacroF1"])
            self.assertIsNone(summary["correlations"]["emptyRowRateVsMacroF1"])


if __name__ == "__main__":
    unittest.main()
