#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from combine_experiment6_v610_with_text_semantic import FIVE_FIELDS, build, markdown


class CombinedReportTest(unittest.TestCase):
    def test_two_case_six_field_macro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            five = {
                field: {name: {"mean": 0.2} for name in ("precision", "recall", "f1")}
                for field in FIVE_FIELDS
            }
            case_ids = ("6_mistral_base_m", "6_mistral_base_d")
            v610 = {
                "protocol": "experiment6-reference-aligned-v6.1.0",
                "cases": [
                    {"outputId": case, "runs": 10, "aggregate": {"fields": five}}
                    for case in case_ids
                ],
            }
            semantic_metric = {
                name: {"mean": 0.8} for name in ("precision", "recall", "f1")
            }
            semantic = {
                "protocol": "narrative2-reference-aligned-hybrid-v5.1",
                "judge": {"model": "gpt-5.5", "reasoningEffort": "medium", "minimumConfidence": 0.8, "disabled": False},
                "cases": [
                    {
                        "outputId": case,
                        "ablations": {"semantic_gpt55_medium": {"fields": {"Text": semantic_metric}}},
                    }
                    for case in case_ids
                ],
            }
            v610_path = root / "v610.json"
            semantic_path = root / "semantic.json"
            v610_path.write_text(json.dumps(v610), encoding="utf-8")
            semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
            result = build(v610_path, semantic_path)
            self.assertAlmostEqual(result["cases"][0]["macro"]["f1"], 0.3)
            text = markdown(result)
            self.assertIn("| 6_mistral_base_m | Text | 0.800000", text)
            self.assertIn("TN 未定義", text)
            semantic["judge"]["disabled"] = True
            semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "judge identity"):
                build(v610_path, semantic_path)


if __name__ == "__main__":
    unittest.main()
