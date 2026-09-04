#!/usr/bin/env python3

from __future__ import annotations

import unittest

from build_experiment6_six_field_mean_table import FIELDS, STAGE, render


class SixFieldTableTest(unittest.TestCase):
    def test_all_six_fields_and_macro_are_rendered(self) -> None:
        metric = {
            name: {"precision": {"mean": 0.1}, "recall": {"mean": 0.2}, "f1": {"mean": 0.3}}
            for name in FIELDS
        }
        report = {
            "judge": {"model": "gpt-5.5", "reasoningEffort": "medium", "minimumConfidence": 0.8},
            "cases": [
                {
                    "outputId": "6_mistral_base_m",
                    "ablations": {
                        STAGE: {
                            "fields": metric,
                            "macro": {
                                "precision": {"mean": 0.1},
                                "recall": {"mean": 0.2},
                                "f1": {"mean": 0.3},
                            },
                        }
                    },
                }
            ],
        }
        text = render(report)
        for field in FIELDS:
            self.assertIn(f"| 6_mistral_base_m | {field} |", text)
        self.assertIn("Macro (6 fields)", text)
        self.assertIn("gpt-5.5", text)


if __name__ == "__main__":
    unittest.main()
