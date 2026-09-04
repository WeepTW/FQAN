#!/usr/bin/env python3
"""Regression tests for Mistral chat repaired projection decisions."""

from __future__ import annotations

import unittest

from materialize_experiment6_mistral_chat_repaired_projection import choose_result


VALID = {
    "ObjectName": ["Africa"],
    "DataName": "Fertility rate",
    "Position": [{"Begin": [0, 1], "End": [0, 1]}],
    "Trend": "decline",
    "Num": [],
    "Text": "Africa declined.",
}


class ProjectionDecisionTest(unittest.TestCase):
    def test_strict_result_is_preserved(self) -> None:
        result, status, operations = choose_result(
            {"formatValid": True, "result": [VALID], "rawResponse": "ignored"}, None
        )
        self.assertEqual(result, [VALID])
        self.assertEqual(status, "strict-valid-preserved")
        self.assertEqual(operations, [])

    def test_recorded_repair_precedes_raw_recovery(self) -> None:
        recorded = {
            "formatValid": True,
            "repairMethod": "recorded-v1",
            "repairedPayload": {"result": [VALID]},
        }
        result, status, operations = choose_result(
            {"formatValid": False, "result": [], "rawResponse": "not JSON"}, recorded
        )
        self.assertEqual(result, [VALID])
        self.assertEqual(status, "generation-repair-preserved")
        self.assertEqual(operations, ["recorded-v1"])

    def test_unique_six_field_object_is_recovered(self) -> None:
        result, status, operations = choose_result(
            {
                "formatValid": False,
                "result": [],
                "rawResponse": "answer\n" + __import__("json").dumps(VALID),
            },
            None,
        )
        self.assertEqual(result, [VALID])
        self.assertEqual(status, "unique-object-recovered")
        self.assertEqual(operations, ["unique-six-field-binding-object-v1"])

    def test_prompt_echo_becomes_empty_not_invented_binding(self) -> None:
        raw = "## Output examples\n[EXAMPLE 1]\n{}\n[EXAMPLE 2]\n{}"
        result, status, operations = choose_result(
            {"formatValid": False, "result": [], "rawResponse": raw}, None
        )
        self.assertEqual(result, [])
        self.assertEqual(status, "unrecoverable-as-empty")
        self.assertEqual(operations, ["prompt_echo_guard"])


if __name__ == "__main__":
    unittest.main()
