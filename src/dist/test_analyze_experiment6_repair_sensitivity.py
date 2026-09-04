#!/usr/bin/env python3
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

from analyze_experiment6_repair_sensitivity import (
    derive_predictions,
    strict_payload_result,
    validate_binding,
)


def binding(**updates):
    value = {
        "ObjectName": ["United States"],
        "DataName": "Revenue",
        "Position": [{"Begin": [1, 2], "End": [1, 2]}],
        "Trend": "None",
        "Num": [12.0],
        "Text": "Revenue was 12.",
    }
    value.update(updates)
    return value


class RepairSensitivityTests(unittest.TestCase):
    def test_full_schema_is_required(self) -> None:
        self.assertEqual(validate_binding(binding()), (True, "valid"))
        valid, reason = validate_binding(binding(Num=[["12"]]))
        self.assertFalse(valid)
        self.assertEqual(reason, "Num_not_finite_number_array")
        valid, reason = validate_binding(binding(extra="x"))
        self.assertFalse(valid)
        self.assertTrue(reason.startswith("binding_keys="))

    def test_payload_contract_matches_direct_strict_parser(self) -> None:
        result, reason = strict_payload_result(
            {"result": [binding()], "reason": "ok"}
        )
        self.assertEqual(reason, "valid")
        self.assertEqual(result, [binding()])
        result, reason = strict_payload_result({"result": [binding()]})
        self.assertIsNone(result)
        self.assertEqual(reason, "top_level_contract")

    def test_derived_rows_do_not_mutate_official_predictions(self) -> None:
        predictions = [
            {
                "index": 0,
                "source": "S1",
                "result": [],
                "formatValid": False,
                "rawResponse": "## Output {...}",
            }
        ]
        original = copy.deepcopy(predictions)
        repairs = [
            {
                "index": 0,
                "source": "S1",
                "official": False,
                "excludedFromScores": True,
                "repair": {
                    "available": True,
                    "method": "fence-strip-balanced-json",
                    "payload": {"result": [binding()], "reason": "ok"},
                },
            }
        ]
        derived, summary = derive_predictions(predictions, repairs)
        self.assertEqual(predictions, original)
        self.assertTrue(derived[0]["formatValid"])
        self.assertFalse(derived[0]["parserDiagnostic"]["claimEligible"])
        self.assertEqual(summary["repairAvailableRows"], 1)
        self.assertEqual(summary["repairSchemaValidRows"], 1)

    def test_schema_rejected_repair_stays_invalid(self) -> None:
        predictions = [
            {"index": 0, "source": "S1", "result": [], "formatValid": False}
        ]
        repairs = [
            {
                "index": 0,
                "source": "S1",
                "official": False,
                "excludedFromScores": True,
                "repair": {
                    "available": True,
                    "payload": {"result": [binding(Num=[["12"]])], "reason": ""},
                },
            }
        ]
        derived, summary = derive_predictions(predictions, repairs)
        self.assertFalse(derived[0]["formatValid"])
        self.assertEqual(summary["repairSchemaRejectedRows"], 1)


if __name__ == "__main__":
    unittest.main()
