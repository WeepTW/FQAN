#!/usr/bin/env python3
"""Tests for the fresh 6_mistral_d finalizer."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


DIST = Path(__file__).resolve().parent
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))

import finalize_experiment6_mistral_d_dynamic_finetuned as finalizer


class MistralDynamicFinalizerTests(unittest.TestCase):
    def test_unified_config_replaces_only_historical_mistral_d(self) -> None:
        source = json.loads(
            (
                finalizer.REPO_ROOT
                / "config"
                / "experiment6_binding_unified34_v2.json"
            ).read_text(encoding="utf-8")
        )
        config = finalizer.build_unified_config(
            source,
            {
                "protocol": "fresh-protocol",
                "compatibilityFingerprint": "fresh-fingerprint",
            },
        )
        groups = {group["name"]: group for group in config["sourceGroups"]}
        self.assertEqual(len(groups["corrected12"]["caseIds"]), 12)
        self.assertEqual(len(groups["historical21"]["caseIds"]), 21)
        self.assertNotIn(finalizer.CASE_ID, groups["historical21"]["caseIds"])
        self.assertEqual(groups["fresh_mistral_d"]["caseIds"], [finalizer.CASE_ID])
        self.assertEqual(
            groups["fresh_mistral_d"]["sourceCompatibilityFingerprint"],
            "fresh-fingerprint",
        )
        self.assertFalse(groups["fresh_mistral_d"]["requireRepairCoverage"])
        all_ids = [
            case_id
            for group in config["sourceGroups"]
            for case_id in group["caseIds"]
        ]
        self.assertEqual(len(all_ids), 34)
        self.assertEqual(len(set(all_ids)), 34)

    def test_thread_contract_rejects_non_four(self) -> None:
        with self.assertRaises(finalizer.FinalizationError):
            finalizer.require(2 == 4, "evaluation thread contract is fixed at 4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
