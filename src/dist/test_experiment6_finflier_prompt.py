#!/usr/bin/env python3
"""Regression tests for the No-adaptor × FinFlier Experiment 6 input."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

DIST = Path(__file__).resolve().parent
REPO = DIST.parent
sys.path.insert(0, str(DIST))
sys.path.insert(0, str(REPO))

import experiment6_finflier as orchestrator
import run_experiment6_narrative2_generation as runner
from narrative.finflier_prompt import build_prompt, load_prompt_asset


CONFIG = REPO / "config" / "experiment6_narrative2_generation_finflier_no_adapter.json"
ASSET = REPO / "narrative" / "assets" / "generation" / "finflier_prompt_v1.json"


class FinFlierPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = runner.load_config(CONFIG)
        cls.asset = load_prompt_asset(
            ASSET, cls.config["finflierPrompt"]["asset"]["sha256"]
        )

    def test_asset_matches_reference_and_contains_ten_general_examples(self) -> None:
        reference = runner.PATHS.resolve("code", "FinFlier/backend/app.py")
        self.assertEqual(
            hashlib.sha256(reference.read_bytes()).hexdigest(),
            self.asset.source["sha256"],
        )
        self.assertEqual(self.asset.general_example_count, 10)
        self.assertEqual(self.asset.default_prompt.count("\nresult:"), 10)
        self.assertEqual(
            self.asset.dispatch_order,
            (
                "head and shoulder",
                "cup and handle",
                "rounding bottom",
                "double top",
                "triple top",
            ),
        )

    def test_special_dispatch_is_first_match_only(self) -> None:
        prompt, audit = build_prompt(
            self.asset,
            chart_data='[{"Close": 1}]',
            narrative="A head and shoulder appears before a double top.",
            coordinate_contract=runner.POSITION_INDEX_CONTRACT,
            output_contract=runner.DIRECT_OUTPUT_CONTRACT,
        )
        self.assertEqual(audit["specialExampleId"], "head and shoulder")
        self.assertEqual(prompt.count(self.asset.special_prompts["head and shoulder"]["text"].strip()), 1)
        self.assertNotIn(self.asset.special_prompts["double top"]["text"].strip(), prompt)
        self.assertEqual(audit["finalPromptSha256"], runner.sha256_text(prompt))

    def test_matrix_is_three_direct_no_adaptor_cases(self) -> None:
        cases = runner.expand_matrix(self.config)
        self.assertEqual({case.output_id for case in cases}, set(orchestrator.CASE_DEVICES))
        self.assertTrue(all(case.route == "direct-binding" for case in cases))
        self.assertTrue(all(runner.source_kind(case.source_id) == "base" for case in cases))
        self.assertEqual(self.config["inputType"], "FinFlier")
        self.assertEqual(self.config["expectedFormalPredictions"], 2550)

    def test_generation_rows_use_finflier_without_gold(self) -> None:
        rows, report = runner.read_input_rows(self.config, 1)
        self.assertEqual(len(rows), 1)
        prompt = rows[0].direct_prompts["original"]
        audit = report["promptAudit"][0]["finflierPrompt"]
        self.assertTrue(prompt.startswith(self.asset.default_prompt.strip()))
        self.assertTrue(audit["finflierPromptApplied"])
        self.assertEqual(audit["generalExampleCount"], 10)
        self.assertEqual(audit["finalPromptSha256"], runner.sha256_text(prompt))
        for forbidden in ('"targetBindings"', '"Binding_Result"', '"gold_targets"'):
            self.assertNotIn(forbidden, prompt)

    def test_prompt_bundle_records_input_type(self) -> None:
        rows, _ = runner.read_input_rows(self.config, 1)
        with tempfile.TemporaryDirectory() as directory:
            _, report = runner.materialize_prompt_bundles(
                Path(directory), rows, self.config
            )
            self.assertEqual(report["inputType"], "FinFlier")
            manifest = json.loads(
                (Path(directory) / "input_bundles" / "original" / "bundle_manifest.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["inputType"], "FinFlier")


if __name__ == "__main__":
    unittest.main()
