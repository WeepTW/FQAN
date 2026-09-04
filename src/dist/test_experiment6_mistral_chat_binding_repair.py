#!/usr/bin/env python3
"""Tests for the isolated gold-free Mistral chat repair rule."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiment6_mistral_chat_binding_repair import repair_unique_binding


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "Experiment"
OLD_ROOT = (
    REPO_ROOT
    / "Experiment"
    / "experiment_6_narrative2_corrected12_20260812T153853Z"
)


def raw_prediction(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8").splitlines()[0])["rawResponse"])


class MistralChatRepairTests(unittest.TestCase):
    def test_single_numbered_target_binding_is_recovered(self) -> None:
        path = (
            SMOKE_ROOT
            / "experiment_6_mistral_base_md_chat_smoke_d_Econ010_20260820T190925Z"
            / "cases/6_mistral_base_d/run_01/predictions.jsonl"
        )
        repair = repair_unique_binding(raw_prediction(path))
        self.assertTrue(repair["available"])
        self.assertEqual(repair["method"], "unique-six-field-binding-object-v1")
        self.assertEqual(len(repair["payload"]["result"]), 1)

    def test_historical_multi_example_echo_is_rejected(self) -> None:
        path = OLD_ROOT / "cases/6_mistral_base_d/run_01/predictions.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        raw = next(item["rawResponse"] for item in records if item["source"] == "Econ_010")
        repair = repair_unique_binding(raw)
        self.assertFalse(repair["available"])
        self.assertEqual(repair["reason"], "prompt_echo_guard")

    def test_multiple_binding_objects_are_never_promoted(self) -> None:
        binding = {
            "ObjectName": ["x"], "DataName": "y", "Position": [],
            "Trend": "None", "Num": [], "Text": "z",
        }
        raw = json.dumps(binding) + "\n" + json.dumps({**binding, "Text": "other"})
        repair = repair_unique_binding(raw)
        self.assertFalse(repair["available"])
        self.assertEqual(repair["reason"], "binding_object_not_unique")

    def test_numbered_target_explanation_with_one_binding_is_recovered(self) -> None:
        binding = {
            "ObjectName": ["Fox News"], "DataName": "Fox", "Position": [],
            "Trend": "None", "Num": [6.46], "Text": "Fox News recorded 6.46m",
        }
        raw = (
            "[EXAMPLE 11] target assertion\n"
            "[EXAMPLE 12] another target assertion\n"
            + json.dumps({"result": [binding], "reason": "Success"})
        )
        repair = repair_unique_binding(raw)
        self.assertTrue(repair["available"])
        self.assertEqual(repair["payload"]["result"], [binding])


if __name__ == "__main__":
    unittest.main()
