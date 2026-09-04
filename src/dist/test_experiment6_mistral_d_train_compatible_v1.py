#!/usr/bin/env python3
"""Contract tests for the isolated existing-adapter Mistral-d successor."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dist"))
MODULE_PATH = REPO / "dist/run_experiment6_mistral_d_train_compatible_v1.py"
CONFIG_PATH = (
    REPO
    / "config/experiment6_narrative2_generation_mistral_d_train_compatible_v1.json"
)
SPEC = importlib.util.spec_from_file_location(
    "experiment6_mistral_d_train_compatible_v1_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MistralDTrainCompatibleSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MODULE.ORIGINAL_LOAD_CONFIG(CONFIG_PATH)
        cls.contract = MODULE.validate_contract(cls.config)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(cls.contract["baseModel"]),
            local_files_only=True,
            trust_remote_code=True,
        )
        cls.rows, _ = MODULE.ORIGINAL_READ_INPUT_ROWS(cls.config, 0)
        cls.examples = MODULE.load_rendered_examples(
            MODULE.core.workspace_path(cls.config["exampleCsv"])
        )

    def test_config_isolates_existing_finetuned_dynamic_adapter(self) -> None:
        cases = MODULE.core.expand_matrix(self.config)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.output_id, "6_mistral_d")
        self.assertEqual(case.source_id, "finqa_mistral_d")
        self.assertEqual(case.prompt_mode, "dynamic-shot")
        self.assertEqual(case.route, "adapter-converter")
        self.assertEqual(self.config["retriever"]["maxNewTokens"], 512)
        self.assertEqual(self.contract["completionReserveTokens"], 512)

    def test_all_prompts_are_training_compatible_and_context_safe(self) -> None:
        transformed, report = MODULE.transform_rows(
            self.rows,
            self.config,
            tokenizer=self.tokenizer,
            rendered_examples=self.examples,
        )
        self.assertEqual(len(transformed), 85)
        self.assertEqual(report["rows"], 85)
        self.assertEqual(report["bareExampleMarkerRows"], 0)
        self.assertLessEqual(report["maxPromptPlusCompletionTokens"], 8192)
        self.assertEqual(sum(report["dispositions"].values()), 85)
        self.assertIn(
            "target-last-full-nearest-selected", report["dispositions"]
        )
        for row in transformed:
            prompt = row.retriever_prompts["dynamic-shot"]
            self.assertIn("# Retriever + Data Binding", prompt)
            target_start = prompt.rindex("# Retriever + Data Binding")
            target_section = prompt[target_start:]
            example_prefix = prompt[:target_start]
            self.assertIn('Result: {"RetFact":', example_prefix)
            self.assertNotIn("Context:", example_prefix)
            self.assertIn("## Context", target_section)
            self.assertIn("## Output Format", target_section)
            self.assertIn(f"question:{row.text}", target_section)
            self.assertNotIn(
                "question:What data-text bindings are described in the narrative?",
                target_section,
            )
            self.assertNotIn(row.data_raw, target_section)
            self.assertIn('"Trend": "None"', target_section)
            self.assertNotIn('"Trend":"None"', target_section)
            self.assertNotIn("## Binding coordinate contract", prompt)
            self.assertNotIn("[EXAMPLE", prompt)
            if "### Example" in prompt:
                self.assertLess(
                    prompt.index("### Example"),
                    target_start,
                )

    def test_prompt_transform_is_byte_deterministic(self) -> None:
        first, first_report = MODULE.transform_rows(
            self.rows,
            self.config,
            tokenizer=self.tokenizer,
            rendered_examples=self.examples,
        )
        second, second_report = MODULE.transform_rows(
            self.rows,
            self.config,
            tokenizer=self.tokenizer,
            rendered_examples=self.examples,
        )
        self.assertEqual(
            [row.retriever_prompts["dynamic-shot"] for row in first],
            [row.retriever_prompts["dynamic-shot"] for row in second],
        )
        self.assertEqual(first_report, second_report)


if __name__ == "__main__":
    unittest.main()
