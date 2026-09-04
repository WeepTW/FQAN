#!/usr/bin/env python3
"""Regression tests for the existing Mistral dynamic adapter prompt contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO / ".external/FINDER/Retriever Codes/Mistral/mistral_dynamic_adapter_prompt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mistral_dynamic_adapter_prompt_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PROMPT = """# Financial RetFact retrieval

Source: Econ_020

## Chart data (lossless compact form)
chart payload

## Narrative
target narrative

## RetFact examples
[EXAMPLE 01] bare answer one
[EXAMPLE 02] bare answer two

## Output contract
Return the candidate RetFact in canonical JSON.
"""


class CharacterTokenizer:
    """Small reversible tokenizer for policy-only unit tests."""

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        del add_special_tokens
        token_ids = [ord(character) for character in str(text)]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return {"input_ids": token_ids}

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class MistralDynamicAdapterPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def test_replaces_bare_examples_and_preserves_training_boundary(self) -> None:
        rendered = [
            "Context:\nfull example context one\nResult: {\"RetFact\":\"one\"}",
            "Context:\nfull example context two\nResult: {\"RetFact\":\"two\"}",
        ]
        result = MODULE.build_training_compatible_prompt(
            prompt=PROMPT,
            rendered_examples=rendered,
            selected_indices=(0, 1),
            tokenizer=self.tokenizer,
            label_marker="##Label Descriptions:",
            train_max_seq_length=380,
            completion_reserve_tokens=80,
            preserve_target_over_budget=True,
            reverse_selected=False,
        )
        effective = result["prompt"]
        audit = result["audit"]

        self.assertNotIn("[EXAMPLE", effective)
        self.assertLess(
            effective.index("## Output contract"),
            effective.index("### Example"),
        )
        self.assertIn("Context:\nfull example context one", effective)
        self.assertLessEqual(audit["effectivePromptTokens"], audit["promptBudgetTokens"])
        self.assertTrue(audit["trainingPrefixApplied"])
        self.assertEqual(audit["selectedExampleIndices"], [0, 1])

    def test_preserves_long_target_and_drops_examples_instead_of_truncating_chart(self) -> None:
        long_prompt = PROMPT.replace("chart payload", "x" * 500)
        result = MODULE.build_training_compatible_prompt(
            prompt=long_prompt,
            rendered_examples=["Context:\nexample\nResult: {}"],
            selected_indices=(0,),
            tokenizer=self.tokenizer,
            label_marker="##Label Descriptions:",
            train_max_seq_length=300,
            completion_reserve_tokens=80,
            preserve_target_over_budget=True,
            reverse_selected=False,
        )

        self.assertIn("x" * 500, result["prompt"])
        self.assertNotIn("### Example", result["prompt"])
        self.assertEqual(result["audit"]["disposition"], "target-preserved-examples-dropped")
        self.assertGreater(
            result["audit"]["effectivePromptTokens"],
            result["audit"]["promptBudgetTokens"],
        )

    def test_target_last_keeps_only_full_nearest_examples(self) -> None:
        rendered = ["A" * 90, "B" * 20, "C" * 20]
        target = "# Target\n\n## Output Format\n{}\n"
        result = MODULE.build_training_compatible_prompt(
            prompt=PROMPT,
            rendered_examples=rendered,
            selected_indices=(0, 1, 2),
            tokenizer=self.tokenizer,
            label_marker="##Label Descriptions:",
            train_max_seq_length=220,
            completion_reserve_tokens=80,
            preserve_target_over_budget=True,
            reverse_selected=False,
            target_prompt_override=target,
            placement_policy="target-last-full-nearest-v1",
        )

        effective = result["prompt"]
        audit = result["audit"]
        self.assertNotIn("A" * 90, effective)
        self.assertIn("B" * 20, effective)
        self.assertIn("C" * 20, effective)
        self.assertTrue(effective.endswith(target))
        self.assertLess(effective.index("### Example"), effective.index("# Target"))
        self.assertEqual(audit["selectedExampleIndices"], [1, 2])
        self.assertEqual(audit["placementPolicy"], "target-last-full-nearest-v1")

    def test_rejects_completion_reserve_that_exhausts_training_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "completion reserve"):
            MODULE.build_training_compatible_prompt(
                prompt=PROMPT,
                rendered_examples=["Context:\nexample\nResult: {}"],
                selected_indices=(0,),
                tokenizer=self.tokenizer,
                label_marker="##Label Descriptions:",
                train_max_seq_length=64,
                completion_reserve_tokens=64,
                preserve_target_over_budget=True,
                reverse_selected=False,
            )


if __name__ == "__main__":
    unittest.main()
