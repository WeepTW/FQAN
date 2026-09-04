#!/usr/bin/env python3
"""Regression tests for Mistral target-boundary chat prompt ordering."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO / ".external/FINDER/Retriever Codes/Mistral/mistral_direct_binding_chat_inference_v4.py"
)
SPEC = importlib.util.spec_from_file_location("mistral_target_boundary_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PROMPT = """Extract Binding JSON only.

Prompt mode: dynamic-shot

Source: Econ_020

## Binding coordinate contract
zero-based coordinates

## Chart data (lossless compact form)
chart payload

## Narrative
target narrative

## RetFact examples
[EXAMPLE 01] example answer one
[EXAMPLE 02] example answer two

## Output contract
Return exactly {\"result\":[]}.
"""


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        self.assertions = (tokenize, add_generation_prompt)
        return [1, 2, 3]


class TargetLastPromptTests(unittest.TestCase):
    def test_moves_examples_before_target_and_preserves_sections(self) -> None:
        transformed = MODULE.target_last_prompt(PROMPT)
        self.assertLess(transformed.index("## RetFact examples"), transformed.index("Source:"))
        self.assertLess(transformed.index("Source:"), transformed.index("## Narrative"))
        self.assertLess(transformed.index("## Narrative"), transformed.index("## Output contract"))
        for section in PROMPT.rstrip("\n").split("\n\n"):
            self.assertIn(section, transformed)
        self.assertLess(
            transformed.index("## Output contract"),
            transformed.index("## Target answer boundary"),
        )
        self.assertTrue(transformed.endswith(MODULE.TARGET_BOUNDARY))

    def test_native_chat_uses_transformed_content(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(MODULE.native_chat_ids(tokenizer, PROMPT), [1, 2, 3])
        self.assertEqual(tokenizer.assertions, (True, True))
        content = tokenizer.messages[0]["content"]
        self.assertLess(content.index("## RetFact examples"), content.index("Source:"))
        self.assertIn("Do not reproduce, number, summarize, or continue", content)
        self.assertTrue(content.endswith(MODULE.TARGET_BOUNDARY))

    def test_rejects_prompt_without_exactly_one_example_block(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            MODULE.target_last_prompt(PROMPT.replace("## RetFact examples", "## Notes"))
        duplicate = PROMPT.replace(
            "## Output contract", "## RetFact examples\n[EXAMPLE 03] duplicate\n\n## Output contract"
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            MODULE.target_last_prompt(duplicate)


if __name__ == "__main__":
    unittest.main()
