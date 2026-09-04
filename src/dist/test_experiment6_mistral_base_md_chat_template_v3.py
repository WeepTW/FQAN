#!/usr/bin/env python3
"""Transformers-5 and frozen-contract tests for the Mistral chat runner v3."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from transformers.tokenization_utils_base import BatchEncoding

import run_experiment6_mistral_base_md_chat_template_v3 as target


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "config"
    / "experiment6_narrative2_generation_mistral_base_md_chat_template_v3.json"
)


class ChatTemplateV3Tests(unittest.TestCase):
    def test_transformers_five_batch_encoding_is_normalized(self) -> None:
        value = BatchEncoding({"input_ids": [1, 3, 4], "attention_mask": [1, 1, 1]})
        self.assertEqual(target.token_ids(value), [1, 3, 4])

    def test_all_scientific_implementation_files_are_hash_locked(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        contract = target.validate_contract(config)
        self.assertEqual(len(contract["chatTemplateSha256"]), 64)
        self.assertEqual(
            set(name for name in contract if name in {
                "wrapper", "wrapperBase", "inference", "inferenceBase"
            }),
            {"wrapper", "wrapperBase", "inference", "inferenceBase"},
        )

    def test_matrix_is_exactly_many_and_dynamic_no_adapter(self) -> None:
        loaded = target.base.core.load_config(CONFIG)
        cases = target.base.core.expand_matrix(loaded)
        self.assertEqual(
            [(case.output_id, case.prompt_mode, case.route) for case in cases],
            [
                ("6_mistral_base_m", "many-shot", "direct-binding"),
                ("6_mistral_base_d", "dynamic-shot", "direct-binding"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
