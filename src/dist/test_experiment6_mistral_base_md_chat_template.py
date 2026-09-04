#!/usr/bin/env python3
"""Regression tests for the isolated Mistral base chat-template diagnostic."""

from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = (
    REPO_ROOT
    / ".external/FINDER/Retriever Codes"
    / "Mistral"
    / "mistral_direct_binding_chat_inference.py"
)
WRAPPER_PATH = REPO_ROOT / "dist" / "run_experiment6_mistral_base_md_chat_template_v2.py"
CONFIG_PATH = REPO_ROOT / "config" / "experiment6_narrative2_generation_mistral_base_md_chat_template_v2.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = load_module("mistral_chat_inference_test_target", INFERENCE_PATH)
wrapper = load_module("mistral_chat_wrapper_test_target", WRAPPER_PATH)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return [1, *range(10, 10 + len(messages[0]["content"])), 2]


class ChatTemplateInferenceTests(unittest.TestCase):
    def test_semantic_prompt_is_wrapped_as_one_native_user_message(self) -> None:
        tokenizer = FakeTokenizer()
        prompt = "semantic prompt body"
        ids = inference.native_chat_ids(tokenizer, prompt)
        self.assertTrue(ids)
        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": prompt}])
        self.assertTrue(tokenizer.calls[0][1])
        self.assertTrue(tokenizer.calls[0][2])

    def test_build_records_forbids_truncation_and_binds_exact_ids(self) -> None:
        frame = pd.DataFrame(
            [{"input": "short", "Rel_Fact": "__BLINDED__", "Source": "S1"}]
        )
        tokenizer = FakeTokenizer()
        records = inference.build_records(
            frame,
            tokenizer,
            template_sha256="a" * 64,
            max_input_length=32,
            max_new_tokens=4,
            context_window=36,
            sort_by_length=True,
        )
        self.assertEqual(records[0]["prompt"], "short")
        self.assertEqual(len(records[0]["input_sha256"]), 64)
        with self.assertRaisesRegex(RuntimeError, "exceeds input limit"):
            inference.build_records(
                frame,
                tokenizer,
                template_sha256="a" * 64,
                max_input_length=2,
                max_new_tokens=4,
                context_window=36,
                sort_by_length=True,
            )

    def test_command_is_no_adapter_direct_generation(self) -> None:
        args = argparse.Namespace(
            limit=3,
            max_tokens=4096,
            max_input_tokens=8192,
        )
        contract = {
            "contextWindow": 12288,
            "chatTemplateSha256": "b" * 64,
        }
        command = wrapper.build_command(
            csv_path=Path("input.csv"),
            raw_output=Path("output.jsonl"),
            prompt_mode="many-shot",
            args=args,
            batch_size=6,
            contract=contract,
        )
        self.assertIn("--no-adapter", command)
        self.assertEqual(command[command.index("--structured-output") + 1], "off")
        self.assertNotIn("--adapter-dir", command)
        self.assertNotIn("converter", " ".join(command).lower())

    def test_config_selects_exactly_two_direct_cases_and_locks_code(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        wrapper.validate_contract(config)
        core = wrapper.core
        loaded = core.load_config(CONFIG_PATH)
        cases = core.expand_matrix(loaded)
        self.assertEqual(
            [case.output_id for case in cases],
            ["6_mistral_base_m", "6_mistral_base_d"],
        )
        self.assertTrue(all(case.route == "direct-binding" for case in cases))
        self.assertTrue(all(case.source_id == "mistral_v0_3" for case in cases))


if __name__ == "__main__":
    unittest.main()
