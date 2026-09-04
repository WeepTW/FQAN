#!/usr/bin/env python3
"""Regression checks for Experiment 7 target-answer execution."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from threading import Lock
from time import sleep

os.environ.setdefault("FQAN_ALLOW_GENERATED_CODE_EXECUTION", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_few10_generator_input as few10_builder
import new_full_finqa_run as finqa_runner
from build_few10_generator_input import load_matched_examples
from example_selection import (
    attach_materialized_selection,
    build_in_context_prompt,
    cache_items_by_key,
    formal_target_gold_missing,
    format_example_prompt,
    selected_from_cache,
    selected_from_materialized_row,
    selection_key_for_row,
)
from new_full_finqa_run import (
    EngineConfig,
    ResumeOutputMismatch,
    build_prompt,
    chat_completion_kwargs_for_sampling,
    effective_max_tokens_for_config,
    execute_codes,
    execute_python_code,
    generated_code_has_percent_scale_multiply,
    generator_batch_size_for_config,
    local_llama_cpp_runtime_public_dict,
    normalize_engine,
    parse_choices,
    pythonize_finqa_program,
    request_timeout_seconds_for_config,
    resolve_engine,
    run_generation,
    route_execution_status,
    sampling_policy_for_config,
    score_existing_output,
)
from finqa_metrics import finqa_equal
from experiment_7_stress_first25 import load_rows, select_stress_rows


def assert_close(actual: object, expected: float, tolerance: float = 1e-9) -> None:
    assert actual is not None, f"expected {expected}, got None"
    assert math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def engine_config(engine: str, route: str) -> EngineConfig:
    return EngineConfig(
        requested_engine=engine,
        engine=engine,
        route=route,
        model=engine,
        actual_model=engine,
        formal_model=engine,
        runtime_profile="test",
        endpoint="http://localhost:9/v1",
        api_version=None,
        api_key="EMPTY",
        missing_credentials=[],
        credential_sources={},
        credential_files=[],
        credential_warnings=[],
    )




def test_generator_batch_size_is_local_vllm_only() -> None:
    previous = os.environ.get("GENERATOR_BATCH_SIZE")
    try:
        os.environ["GENERATOR_BATCH_SIZE"] = "2"
        assert generator_batch_size_for_config(
            engine_config("qwen3_6", "local_vllm_openai_compatible")
        ) == 2
        assert generator_batch_size_for_config(
            engine_config("gpt5_5", "closed_api_openai")
        ) == 1
        os.environ["GENERATOR_BATCH_SIZE"] = "0"
        try:
            generator_batch_size_for_config(
                engine_config("qwen3_6", "local_vllm_openai_compatible")
            )
        except ValueError as exc:
            assert "must be >= 1" in str(exc)
        else:
            raise AssertionError("invalid generator batch size must fail")
    finally:
        if previous is None:
            os.environ.pop("GENERATOR_BATCH_SIZE", None)
        else:
            os.environ["GENERATOR_BATCH_SIZE"] = previous


def test_local_vllm_batch_generation_preserves_order_and_resume() -> None:
    previous_batch = os.environ.get("GENERATOR_BATCH_SIZE")
    original_generate_codes = finqa_runner.generate_codes
    lock = Lock()
    active = 0
    max_active = 0
    called: list[int] = []

    def fake_generate_codes(
        config: EngineConfig,
        prompt: str,
        profile: str,
        max_tokens: int = 128,
        example: dict[str, object] | None = None,
    ) -> list[str]:
        del config, profile, max_tokens, example
        nonlocal active, max_active
        index = int(prompt.strip().removeprefix("row-"))
        with lock:
            active += 1
            max_active = max(max_active, active)
            called.append(index)
        try:
            sleep(0.04 if index % 2 == 0 else 0.01)
            return [f"ans = {index}"]
        finally:
            with lock:
                active -= 1

    try:
        os.environ["GENERATOR_BATCH_SIZE"] = "2"
        finqa_runner.generate_codes = fake_generate_codes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            output_jsonl = root / "output.jsonl"
            input_json.write_text(
                json.dumps(
                    [
                        {
                            "id": str(index),
                            "question": f"question {index}?",
                            "prompt": f"row-{index}",
                            "answer": index,
                        }
                        for index in range(4)
                    ]
                ),
                encoding="utf-8",
            )
            config = engine_config("qwen3_6", "local_vllm_openai_compatible")
            first = run_generation(
                config,
                input_json,
                output_jsonl,
                "greedy",
                2,
                0.0,
                128,
                False,
            )
            assert first["rows"] == 2
            assert first["generator_batch_size"] == 2
            resumed = run_generation(
                config,
                input_json,
                output_jsonl,
                "greedy",
                -1,
                0.0,
                128,
                True,
            )
            rows = [
                json.loads(line)
                for line in output_jsonl.read_text(encoding="utf-8").splitlines()
            ]
        assert resumed["rows"] == 4
        assert resumed["resume_existing_output_rows"] == 2
        assert resumed["generator_batch_size"] == 2
        assert [row["id"] for row in rows] == ["0", "1", "2", "3"]
        assert [row["executed"] for row in rows] == [0, 1, 2, 3]
        assert all(row["generator_batch_size"] == 2 for row in rows)
        assert sorted(called) == [0, 1, 2, 3]
        assert max_active == 2
    finally:
        finqa_runner.generate_codes = original_generate_codes
        if previous_batch is None:
            os.environ.pop("GENERATOR_BATCH_SIZE", None)
        else:
            os.environ["GENERATOR_BATCH_SIZE"] = previous_batch


def test_resume_rejects_duplicate_and_placeholder_rows_without_modifying_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_json = root / "input.json"
        output_jsonl = root / "output.jsonl"
        examples = [
            {
                "id": str(index),
                "source_csv_row": index,
                "question": f"question {index}?",
                "prompt": f"row-{index}",
                "answer": index,
            }
            for index in range(3)
        ]
        input_json.write_text(json.dumps(examples), encoding="utf-8")
        corrupt_rows = [
            {**examples[0], "generated": ["ans = 0"], "executed": 0},
            {**examples[0], "generated": ["ans = 0"], "executed": 0},
            {"generated": [""], "executed": None},
        ]
        original = "".join(json.dumps(row) + "\n" for row in corrupt_rows)
        output_jsonl.write_text(original, encoding="utf-8")

        try:
            run_generation(
                engine_config("qwen3_6", "local_vllm_openai_compatible"),
                input_json,
                output_jsonl,
                "greedy",
                -1,
                0.0,
                128,
                True,
            )
        except ResumeOutputMismatch as exc:
            assert "identity_mismatch" in str(exc)
        else:
            raise AssertionError("corrupt resume output must be rejected")
        assert output_jsonl.read_text(encoding="utf-8") == original


class Obj:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def test_matched_dev_artifact_recovers_gold_from_finqa_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        matched_dir = Path(tmp) / "finqa_flan_o_finqa_dev"
        matched_dir.mkdir(parents=True)
        matched_path = matched_dir / "best_matched_with_retrieved_facts_and_questions.json"
        matched_path.write_text(
            json.dumps(
                [
                    {
                        "question": "what is the average payment volume per transaction for american express?",
                        "table_text": "matched table text",
                        "retrieved": ["retriever fact must be preserved"],
                        "answer": None,
                        "program": None,
                    }
                ]
            ),
            encoding="utf-8",
        )
        gold_path = Path(tmp) / "data" / "src" / "FinQA" / "dev.json"
        gold_path.parent.mkdir(parents=True)
        gold_path.write_text(
            json.dumps(
                [
                    {
                        "id": "V/2008/page_17.pdf-1",
                        "qa": {
                            "question": "what is the average payment volume per transaction for american express?",
                            "answer": 127.4,
                            "program": "divide(637, const_5)",
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        previous_root = few10_builder.WORKSPACE_ROOT
        few10_builder.WORKSPACE_ROOT = Path(tmp)
        try:
            rows = load_matched_examples(matched_path, -1)
        finally:
            few10_builder.WORKSPACE_ROOT = previous_root
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "V/2008/page_17.pdf-1"
    assert row["source_id"] == "V/2008/page_17.pdf-1"
    assert row["answer"] == 127.4
    assert row["program"] == "divide(637, const_5)"
    assert row["retrieved"] == ["retriever fact must be preserved"]
    assert row["target_gold_recovered_from_csv"] is True
    assert str(row["target_gold_csv"]).endswith("data/src/FinQA/dev.json")


def test_formal_target_gold_missing_reports_required_fields() -> None:
    missing = formal_target_gold_missing([{"question": "q?", "retrieved": ["r"]}])
    assert missing == [
        {
            "row_index": 0,
            "question": "q?",
            "missing": ["id", "answer", "program"],
            "source": None,
        }
    ]


def test_gpt_execute_chatmock_route_is_blocked_unless_diagnostic() -> None:
    updates = {
        "CHATMOCK_BASE_URL": "http://localhost:9/v1",
        "CHATMOCK_API_KEY": "key",
        "GPT5_5_CODEX_ROUTE": "chatmock",
        "GPT5_3_CODEX_ROUTE": "chatmock",
    }
    previous = {name: os.environ.get(name) for name in [*updates, "ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL"]}
    try:
        os.environ.update(updates)
        os.environ.pop("ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL", None)
        for engine in ("gpt5_5", "gpt5_3_codexS"):
            config = resolve_engine(engine, credential_purpose="execute")
            assert config.route == "codex_cli"
            assert route_execution_status(config) == "credential_blocked"
            assert "diagnostic-only" in config.missing_credentials[0]
        os.environ["ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL"] = "1"
        assert resolve_engine("gpt5_5", credential_purpose="execute").route == "chatmock_openai_compatible"
        assert resolve_engine("gpt5_3_codexS", credential_purpose="execute").route == "chatmock_openai_compatible"
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_new_local_generator_aliases() -> None:
    assert normalize_engine("deepseek") == "deepseek_r1_qwen32b"
    assert normalize_engine("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") == "deepseek_r1_qwen32b"
    assert normalize_engine("qwythos") == "qwythos9b"
    assert normalize_engine("empero-ai/Qwythos-9B-Claude-Mythos-5-1M") == "qwythos9b"
    assert normalize_engine("llama") == "llama4"
    assert normalize_engine("llama4_scout") == "llama4"
    assert normalize_engine("meta-llama/Llama-4-Scout-17B-16E-Instruct") == "llama4"


def test_new_vllm_model_path_overrides_model_id() -> None:
    updates = {
        "VLLM_BASE_URL": "http://localhost:9/v1",
        "DEEPSEEK_R1_MODEL": "wrong/deepseek-model-id",
        "DEEPSEEK_MODEL_PATH": "/tmp/fnqa-test/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "QWYTHOS_MODEL": "wrong/qwythos-model-id",
        "QWYTHOS_MODEL_PATH": "/tmp/fnqa-test/models/empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
        "LLAMA4_MODEL": "wrong/llama4-model-id",
        "LLAMA4_MODEL_PATH": "/tmp/fnqa-test/models/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    }
    previous = {name: os.environ.get(name) for name in updates}
    try:
        os.environ.update(updates)
        cases = {
            "deepseek": updates["DEEPSEEK_MODEL_PATH"],
            "qwythos": updates["QWYTHOS_MODEL_PATH"],
            "llama4": updates["LLAMA4_MODEL_PATH"],
        }
        for engine, expected_path in cases.items():
            config = resolve_engine(engine)
            assert config.route == "local_vllm_openai_compatible"
            assert config.actual_model == expected_path
            assert config.runtime_profile == "formal"
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_parse_choices_falls_back_to_reasoning_content() -> None:
    result = Obj(choices=[Obj(message=Obj(content="", model_extra={"reasoning_content": "ans = 94"}))])
    assert parse_choices(result) == ["ans = 94"]


def test_local_request_timeout_defaults_to_600_seconds() -> None:
    old_local = os.environ.pop("LOCAL_OPENAI_REQUEST_TIMEOUT_SECONDS", None)
    old_global = os.environ.pop("OPENAI_REQUEST_TIMEOUT_SECONDS", None)
    try:
        local_config = engine_config("llama3_3", "local_vllm_openai_compatible")
        gpt_config = engine_config("gpt5_5", "closed_api_openai")
        assert request_timeout_seconds_for_config(local_config) == 600.0
        assert request_timeout_seconds_for_config(gpt_config) == 120.0
        os.environ["LOCAL_OPENAI_REQUEST_TIMEOUT_SECONDS"] = "900"
        assert request_timeout_seconds_for_config(local_config) == 900.0
        os.environ["OPENAI_REQUEST_TIMEOUT_SECONDS"] = "33"
        assert request_timeout_seconds_for_config(local_config) == 33.0
    finally:
        if old_local is not None:
            os.environ["LOCAL_OPENAI_REQUEST_TIMEOUT_SECONDS"] = old_local
        else:
            os.environ.pop("LOCAL_OPENAI_REQUEST_TIMEOUT_SECONDS", None)
        if old_global is not None:
            os.environ["OPENAI_REQUEST_TIMEOUT_SECONDS"] = old_global
        else:
            os.environ.pop("OPENAI_REQUEST_TIMEOUT_SECONDS", None)


def test_bare_expression() -> None:
    prediction, cleaned = execute_codes(["5829 - 5735"], {"question": "what is the net change?"})
    assert_close(prediction, 94.0)
    assert cleaned == ["5829 - 5735\nans = 5829 - 5735"]


def test_answer_print() -> None:
    prediction, cleaned = execute_codes(["answer = 5829 - 5735; print(answer)"], {"question": "what is the net change?"})
    assert_close(prediction, 94.0)
    assert cleaned[0].endswith("ans = answer")


def test_think_block() -> None:
    prediction, cleaned = execute_codes(["<think>reasoning that must not execute</think>\nans = 94"], {"question": "what is the net change?"})
    assert_close(prediction, 94.0)
    assert cleaned == ["ans = 94"]


def test_mixed_reasoning_text_extracts_python_lines() -> None:
    generated = """The user wants to compute the answer.
Code should be:
step_0 = (261 / 6190)
ans = step_0
Done."""
    prediction, cleaned = execute_codes([generated], {"question": "what was the return on total assets?"})
    assert_close(prediction, 261 / 6190)
    assert cleaned == ["step_0 = (261 / 6190)\nans = step_0"]


def test_mixed_qwen_reasoning_extracts_named_assignments() -> None:
    generated = """The user wants the average payment volume per transaction.
From the table:
payments_volume_amex = 637
total_transactions_amex = 5.0
average_payment_volume_per_transaction = payments_volume_amex / total_transactions_amex
ans = average_payment_volume_per_transaction
Proceed."""
    prediction, cleaned = execute_codes(
        [generated],
        {"question": "what is the average payment volume per transaction for american express?"},
    )
    assert_close(prediction, 127.4)
    assert cleaned == [
        "payments_volume_amex = 637\n"
        "total_transactions_amex = 5.0\n"
        "average_payment_volume_per_transaction = payments_volume_amex / total_transactions_amex\n"
        "ans = average_payment_volume_per_transaction"
    ]


def test_mixed_reasoning_ignores_prose_label_assignments() -> None:
    generated = """Calculation: 637 / 5.0 = 127.4
payments_volume_amex = 637
total_transactions_amex = 5.0
ans = payments_volume_amex / total_transactions_amex"""
    prediction, cleaned = execute_codes(
        [generated],
        {"question": "what is the average payment volume per transaction for american express?"},
    )
    assert_close(prediction, 127.4)
    assert "Calculation:" not in cleaned[0]


def test_last_fenced_python_block_wins_after_think_cleanup() -> None:
    generated = """<think>private scratchpad</think>
```python
ans = 1
```
The first block is obsolete.
```python
step_0 = 5829 - 5735
ans = step_0
```"""
    prediction, cleaned = execute_codes([generated], {"question": "what is the net change?"})
    assert_close(prediction, 94.0)
    assert cleaned == ["step_0 = 5829 - 5735\nans = step_0"]


def test_finqa_dsl_translation_and_execution() -> None:
    program = "multiply(100, const_1000000), divide(25035519, #0)"
    translated = pythonize_finqa_program(program)
    assert "step_0 = (100 * 1000000)" in translated
    assert "step_1 = (25035519 / step_0)" in translated
    assert translated.endswith("ans = step_1")
    prediction, cleaned = execute_codes([program], {"question": "what is the ratio?"})
    assert_close(prediction, 0.25035519)
    assert cleaned == [translated]


def test_selected_examples_rebuild_prompt_as_python_pot() -> None:
    base_example = {
        "prompt": "stale full prompt should not be used",
        "table": "target table",
        "retrieved": ["2015 net revenue was 5829", "2014 net revenue was 5735"],
        "question": "what is the net change?",
        "selected_examples": [
            {
                "table": "shot table",
                "rel_fact": "fact",
                "question": "shot question?",
                "program": "multiply(1441499, 44.89)",
            }
        ],
    }
    prompt = build_prompt(base_example)
    assert finqa_runner.generator_system_prompt_for_example(base_example) == "Return only Python code that computes the answer."
    assert "stale full prompt" not in prompt
    assert "step_0 = (1441499 * 44.89)" in prompt
    assert "ans = step_0" in prompt
    assert "Generate only executable Python code." not in prompt
    assert "Assign the final result to a variable named ans." not in prompt
    assert "Do not provide investment advice." not in prompt
    assert "Do not explain in natural language after the code." not in prompt
    assert prompt.rstrip().endswith("#Python code below")

    dev_prompt = build_prompt(
        {
            **base_example,
            "source": "/tmp/finqa_flan_r_finqa_dev/best_matched_with_retrieved_facts_and_questions.json",
        }
    )
    assert finqa_runner.generator_system_prompt_for_example(
        {
            **base_example,
            "source": "/tmp/finqa_flan_r_finqa_dev/best_matched_with_retrieved_facts_and_questions.json",
        }
    ) == finqa_runner.GENERATOR_CODE_ONLY_SYSTEM_PROMPT
    assert "Generate only executable Python code." in dev_prompt
    assert "Assign the final result to a variable named ans." in dev_prompt
    assert "Do not provide investment advice." in dev_prompt
    assert "Do not explain in natural language after the code." in dev_prompt

    finqa10_prompt = build_prompt(
        {
            **base_example,
            "target_gold_csv": "/tmp/fnqa-test/data/testing/finqa_10_rel_fact_instruction.csv",
        }
    )
    assert "Generate only executable Python code." in finqa10_prompt

    finqa_test_prompt = build_prompt(
        {
            **base_example,
            "source": "/tmp/finqa_flan_r_finqa_test/best_matched_with_retrieved_facts_and_questions.json",
        }
    )
    assert "Generate only executable Python code." not in finqa_test_prompt
    assert finqa_runner.generator_system_prompt_for_example(
        {
            **base_example,
            "source": "/tmp/finqa_flan_r_finqa_test/best_matched_with_retrieved_facts_and_questions.json",
        }
    ) == "Return only Python code that computes the answer."


def test_example_selection_prompt_is_examples_only_python_pot() -> None:
    prompt = build_in_context_prompt(
        [
            {
                "table": "shot table",
                "rel_fact": "fact",
                "question": "shot question?",
                "program": "multiply(100, const_1000000), divide(25035519, #0)",
            }
        ]
    )
    assert "Question: shot question?" in prompt
    assert "step_0 = (100 * 1000000)" in prompt
    assert "step_1 = (25035519 / step_0)" in prompt
    assert prompt.rstrip().endswith("ans = step_1")
    assert "multiply(" not in prompt
    assert "#0" not in prompt


def test_unparseable_dsl_example_falls_back_to_gold_answer() -> None:
    prompt = format_example_prompt(
        {
            "table": "shot table",
            "rel_fact": "fact",
            "question": "shot question?",
            "program": "table_average(operating profit, none)",
            "answer": 123.0,
        },
        include_code=True,
    )
    assert "table_average(" not in prompt
    assert prompt.rstrip().endswith("ans = 123.0")


def test_prompt_scope_appends_target_like_finder() -> None:
    shot_prompt = format_example_prompt(
        {
            "table": "shot table",
            "rel_fact": "fact",
            "question": "shot question?",
            "program": "multiply(1441499, 44.89)",
        },
        include_code=True,
    )
    prompt = build_prompt(
        {
            "prompt": shot_prompt,
            "prompt_scope": "in_context_examples_only",
            "table": "target table",
            "retrieved": ["2015 net revenue was 5829", "2014 net revenue was 5735"],
            "question": "what is the net change?",
        }
    )
    assert "Question: shot question?" in prompt
    assert "Question: what is the net change?" in prompt
    assert "step_0 = (1441499 * 44.89)" in prompt
    assert prompt.count("#Python code below") == 2
    assert prompt.rstrip().endswith("#Python code below")


def cache_test_example(row_id: str = "10") -> dict[str, object]:
    return {
        "id": row_id,
        "source_csv_row": int(row_id) if row_id.isdigit() else row_id,
        "question": f"shot {row_id}",
        "answer": 1,
        "program": "ans = 1",
        "table": "",
        "rel_fact": "fact",
    }


def test_selection_cache_does_not_bind_dev_row_number_to_train_cache() -> None:
    target = {
        "id": "0",
        "source_id": "0",
        "source_csv_row": 0,
        "question": "what is dev volume?",
    }
    train_cache = {
        "kind": "experiment7_selection_cache",
        "source_mode": "finqa_train_formal",
        "items": {
            "0": {
                "selection_key": "0",
                "stable_id": "0",
                "source_csv_row": 0,
                "normalized_question": "what is train interest expense?",
                "selected_example_ids": ["10"],
            }
        },
    }
    example_lookup = {"10": cache_test_example("10")}
    strict_map = cache_items_by_key(train_cache)
    selected, cache_item, missing = selected_from_cache(target, strict_map, example_lookup, 4)
    assert selected == []
    assert cache_item is None
    assert missing == ["target_row_not_in_selection_cache"]

    legacy_map = cache_items_by_key(train_cache, allow_legacy_selection_binding=True)
    selected, cache_item, missing = selected_from_cache(
        target,
        legacy_map,
        example_lookup,
        4,
        allow_legacy_selection_binding=True,
    )
    assert len(selected) == 1
    assert missing == []
    assert isinstance(cache_item, dict)
    assert str(cache_item.get("_cache_lookup_key_type", "")).startswith("legacy_")


def test_selection_cache_exact_selection_key_still_resolves() -> None:
    target = {
        "id": "dev-0",
        "source_csv_row": 0,
        "question": "what is dev volume?",
    }
    key = selection_key_for_row(target)
    cache_payload = {
        "kind": "experiment7_selection_cache",
        "source_mode": "matched_retriever_artifact",
        "items": {
            key: {
                "selection_key": key,
                "stable_id": key,
                "source_csv_row": 0,
                "normalized_question": "what is dev volume?",
                "selected_example_ids": ["10"],
            }
        },
    }
    example_lookup = {"10": cache_test_example("10")}
    selected, cache_item, missing = selected_from_cache(target, cache_items_by_key(cache_payload), example_lookup, 4)
    assert len(selected) == 1
    assert missing == []
    assert isinstance(cache_item, dict)
    assert cache_item.get("_cache_lookup_key_type") == "selection_key"

def test_selection_cache_prefers_materialized_policy_examples_over_train_row_ids() -> None:
    target = {
        "id": "dev-0",
        "source_csv_row": 0,
        "question": "what is dev volume?",
    }
    key = selection_key_for_row(target)
    cache_payload = {
        "kind": "experiment7_selection_cache",
        "source_mode": "matched_retriever_artifact",
        "items": {
            key: {
                "selection_key": key,
                "stable_id": key,
                "source_csv_row": 0,
                "normalized_question": "what is dev volume?",
                "selected_example_ids": ["1161"],
                "selected_examples": [
                    {
                        "id": "1161",
                        "question": "policy candidate question must win",
                        "answer": 0.41784,
                        "program": "ans = 0.41784",
                        "table": "policy table",
                        "rel_fact": "policy fact",
                    }
                ],
            }
        },
    }
    example_lookup = {
        "1161": {
            **cache_test_example("1161"),
            "question": "wrong train csv row question",
            "program": "ans = 999",
        }
    }
    selected, cache_item, missing = selected_from_cache(target, cache_items_by_key(cache_payload), example_lookup, 4)
    assert len(selected) == 1
    assert missing == []
    assert isinstance(cache_item, dict)
    assert selected[0]["question"] == "policy candidate question must win"
    assert selected[0]["program"] == "ans = 0.41784"
    assert selected[0]["selection_source"] == "experiment7_selection_cache_materialized"


def test_materialized_selection_requires_exact_key_and_passed_audit() -> None:
    target = {
        "id": "dev-0",
        "source_csv_row": 0,
        "question": "what is dev volume?",
    }
    key = selection_key_for_row(target)
    passed_materialized = {
        "selection_key": key,
        "question": "what is dev volume?",
        "selected_examples": [cache_test_example("1161")],
        "example_selection": {"selection_binding_status": "passed"},
    }
    selected, missing = selected_from_materialized_row(target, passed_materialized, 4)
    assert len(selected) == 1
    assert missing == []

    wrong_key = dict(passed_materialized, selection_key="row:1|question:what is dev volume?")
    selected, missing = selected_from_materialized_row(target, wrong_key, 4)
    assert selected == []
    assert missing == ["materialized_selection_key_mismatch"]

    not_passed = dict(passed_materialized, example_selection={"selection_binding_status": "blocked_example_selection_cache"})
    selected, missing = selected_from_materialized_row(target, not_passed, 4)
    assert selected == []
    assert missing == ["materialized_selection_binding_not_passed"]


def test_materialized_selection_summary_reports_zero_row_collision() -> None:
    target = {
        "id": "dev-0",
        "source_csv_row": 0,
        "question": "what is dev volume?",
        "answer": 1,
        "program": "ans = 1",
    }
    key = selection_key_for_row(target)
    materialized = {
        **target,
        "selection_key": key,
        "selected_examples": [cache_test_example("1161")],
        "example_selection": {"selection_binding_status": "passed"},
    }
    args = Obj(
        shot_number=4,
        formal_finder_ready=True,
        selection_mode="cache",
        selection_cache_json=Path("selection_cache.json"),
        materialized_selection_jsonl=Path("materialized_selected_examples.jsonl"),
    )
    output, summary = attach_materialized_selection([target], [materialized], args, "matched_retriever_artifact")
    assert output[0]["example_selection"]["selection_binding_status"] == "passed"
    assert output[0]["example_selection"]["cache_match_key_type"] == "selection_key"
    assert summary["cache_match_key_types"] == {"selection_key": 1}
    assert summary["legacy_cache_alias_rows"] == 0
    assert summary["row_number_collision_rows"] == 0


def test_materialized_selection_backfills_missing_cached_retriever_row_identity() -> None:
    target = {
        "id": "ETR/2016/page_23.pdf-2",
        "question": "what is the net change in net revenue during 2015 for entergy corporation?",
        "retrieved": ["retriever output must be preserved"],
    }
    materialized = {
        "id": "0",
        "source_csv_row": 0,
        "selection_key": (
            "row:0|question:what is the net change in net revenue during 2015 for entergy corporation?"
        ),
        "question": target["question"],
        "selected_examples": [cache_test_example("1161")],
        "example_selection": {"selection_binding_status": "passed"},
    }
    args = Obj(
        shot_number=4,
        formal_finder_ready=True,
        selection_mode="cache",
        selection_cache_json=Path("selection_cache.json"),
        materialized_selection_jsonl=Path("materialized_selected_examples.jsonl"),
    )

    output, summary = attach_materialized_selection(
        [target], [materialized], args, "matched_retriever_artifact"
    )

    assert output[0]["id"] == target["id"]
    assert output[0]["source_csv_row"] == 0
    assert output[0]["selection_key"] == materialized["selection_key"]
    assert output[0]["retrieved"] == target["retrieved"]
    assert output[0]["example_selection"]["selection_binding_status"] == "passed"
    assert summary["identity_backfill_rows"] == 1
    assert summary["cache_rows"] == 1


def test_materialized_selection_does_not_backfill_mismatched_question() -> None:
    target = {"id": "target-id", "question": "target question"}
    materialized = {
        "source_csv_row": 0,
        "selection_key": "row:0|question:different question",
        "question": "different question",
        "selected_examples": [cache_test_example("1161")],
        "example_selection": {"selection_binding_status": "passed"},
    }
    args = Obj(
        shot_number=4,
        formal_finder_ready=True,
        selection_mode="cache",
        selection_cache_json=Path("selection_cache.json"),
        materialized_selection_jsonl=Path("materialized_selected_examples.jsonl"),
    )
    output, summary = attach_materialized_selection(
        [target], [materialized], args, "matched_retriever_artifact"
    )
    assert output[0]["example_selection"]["selection_binding_status"] == "blocked_example_selection_cache"
    assert summary["identity_backfill_rows"] == 0


def test_local_sampling_policy_qwen_precise_coding() -> None:
    config = engine_config("qwen3_6", "local_vllm_openai_compatible")
    env_names = [
        "QWEN3_6_ENABLE_THINKING",
        "QWEN3_6_TEMPERATURE",
        "QWEN3_6_TOP_P",
        "QWEN3_6_TOP_K",
    ]
    old_values = {name: os.environ.pop(name, None) for name in env_names}
    try:
        policy = sampling_policy_for_config(config, "greedy")
        assert policy.temperature == 0.6
        assert policy.top_p == 0.95
        assert policy.top_k == 20
        assert policy.min_p == 0.0
        assert policy.presence_penalty == 0.0
        assert policy.repetition_penalty == 1.0
        assert policy.enable_thinking is True
        kwargs = chat_completion_kwargs_for_sampling(config, policy)
        assert kwargs["temperature"] == 0.6
        assert kwargs["top_p"] == 0.95
        assert kwargs["presence_penalty"] == 0.0
        assert kwargs["extra_body"] == {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        os.environ["QWEN3_6_ENABLE_THINKING"] = "false"
        os.environ["QWEN3_6_TEMPERATURE"] = "0.0"
        os.environ["QWEN3_6_TOP_P"] = "1.0"
        os.environ["QWEN3_6_TOP_K"] = "1"
        override_policy = sampling_policy_for_config(config, "greedy")
        assert override_policy.temperature == 0.0
        assert override_policy.top_p == 1.0
        assert override_policy.top_k == 1
        assert override_policy.enable_thinking is False
        override_kwargs = chat_completion_kwargs_for_sampling(config, override_policy)
        assert override_kwargs["temperature"] == 0.0
        assert override_kwargs["top_p"] == 1.0
        assert override_kwargs["extra_body"]["top_k"] == 1
        assert override_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value



def test_percentage_equivalent_is_diagnostic_not_formal_ea() -> None:
    assert not finqa_equal(0.2469, 24.69, False)
    assert finqa_equal(0.2469, 24.69, True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_json = tmp_path / "input.json"
        output_jsonl = tmp_path / "output.jsonl"
        input_json.write_text(
            json.dumps([{"question": "what percentage comes from canada?", "answer": 24.69}]),
            encoding="utf-8",
        )
        output_jsonl.write_text(
            json.dumps({"generated": ["ans = 60 / 243"], "executed": 0.2469, "answer": 24.69}) + "\n",
            encoding="utf-8",
        )
        result = score_existing_output(input_json, output_jsonl, -1)

    assert result["execution_accuracy"] == 0.0
    assert result["raw_execution_accuracy"] == 0.0
    assert result["percentage_equivalent_accuracy"] == 1.0
    assert result["percentage_equivalent_metric_scope"] == "diagnostic_only"
    assert result["score_policy"]["formal_execution_accuracy"] == "strict_finder"

def test_percent_scale_multiply_at_line_end_is_dev_only_extension() -> None:
    line_end_scale_code = "ans = 60 / 243 * 100"
    original_finder_scale_code = "ans = 60 / 243 * 100\n"
    dev_example = {"target_gold_csv": "/tmp/data/src/FinQA/dev.json"}
    test_example = {"target_gold_csv": "/tmp/data/src/FinQA/test.json"}

    assert generated_code_has_percent_scale_multiply(line_end_scale_code, dev_example)
    assert not generated_code_has_percent_scale_multiply(line_end_scale_code, test_example)
    assert generated_code_has_percent_scale_multiply(original_finder_scale_code, test_example)

    cleaned_code = "canada_total = 60\ntotal = 243\nans = canada_total / total"
    raw_reasoning = "Percentage = (60 / 243) * 100 = 24.69"
    assert not generated_code_has_percent_scale_multiply(cleaned_code, dev_example, raw_reasoning)


def test_local_sampling_policy_llama_official_default() -> None:
    config = engine_config("llama3_3", "local_vllm_openai_compatible")
    policy = sampling_policy_for_config(config, "greedy")
    assert policy.temperature == 0.6
    assert policy.top_p == 0.9
    assert chat_completion_kwargs_for_sampling(config, policy) == {"temperature": 0.6, "top_p": 0.9, "n": 1}


def test_local_sampling_policy_new_vllm_generators() -> None:
    deepseek_config = engine_config("deepseek_r1_qwen32b", "local_vllm_openai_compatible")
    deepseek_policy = sampling_policy_for_config(deepseek_config, "greedy")
    assert deepseek_policy.temperature == 0.6
    assert deepseek_policy.top_p == 0.95
    assert chat_completion_kwargs_for_sampling(deepseek_config, deepseek_policy) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 1,
    }

    qwythos_config = engine_config("qwythos9b", "local_vllm_openai_compatible")
    qwythos_policy = sampling_policy_for_config(qwythos_config, "greedy")
    assert qwythos_policy.temperature == 0.6
    assert qwythos_policy.top_p == 0.95
    assert chat_completion_kwargs_for_sampling(qwythos_config, qwythos_policy) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 1,
        "extra_body": {"top_k": 20, "repetition_penalty": 1.05},
    }

    llama4_config = engine_config("llama4", "local_vllm_openai_compatible")
    llama4_policy = sampling_policy_for_config(llama4_config, "greedy")
    assert llama4_policy.temperature == 0.6
    assert llama4_policy.top_p == 0.9
    assert chat_completion_kwargs_for_sampling(llama4_config, llama4_policy) == {
        "temperature": 0.6,
        "top_p": 0.9,
        "n": 1,
    }


def test_qwythos_effective_max_tokens_defaults_to_8192() -> None:
    config = engine_config("qwythos9b", "local_vllm_openai_compatible")
    old_value = os.environ.pop("QWYTHOS_MAX_TOKENS", None)
    try:
        assert effective_max_tokens_for_config(config, 128) == 8192
        os.environ["QWYTHOS_MAX_TOKENS"] = "2048"
        assert effective_max_tokens_for_config(config, 128) == 2048
        assert effective_max_tokens_for_config(engine_config("qwen3_6", "local_vllm_openai_compatible"), 128) == 128
    finally:
        if old_value is None:
            os.environ.pop("QWYTHOS_MAX_TOKENS", None)
        else:
            os.environ["QWYTHOS_MAX_TOKENS"] = old_value


def test_local_sampling_policy_mistral_llama_cpp_stable_python() -> None:
    config = engine_config("mistral4", "local_llama_cpp_openai_compatible")
    policy = sampling_policy_for_config(config, "greedy")
    assert policy.temperature == 0.2
    assert policy.reasoning_effort is None
    assert policy.reasoning_effort_intent == "none"
    kwargs = chat_completion_kwargs_for_sampling(config, policy)
    assert kwargs == {"temperature": 0.2, "top_p": 1.0, "n": 1}


def test_gpt_sampling_policy_keeps_greedy_profile() -> None:
    config = engine_config("gpt5_5", "codex_cli")
    policy = sampling_policy_for_config(config, "greedy")
    assert policy.temperature == 0.0
    assert policy.top_p == 1.0
    assert policy.top_k is None
    assert chat_completion_kwargs_for_sampling(config, policy) == {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "reasoning_effort": "medium",
    }


def test_codex_cli_system_prompt_preserves_test_flow() -> None:
    config = engine_config("gpt5_5", "codex_cli")
    original = finqa_runner.generate_text_with_codex_cli
    captured: list[str] = []

    def fake_generate_text_with_codex_cli(
        config: EngineConfig,
        prompt: str,
        system_prompt: str,
        profile: str,
    ) -> list[str]:
        del config, prompt, profile
        captured.append(system_prompt)
        return ["ans = 1"]

    try:
        finqa_runner.generate_text_with_codex_cli = fake_generate_text_with_codex_cli
        finqa_runner.generate_codes_with_codex_cli(
            config,
            "prompt",
            "greedy",
            example={"source": "/tmp/finqa_flan_r_finqa_test/best_matched.json"},
        )
        assert captured[-1] == "Return only Python code that computes the answer."
        finqa_runner.generate_codes_with_codex_cli(
            config,
            "prompt",
            "greedy",
            example={"source": "/tmp/finqa_flan_r_finqa_dev/best_matched.json"},
        )
        assert captured[-1] == finqa_runner.GENERATOR_CODE_ONLY_SYSTEM_PROMPT
    finally:
        finqa_runner.generate_text_with_codex_cli = original


def test_codex_cli_text_route_disables_image_generation() -> None:
    config = engine_config("gpt5_3_codexS", "codex_cli")
    old_features = os.environ.pop("CODEX_CLI_DISABLED_FEATURES", None)
    old_tier = os.environ.get("CODEX_CLI_SERVICE_TIER")
    try:
        os.environ["CODEX_CLI_SERVICE_TIER"] = "fast"
        command = finqa_runner.build_codex_cli_command(config, "/usr/local/bin/codex", Path("/tmp/out.txt"))
        assert command[:4] == ["/usr/local/bin/codex", "exec", "-c", "service_tier=\"fast\""]
        disable_index = command.index("--disable")
        assert command[disable_index : disable_index + 2] == ["--disable", "image_generation"]
        assert command[-1] == "-"
    finally:
        if old_features is None:
            os.environ.pop("CODEX_CLI_DISABLED_FEATURES", None)
        else:
            os.environ["CODEX_CLI_DISABLED_FEATURES"] = old_features
        if old_tier is None:
            os.environ.pop("CODEX_CLI_SERVICE_TIER", None)
        else:
            os.environ["CODEX_CLI_SERVICE_TIER"] = old_tier


def test_llama_cpp_runtime_metadata_includes_q4km_knobs() -> None:
    updates = {
        "LLAMA_CPP_MODEL_PATH": "/models/UD-Q4_K_M/model-00001.gguf",
        "LLAMA_CPP_QUANT": "UD-Q4_K_M",
        "LLAMA_CPP_CACHE_TYPE_K": "q8_0",
        "LLAMA_CPP_CACHE_TYPE_V": "q8_0",
        "LLAMA_CPP_CPU_MOE": "1",
        "LLAMA_CPP_FIT_TARGET": "2048,2048",
        "LLAMA_CPP_OP_OFFLOAD": "off",
        "LLAMA_CPP_FLASH_ATTN": "off",
        "LLAMA_CPP_CACHE_RAM": "0",
        "LLAMA_CPP_DEVICE": "CUDA0",
        "LLAMA_CPP_MAIN_GPU": "0",
    }
    previous = {name: os.environ.get(name) for name in updates}
    try:
        os.environ.update(updates)
        runtime = local_llama_cpp_runtime_public_dict()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    values = runtime["values"]
    assert runtime["profile"] == "llama_cpp_UD-Q4_K_M"
    for name, value in updates.items():
        assert values[name] == value


def test_stress_first25_prefers_baseline_and_long_cases() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dirs = [
            root / "generator" / "gpt5_5" / "finqa_flan_o_finqa_test",
            root / "generator" / "gpt5_5" / "finqa_flan_z_finqa_test",
        ]
        rows_by_case = [
            [
                {"id": "ETR/2016/page_23.pdf-2", "selection_key": "ETR/2016/page_23.pdf-2", "table": "t", "retrieved": ["r"], "question": "q?", "program": "ans = 5829 - 5735", "answer": 94},
                {"id": "long-prompt-test", "selection_key": "same-question", "table": "x" * 200, "retrieved": ["r" * 300], "question": "q?", "program": "add(1, 2)", "answer": 3},
            ],
            [
                {"id": "long-code-dev", "selection_key": "same-question", "table": "t", "retrieved": ["z"], "question": "q?", "program": "multiply(100, const_1000000), divide(25035519, #0)", "answer": 0.25035519},
                {"id": "short-dev", "selection_key": "short-dev", "table": "t", "retrieved": ["z"], "question": "q?", "program": "ans = 1", "answer": 1},
            ],
        ]
        paths = []
        for case_dir, rows in zip(case_dirs, rows_by_case):
            case_dir.mkdir(parents=True)
            path = case_dir / "generator_input.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            paths.append(path)
        rows = load_rows(paths)
        selected = select_stress_rows(rows, 6)
        selected_ids = {row.get("id") for row in selected}
        assert selected[0]["id"] == "ETR/2016/page_23.pdf-2"
        assert "long-prompt-test" in selected_ids
        assert "long-code-dev" in selected_ids
        reasons = [reason for row in selected for reason in row["stress_sample"]["selection_reasons"]]
        assert any(reason.startswith("retfact_prompt_type_comparison") for reason in reasons)


def test_formal_gate_defaults_to_first10_and_renamed() -> None:
    runner = (Path(__file__).resolve().parent / "experiment_7_runner.sh").read_text(encoding="utf-8")
    assert 'FIRST_GATE_LIMIT="${FIRST_GATE_LIMIT:-${FIRST50_GATE_LIMIT:-10}}"' in runner
    assert '_first10' in runner
    assert '_first50' not in runner
    assert 'first50 gate' not in runner
    assert 'expected 50' not in runner
    assert 'expected 10' in runner


def test_formal_gpt_workers_gate_chatmock_route() -> None:
    runner = (Path(__file__).resolve().parent / "experiment_7_runner.sh").read_text(encoding="utf-8")
    assert 'EXPERIMENT7_USE_CHATMOCK_ROUTE="${EXPERIMENT7_USE_CHATMOCK_ROUTE:-1}"' in runner
    assert 'if [[ "${EXPERIMENT7_USE_CHATMOCK_ROUTE}" == "1" ]]' in runner
    assert 'ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 GPT5_5_CODEX_ROUTE=${GPT5_5_CODEX_ROUTE:-chatmock}' in runner
    assert 'ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 GPT5_3_CODEX_ROUTE=${GPT5_3_CODEX_ROUTE:-chatmock}' in runner
    assert 'GPT5_5_CODEX_ROUTE=${GPT5_5_CODEX_ROUTE:-openai}' in runner
    assert 'GPT5_3_CODEX_ROUTE=${GPT5_3_CODEX_ROUTE:-api_key}' in runner


def test_ea_summary_blocks_chatmock_formal_claim() -> None:
    generator = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    assert 'diagnostic_backends = {"chatmock", "chatmock_openai_compatible"}' in generator
    assert "formal_claim_ready = formal_finder_ready and formal_full_row_count and formal_route_ready" in generator
    assert '"formal_route_ready": formal_route_ready' in generator
    assert "ChatMock output is diagnostic-only and must not be claimed as formal GPT-series EA." in generator


def test_formal_runner_defaults_to_full_12_retriever_matrix() -> None:
    runner = (Path(__file__).resolve().parent / "experiment_7_runner.sh").read_text(encoding="utf-8")
    generator = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    assert 'DEFAULT_EXPERIMENT7_ENGINES="qwen3_6 llama4 gpt4_1 gpt5_3_codexS gpt5_5"' in generator
    assert 'DEFAULT_EXPERIMENT7_ENGINES="qwen3_6 mistral4 llama3_3' not in generator
    assert 'EXPERIMENT7_USE_ACTIVE_RETFACT_SCOPE="${EXPERIMENT7_USE_ACTIVE_RETFACT_SCOPE:-0}"' in runner
    assert 'DEFAULT_EXPERIMENT7_MATRIX="${DEFAULT_EXPERIMENT7_FULL_MATRIX}"' in runner
    marker = 'DEFAULT_EXPERIMENT7_FULL_MATRIX="'
    start = runner.index(marker) + len(marker)
    end = runner.index('"', start)
    matrix = runner[start:end].split()
    assert len(matrix) == 24
    expected_retrievers = {
        "finqa_flan_o",
        "finqa_flan_z",
        "finqa_flan_m",
        "finqa_flan_d",
        "finqa_mistral_o",
        "finqa_mistral_z",
        "finqa_mistral_m",
        "finqa_mistral_d",
        "finqa_t5gemma2_o",
        "finqa_t5gemma2_z",
        "finqa_t5gemma2_m",
        "finqa_t5gemma2_d",
    }
    assert {item.split(":", 1)[0] for item in matrix} == expected_retrievers
    assert {item.split(":", 1)[1] for item in matrix} == {"finqa_test", "finqa_dev"}


def test_formal_runner_does_not_auto_backfill_or_start_chatmock_service() -> None:
    runner = (Path(__file__).resolve().parent / "experiment_7_runner.sh").read_text(encoding="utf-8")
    assert 'EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL="${EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL:-0}"' in runner
    assert 'formal run requires retriever-specific matched-json' in runner
    assert 'EXPERIMENT7_USE_CHATMOCK_SERVICE="${EXPERIMENT7_USE_CHATMOCK_SERVICE:-0}"' in runner
    assert 'for required_session in API_key chatmock vllm/llama_cpp run monitor; do' in runner
    assert 'if [[ "${EXPERIMENT7_USE_CHATMOCK_SERVICE}" == "1" ]]; then' in runner
    assert 'window_pairs+=("chatmock:${WINDOW_PREFIX}_chatmock_service")' in runner


def test_public_release_dry_run_uses_nonruntime_contracts() -> None:
    directory = Path(__file__).resolve().parent
    reproduce = (directory / "reproduce.sh").read_text(encoding="utf-8")
    experiment6 = (directory / "experiment6_v4.py").read_text(encoding="utf-8")
    experiment7 = (directory / "experiment_7_runner.sh").read_text(encoding="utf-8")
    assert 'experiment_6.sh" public-preflight' in reproduce
    assert "PUBLIC_PREFLIGHT_ONLY=1 PREFLIGHT_ONLY=1" in reproduce
    assert '"public-preflight"' in experiment6
    public_branch = 'if [[ "${PUBLIC_PREFLIGHT_ONLY:-0}" == "1" ]]'
    formal_branch = 'if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]'
    assert public_branch in experiment7
    assert formal_branch in experiment7
    assert experiment7.index(public_branch) < experiment7.index(formal_branch)


def test_validation_only_score_report_is_not_failure() -> None:
    generator = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    assert 'route_status == "validation_completed_execute_disabled"' in generator
    assert '"validation_only"' in generator
    assert 'validation_completed_execute_disabled" else "blocked"' in generator


def test_in_context_selection_defaults_to_existing_finder_train_csv() -> None:
    script = (Path(__file__).resolve().parent / "experiment_7_in_context_selection.sh").read_text(encoding="utf-8")
    assert 'data/src/FINDER/finqa_train_rel_fact_instruction.csv' in script
    assert 'data/finqa_original/finqa_train_rel_fact_instruction.csv' in script
    assert '*/data/src/FINDER/finqa_train_rel_fact_instruction.csv' in script


def test_generator_answer_accepts_existing_finder_selection_cache_source() -> None:
    script = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    assert '"/data/src/FINDER/"' in script
    assert "source_mode == expected or inferred_source_mode == expected" in script


def test_generator_answer_dev_only_binding_fix_preserves_test_flow() -> None:
    script = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    assert 'is_dev_dataset_id()' in script
    assert 'if [[ -z "${expected_source_mode}" ]] && is_dev_dataset_id "${dataset_id}"; then' in script
    assert 'expected_source_mode="matched_retriever_artifact"' in script
    assert 'selection_cache_matches_dataset_scope()' in script
    assert '*_finqa_dev/*|*/finqa_dev/*) return 1 ;;' in script
    assert 'if ! is_dev_dataset_id "${dataset_id}"; then' in script
    assert 'case_allow_legacy_selection_binding="1"' in script
    assert 'case_require_target_selection_cache="0"' in script
    assert 'case_allow_materialized_selection_cache="1"' in script
    assert 'if [[ "${case_require_target_selection_cache}" == "1" && -z "${formal_csv_source}" ]]; then' in script
    assert 'if [[ "${case_allow_materialized_selection_cache}" == "1" ]]; then' in script
    assert 'if [[ "${RETRIEVER_MAX_INFER_SAMPLES}" != "-1" ]] && is_dev_dataset_id "${dataset_id}"; then' in script
    assert 'QWEN3_6_DEV_ENABLE_THINKING="${QWEN3_6_DEV_ENABLE_THINKING:-${QWEN3_6_ENABLE_THINKING}}"' in script


def test_extra_raw_retriever_routes_are_supported_but_not_default() -> None:
    generator = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    binding = (Path(__file__).resolve().parent / "experiment_7_selection_cache_binding.sh").read_text(encoding="utf-8")
    refresh = (Path(__file__).resolve().parent / "experiment_7_refresh_dev_retfacts.sh").read_text(encoding="utf-8")
    assert 'finqa_flan_r|flan_r) printf "finqa_flan_r' in generator
    assert 'finqa_Mistral_r|finqa_mistral_r|mistral_r) printf "finqa_mistral_r' in generator
    assert '*_r) printf "raw' in generator
    assert 'raw) printf "src/FINDER' in binding
    assert '*_r) printf "raw' in refresh
    default_matrix = generator.split('DEFAULT_EXPERIMENT7_MATRIX="', 1)[1].split('"', 1)[0]
    assert "finqa_flan_r" not in default_matrix
    assert "finqa_mistral_r" not in default_matrix
    assert 'EXPERIMENT7_EA_ID_SUFFIX="${EXPERIMENT7_EA_ID_SUFFIX:-}"' in generator
    assert 'EA_ID_SUFFIX="${EXPERIMENT7_EA_ID_SUFFIX}"' in generator


def test_diagnostic_ea_can_preserve_global_latest_pointer() -> None:
    generator = (Path(__file__).resolve().parent / "experiment_7_generator_answer.sh").read_text(encoding="utf-8")
    assert 'UPDATE_EA_LATEST="${UPDATE_EA_LATEST:-1}"' in generator
    assert 'EA_UPDATE_LATEST="${UPDATE_EA_LATEST}"' in generator
    assert 'if os.environ.get("EA_UPDATE_LATEST") == "1":' in generator
    assert 'UPDATE_EA_LATEST=%s' in generator


def test_local_terminal_state_unblocks_blocked_dependencies() -> None:
    runner = (Path(__file__).resolve().parent / "experiment_7_runner.sh").read_text(encoding="utf-8")
    assert "wait_for_terminal_state gpt55" not in runner
    assert "wait_for_terminal_state qwen" in runner
    assert "status in {'completed', 'blocked'}" in runner
    assert "qwen_vllm.state.json" in runner
    assert "llama4.state.json" in runner
    assert "Authorization" in runner
    assert "VLLM_API_KEY" in runner
    assert "answer_llama4.smoke_started" in runner
    assert "llama33.state.json" not in runner
    assert "answer_llama3_3.full_started" not in runner


def test_generated_code_execution_requires_explicit_opt_in() -> None:
    previous = os.environ.pop("FQAN_ALLOW_GENERATED_CODE_EXECUTION", None)
    try:
        try:
            execute_python_code("ans = 1")
        except RuntimeError as exc:
            assert "disabled" in str(exc)
        else:
            raise AssertionError("generated code execution should be disabled by default")
    finally:
        if previous is not None:
            os.environ["FQAN_ALLOW_GENERATED_CODE_EXECUTION"] = previous


def test_formal_launcher_delegates_to_fqan_runner() -> None:
    launcher = (Path(__file__).resolve().parent / "experiment_7_formal_tmux_run.sh").read_text(encoding="utf-8")
    assert 'exec "${SCRIPT_DIR}/experiment_7_runner.sh"' in launcher
    assert "FORMAL_ENGINES is deprecated" in launcher


if __name__ == "__main__":
    for test in [
        test_matched_dev_artifact_recovers_gold_from_finqa_json,
        test_formal_target_gold_missing_reports_required_fields,
        test_gpt_execute_chatmock_route_is_blocked_unless_diagnostic,
        test_new_local_generator_aliases,
        test_new_vllm_model_path_overrides_model_id,
        test_parse_choices_falls_back_to_reasoning_content,
        test_local_request_timeout_defaults_to_600_seconds,
        test_generator_batch_size_is_local_vllm_only,
        test_local_vllm_batch_generation_preserves_order_and_resume,
        test_resume_rejects_duplicate_and_placeholder_rows_without_modifying_output,
        test_bare_expression,
        test_answer_print,
        test_think_block,
        test_mixed_reasoning_text_extracts_python_lines,
        test_last_fenced_python_block_wins_after_think_cleanup,
        test_finqa_dsl_translation_and_execution,
        test_selected_examples_rebuild_prompt_as_python_pot,
        test_example_selection_prompt_is_examples_only_python_pot,
        test_unparseable_dsl_example_falls_back_to_gold_answer,
        test_prompt_scope_appends_target_like_finder,
        test_selection_cache_does_not_bind_dev_row_number_to_train_cache,
        test_selection_cache_exact_selection_key_still_resolves,
        test_selection_cache_prefers_materialized_policy_examples_over_train_row_ids,
        test_materialized_selection_requires_exact_key_and_passed_audit,
        test_materialized_selection_summary_reports_zero_row_collision,
        test_local_sampling_policy_qwen_precise_coding,
        test_percentage_equivalent_is_diagnostic_not_formal_ea,
        test_percent_scale_multiply_at_line_end_is_dev_only_extension,
        test_local_sampling_policy_llama_official_default,
        test_local_sampling_policy_new_vllm_generators,
        test_qwythos_effective_max_tokens_defaults_to_8192,
        test_local_sampling_policy_mistral_llama_cpp_stable_python,
        test_gpt_sampling_policy_keeps_greedy_profile,
        test_codex_cli_system_prompt_preserves_test_flow,
        test_codex_cli_text_route_disables_image_generation,
        test_llama_cpp_runtime_metadata_includes_q4km_knobs,
        test_stress_first25_prefers_baseline_and_long_cases,
        test_formal_gate_defaults_to_first10_and_renamed,
        test_formal_gpt_workers_gate_chatmock_route,
        test_ea_summary_blocks_chatmock_formal_claim,
        test_formal_runner_defaults_to_full_12_retriever_matrix,
        test_formal_runner_does_not_auto_backfill_or_start_chatmock_service,
        test_public_release_dry_run_uses_nonruntime_contracts,
        test_validation_only_score_report_is_not_failure,
        test_in_context_selection_defaults_to_existing_finder_train_csv,
        test_generator_answer_accepts_existing_finder_selection_cache_source,
        test_generator_answer_dev_only_binding_fix_preserves_test_flow,
        test_extra_raw_retriever_routes_are_supported_but_not_default,
        test_local_terminal_state_unblocks_blocked_dependencies,
        test_generated_code_execution_requires_explicit_opt_in,
        test_formal_launcher_delegates_to_fqan_runner,
    ]:
        test()
    print("target_execution_regression=pass")
