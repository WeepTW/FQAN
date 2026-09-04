#!/usr/bin/env python3
"""Smoke checks for Experiment 6 routing and prediction contracts.

This does not load models or write experiment artifacts. It locks the routing
rules that prevent RetFact-only retriever output from being scored as data
binding predictions.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_experiment6_binding_generation import (  # noqa: E402
    MatrixCase,
    MissingBindingGeneratorError,
    completed_metadata_is_usable,
    completed_run_metadata_is_usable,
    family_from_source_id,
    is_finetuned_retriever_source,
    require_binding_generation_contract,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    fine_tuned_case = MatrixCase("6_mistral_z", "finqa_mistral_z", "narrative_zero_shot")
    base_case = MatrixCase("6_mistral_base_z", "mistral_v0_3", "narrative_zero_shot")
    generator_case = MatrixCase("6_mistral4_z", "mistral4", "narrative_zero_shot")
    args = Namespace(num_runs=1, top_k=3)

    assert_true(is_finetuned_retriever_source("finqa_mistral_z"), "finqa_mistral_z must be fine-tuned")
    assert_true(family_from_source_id("finqa_mistral_z") == "mistral", "finqa_mistral_z family")
    assert_true(family_from_source_id("mistral_v0_3") == "mistral", "mistral_v0_3 base family")
    assert_true(family_from_source_id("flan_t5_large") == "flan", "flan_t5_large base family")
    assert_true(family_from_source_id("t5gemma_2_1b_1b") == "t5gemma2", "t5gemma base family")
    assert_true(family_from_source_id("mistral4") is None, "mistral4 must route to generator runtime")
    assert_true(family_from_source_id("qwen3_6") is None, "qwen3_6 must route to generator runtime")
    assert_true(family_from_source_id("llama3_3") is None, "llama3_3 must route to generator runtime")

    no_adapter_finetuned_metadata = {
        "status": "completed",
        "formal_result": True,
        "num_runs": 1,
        "top_k": 3,
        "runtime": {"family": "mistral", "use_adapter": False},
    }
    assert_true(
        not completed_metadata_is_usable(no_adapter_finetuned_metadata, fine_tuned_case, args),
        "fine-tuned source without adapter must not resume as completed",
    )

    old_retfact_metadata = {
        "status": "completed",
        "formal_result": True,
        "num_runs": 1,
        "top_k": 3,
        "generation_mode": "retriever",
        "runtime": {"family": "mistral", "use_adapter": True},
    }
    assert_true(
        not completed_metadata_is_usable(old_retfact_metadata, fine_tuned_case, args),
        "RetFact-only retriever metadata must not resume as completed",
    )
    assert_true(
        not completed_run_metadata_is_usable(old_retfact_metadata, fine_tuned_case),
        "RetFact-only run metadata must not resume",
    )

    generator_metadata = {
        "status": "completed",
        "formal_result": True,
        "num_runs": 1,
        "top_k": 3,
        "runtime": {"engine": "mistral4", "prediction_contract": "data_binding_generator"},
    }
    assert_true(
        completed_metadata_is_usable(generator_metadata, generator_case, args),
        "generator data-binding metadata should remain resumable",
    )

    converted_retriever_metadata = {
        "status": "completed",
        "formal_result": True,
        "num_runs": 1,
        "top_k": 3,
        "binding_conversion": {"status": "completed"},
        "runtime": {
            "prediction_contract": "data_binding_generator",
            "binding_conversion": {"status": "completed"},
            "stages": [
                {"family": "mistral", "prediction_contract": "retfact_retriever", "use_adapter": True},
                {"engine": "gpt5_5", "prediction_contract": "data_binding_generator"},
            ],
        },
    }
    assert_true(
        completed_metadata_is_usable(converted_retriever_metadata, fine_tuned_case, args),
        "converted retriever data-binding metadata should be resumable",
    )
    assert_true(
        completed_run_metadata_is_usable(converted_retriever_metadata, fine_tuned_case),
        "converted retriever run metadata should be resumable",
    )

    try:
        require_binding_generation_contract(
            base_case,
            [{"case_id": "row_1", "items": []}],
            {"family": "mistral", "prediction_contract": "retfact_retriever", "use_adapter": False},
        )
    except MissingBindingGeneratorError as exc:
        assert_true(exc.failure_category == "runtime_blocked_missing_binding_generator", "blocker category")
    else:
        raise AssertionError("RetFact-only retriever output must be blocked")

    require_binding_generation_contract(
        generator_case,
        [{"case_id": "row_1", "items": []}],
        {"engine": "mistral4", "prediction_contract": "data_binding_generator"},
    )

    print(json.dumps({"status": "ok", "checks": "experiment6_routing_contract"}, sort_keys=True))


if __name__ == "__main__":
    main()
