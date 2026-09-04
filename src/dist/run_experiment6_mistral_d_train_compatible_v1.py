#!/usr/bin/env python3
"""Isolated rerun entry point for the existing fine-tuned Mistral dynamic adapter."""

from __future__ import annotations

import csv
import json
import importlib.util
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiment6_paths import PATHS
import run_experiment6_narrative2_generation as core


PROMPT_POLICY_PATH = PATHS.resolve(
    "repo", ".external/FINDER/Retriever Codes/Mistral/mistral_dynamic_adapter_prompt.py"
)
TRAINING_PROMPT_CORPUS_PATH = PATHS.resolve(
    "data", "finqa_dynamic_shot/finqa_train_rel_fact_instruction.csv"
)
RUNNER_BASE_PATH = Path(core.__file__).resolve()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_training_target_template(path: Path) -> tuple[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first_row = next(csv.DictReader(handle), None)
    if not first_row or not str(first_row.get("input") or "").strip():
        raise RuntimeError(f"training prompt corpus has no input row: {path}")
    prompt = str(first_row["input"]).replace("\r\n", "\n").replace("\r", "\n")
    context_separator = "\n\n## Context\n"
    output_separator = "\n\n## Output Format\n"
    example_separator = "\n\n### Example\n"
    if any(prompt.count(separator) != 1 for separator in (context_separator, output_separator)):
        raise RuntimeError(f"training prompt corpus has an ambiguous template: {path}")
    header, context_and_output = prompt.split(context_separator, 1)
    _, output_and_examples = context_and_output.split(output_separator, 1)
    if example_separator not in "\n\n" + output_and_examples:
        raise RuntimeError(f"dynamic training prompt has no Example boundary: {path}")
    output_body, _ = ("\n\n" + output_and_examples).split(example_separator, 1)
    output_section = "## Output Format\n" + output_body.strip()
    return header.strip(), output_section


TRAINING_TARGET_HEADER, TRAINING_TARGET_OUTPUT = load_training_target_template(
    TRAINING_PROMPT_CORPUS_PATH
)


def render_training_target_prompt(*, narrative: str, question: str) -> str:
    if not question.strip():
        raise core.ProtocolError("targetQuestion must be non-empty")
    context = f"## Context\n{str(narrative).strip()}; question:{question.strip()}"
    return "\n\n".join((TRAINING_TARGET_HEADER, context, TRAINING_TARGET_OUTPUT)) + "\n"


PROMPT_POLICY = load_module("mistral_dynamic_adapter_prompt", PROMPT_POLICY_PATH)
ORIGINAL_LOAD_CONFIG = core.load_config
ORIGINAL_READ_INPUT_ROWS = core.read_input_rows


def validate_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = config.get("mistralDynamicAdapterPrompt")
    if not isinstance(contract, Mapping):
        raise core.ProtocolError("config is missing mistralDynamicAdapterPrompt")
    if contract.get("policy") != PROMPT_POLICY.POLICY:
        raise core.ProtocolError("Mistral dynamic adapter prompt policy mismatch")
    if contract.get("targetPromptFormat") != "finqa-dynamic-training-corpus-v1":
        raise core.ProtocolError("Mistral target prompt format mismatch")
    if contract.get("targetQuestionPolicy") != "narrative-as-question-v1":
        raise core.ProtocolError("Mistral target question policy mismatch")
    if contract.get("targetContextPolicy") != "narrative-only-v1":
        raise core.ProtocolError("Mistral target context policy mismatch")
    if contract.get("placementPolicy") != "target-last-full-nearest-v1":
        raise core.ProtocolError("Mistral prompt placement policy mismatch")
    if contract.get("exampleFormatPolicy") != "compact-retfact-json-v1":
        raise core.ProtocolError("Mistral example format policy mismatch")
    if int(contract.get("completionReserveTokens", 0)) != int(
        config["retriever"]["maxNewTokens"]
    ):
        raise core.ProtocolError(
            "completionReserveTokens must equal retriever.maxNewTokens"
        )
    if int(contract.get("trainMaxSequenceTokens", 0)) <= 0:
        raise core.ProtocolError("trainMaxSequenceTokens must be positive")
    if int(config["retriever"]["maxInputTokens"]) < int(
        contract["trainMaxSequenceTokens"]
    ):
        raise core.ProtocolError(
            "retriever.maxInputTokens cannot be shorter than the adapter training window"
        )

    cases = core.expand_matrix(config)
    if len(cases) != 1:
        raise core.ProtocolError("Mistral-d successor config must contain exactly one case")
    case = cases[0]
    if (
        case.output_id != "6_mistral_d"
        or case.source_id != "finqa_mistral_d"
        or case.prompt_mode != "dynamic-shot"
        or case.route != "adapter-converter"
    ):
        raise core.ProtocolError("successor config must isolate fine-tuned 6_mistral_d")

    files = {
        "wrapper": Path(__file__).resolve(),
        "promptPolicy": PROMPT_POLICY_PATH,
        "trainingPromptCorpus": TRAINING_PROMPT_CORPUS_PATH,
        "runnerBase": RUNNER_BASE_PATH,
    }
    for name, path in files.items():
        specification = contract.get(name)
        if not isinstance(specification, Mapping):
            raise core.ProtocolError(f"prompt contract is missing {name}")
        actual_sha = core.sha256_file(path)
        if actual_sha != specification.get("sha256"):
            raise core.ProtocolError(
                f"{name} SHA-256 mismatch: {actual_sha} != {specification.get('sha256')}"
            )
    return contract


def load_rendered_examples(path: Path) -> list[str]:
    examples = core.load_examples(path)
    rendered = [
        "Result: "
        + json.dumps(
            {"RetFact": example.retfact},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for example in examples
    ]
    if not rendered or any(not value.strip() for value in rendered):
        raise core.ProtocolError(f"compact RetFact examples are empty: {path}")
    return rendered


def transform_rows(
    rows: Sequence[core.InputRow],
    config: Mapping[str, Any],
    *,
    tokenizer: Any,
    rendered_examples: Sequence[str],
) -> tuple[list[core.InputRow], dict[str, Any]]:
    contract = validate_contract(config)
    measurements: list[dict[str, Any]] = []
    transformed_rows: list[core.InputRow] = []
    for row in rows:
        target_prompt = render_training_target_prompt(
            narrative=row.text,
            question=row.text,
        )
        result = PROMPT_POLICY.build_training_compatible_prompt(
            prompt=row.retriever_prompts["dynamic-shot"],
            rendered_examples=rendered_examples,
            selected_indices=row.shot_ids["dynamic-shot"],
            tokenizer=tokenizer,
            label_marker=str(contract["labelMarker"]),
            train_max_seq_length=int(contract["trainMaxSequenceTokens"]),
            completion_reserve_tokens=int(contract["completionReserveTokens"]),
            preserve_target_over_budget=bool(contract["preserveTargetOverBudget"]),
            reverse_selected=bool(contract["reverseSelectedExamples"]),
            target_prompt_override=target_prompt,
            placement_policy=str(contract["placementPolicy"]),
        )
        effective_prompt = str(result["prompt"])
        audit = dict(result["audit"])
        if "[EXAMPLE" in effective_prompt:
            raise core.ProtocolError(f"bare example marker survived for {row.source}")
        if "## Binding coordinate contract" in effective_prompt:
            raise core.ProtocolError(
                f"Experiment 6 output contract survived for {row.source}"
            )
        if not effective_prompt.endswith(target_prompt):
            raise core.ProtocolError(f"target-last boundary failed for {row.source}")
        if int(audit["effectivePromptWithMarkerTokens"]) > int(
            config["retriever"]["maxInputTokens"]
        ):
            raise core.ProtocolError(
                f"effective Mistral prompt exceeds inference input limit: {row.source}"
            )
        if (
            int(audit["effectivePromptWithMarkerTokens"])
            + int(config["retriever"]["maxNewTokens"])
            > int(contract["contextWindowTokens"])
        ):
            raise core.ProtocolError(
                f"effective Mistral prompt exceeds prompt+completion context: {row.source}"
            )

        retriever_prompts = dict(row.retriever_prompts)
        retriever_prompts["dynamic-shot"] = effective_prompt
        transformed_rows.append(
            replace(row, retriever_prompts=retriever_prompts)
        )
        measurements.append({
            "source": row.source,
            "effectivePromptSha256": core.sha256_text(effective_prompt),
            **audit,
        })

    dispositions = Counter(
        str(measurement["disposition"]) for measurement in measurements
    )
    return transformed_rows, {
        "policy": str(contract["policy"]),
        "targetPromptFormat": str(contract["targetPromptFormat"]),
        "targetContextPolicy": str(contract["targetContextPolicy"]),
        "targetQuestionPolicy": str(contract["targetQuestionPolicy"]),
        "placementPolicy": str(contract["placementPolicy"]),
        "exampleFormatPolicy": str(contract["exampleFormatPolicy"]),
        "labelMarker": str(contract["labelMarker"]),
        "trainMaxSequenceTokens": int(contract["trainMaxSequenceTokens"]),
        "completionReserveTokens": int(contract["completionReserveTokens"]),
        "inferenceMaxInputTokens": int(config["retriever"]["maxInputTokens"]),
        "contextWindowTokens": int(contract["contextWindowTokens"]),
        "preserveTargetOverBudget": bool(contract["preserveTargetOverBudget"]),
        "reverseSelectedExamples": bool(contract["reverseSelectedExamples"]),
        "rows": len(measurements),
        "dispositions": dict(sorted(dispositions.items())),
        "maxEffectivePromptWithMarkerTokens": max(
            int(item["effectivePromptWithMarkerTokens"])
            for item in measurements
        ),
        "maxPromptPlusCompletionTokens": max(
            int(item["effectivePromptWithMarkerTokens"])
            + int(config["retriever"]["maxNewTokens"])
            for item in measurements
        ),
        "bareExampleMarkerRows": sum(
            "[EXAMPLE" in row.retriever_prompts["dynamic-shot"]
            for row in transformed_rows
        ),
        "measurements": measurements,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = ORIGINAL_LOAD_CONFIG(path)
    validate_contract(config)
    return config


def read_input_rows(
    config: Mapping[str, Any],
    limit: int,
    row_source: str | None = None,
) -> tuple[list[core.InputRow], dict[str, Any]]:
    rows, input_report = ORIGINAL_READ_INPUT_ROWS(config, limit, row_source)
    contract = validate_contract(config)
    tokenizer = AutoTokenizer.from_pretrained(
        str(contract["baseModel"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    rendered_examples = load_rendered_examples(core.workspace_path(config["exampleCsv"]))
    transformed, prompt_report = transform_rows(
        rows,
        config,
        tokenizer=tokenizer,
        rendered_examples=rendered_examples,
    )
    input_report = dict(input_report)
    input_report["mistralDynamicAdapterPrompt"] = prompt_report
    return transformed, input_report


core.load_config = load_config
core.read_input_rows = read_input_rows


if __name__ == "__main__":
    core.main()
