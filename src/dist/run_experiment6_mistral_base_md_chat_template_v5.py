#!/usr/bin/env python3
"""Target-boundary Mistral chat entry point for the isolated base m/d diagnostic."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from experiment6_paths import PATHS
import run_experiment6_mistral_base_md_chat_template_v2 as base


INFERENCE_SHIM = PATHS.resolve(
    "repo", ".external/FINDER/Retriever Codes/Mistral/mistral_direct_binding_chat_inference_v4.py"
)
INFERENCE_BASE = PATHS.resolve(
    "repo", ".external/FINDER/Retriever Codes/Mistral/mistral_direct_binding_chat_inference.py"
)
WRAPPER_BASE = PATHS.resolve(
    "dist", "run_experiment6_mistral_base_md_chat_template_v2.py"
)
base.INFERENCE_SCRIPT = INFERENCE_SHIM


def load_prompt_policy() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mistral_target_boundary_policy", INFERENCE_SHIM
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {INFERENCE_SHIM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPT_POLICY = load_prompt_policy()


def token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise base.core.ProtocolError("native chat template did not return token IDs")
    return value


def validate_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = config.get("mistralDirectChatTemplate")
    if not isinstance(contract, Mapping):
        raise RuntimeError("config is missing mistralDirectChatTemplate")
    if contract.get("policy") != PROMPT_POLICY.POLICY:
        raise RuntimeError("target-last prompt policy mismatch")
    files = {
        "wrapper": Path(__file__).resolve(),
        "wrapperBase": WRAPPER_BASE,
        "inference": INFERENCE_SHIM,
        "inferenceBase": INFERENCE_BASE,
    }
    for name, path in files.items():
        file_spec = contract.get(name)
        if not isinstance(file_spec, Mapping):
            raise RuntimeError(f"chat-template contract missing {name}")
        actual = base.file_sha256(path)
        if actual != file_spec.get("sha256"):
            raise RuntimeError(
                f"{name} SHA-256 mismatch: {actual} != {file_spec.get('sha256')}"
            )
    return contract


def install_token_preflight(contract: Mapping[str, Any]) -> None:
    def preflight(
        rows: Sequence[Any],
        cases: Sequence[Any],
        config: Mapping[str, Any],
        base_route_mode: str = "historical",
    ) -> dict[str, Any]:
        report = base.ORIGINAL_TOKEN_PREFLIGHT(rows, cases, config, base_route_mode)
        selected = [
            case for case in cases
            if case.source_id == "mistral_v0_3"
            and case.route == "direct-binding"
            and case.prompt_mode in {"many-shot", "dynamic-shot"}
        ]
        if not selected:
            return report
        tokenizer = AutoTokenizer.from_pretrained(
            str(contract["baseModel"]), local_files_only=True, trust_remote_code=True
        )
        template = str(tokenizer.chat_template or "")
        template_sha = base.core.sha256_text(template)
        if template_sha != contract["chatTemplateSha256"]:
            raise base.core.ProtocolError("native chat-template SHA-256 mismatch")
        if len(template.encode("utf-8")) != int(contract["chatTemplateBytes"]):
            raise base.core.ProtocolError("native chat-template byte count mismatch")
        maximum = int(config["directBinding"]["maxInputTokens"])
        completion = int(config["directBinding"]["maxNewTokens"])
        context = int(contract["contextWindow"])
        measurements: list[dict[str, Any]] = []
        for case in selected:
            for row in rows:
                original = row.direct_prompts[case.prompt_mode]
                transformed = PROMPT_POLICY.target_last_prompt(original)
                ids = token_ids(tokenizer.apply_chat_template(
                    [{"role": "user", "content": transformed}],
                    tokenize=True,
                    add_generation_prompt=True,
                ))
                item = {
                    "outputId": case.output_id,
                    "source": row.source,
                    "tokens": len(ids),
                    "originalPromptSha256": base.core.sha256_text(original),
                    "targetLastPromptSha256": base.core.sha256_text(transformed),
                    "inputIdsSha256": base.core.sha256_text(
                        json.dumps(ids, separators=(",", ":"))
                    ),
                }
                measurements.append(item)
                if len(ids) > maximum or len(ids) + completion > context:
                    raise base.core.ProtocolError(
                        f"target-last chat prompt exceeds token gate: {item}"
                    )
        report["mistralNativeChatTemplate"] = {
            "policy": contract["policy"],
            "chatTemplateBytes": len(template.encode("utf-8")),
            "chatTemplateSha256": template_sha,
            "maxInputAllowed": maximum,
            "maxNewTokens": completion,
            "contextWindow": context,
            "minObserved": min(item["tokens"] for item in measurements),
            "maxObserved": max(item["tokens"] for item in measurements),
            "measurements": len(measurements),
            "truncationAllowed": False,
            "inputIdentitySha256": base.core.sha256_text(
                json.dumps(measurements, sort_keys=True, separators=(",", ":"))
            ),
        }
        return report

    base.core.token_preflight = preflight


base.validate_contract = validate_contract
base.install_token_preflight = install_token_preflight


if __name__ == "__main__":
    base.main()
