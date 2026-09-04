"""Versioned FinFlier prompt materialization for Experiment 6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class FinFlierPromptError(RuntimeError):
    """Raised when a FinFlier prompt asset violates its frozen contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


@dataclass(frozen=True)
class FinFlierPromptAsset:
    path: Path
    file_sha256: str
    policy_version: str
    source: Mapping[str, Any]
    general_example_count: int
    default_prompt: str
    default_prompt_sha256: str
    dispatch_order: tuple[str, ...]
    special_prompts: Mapping[str, Mapping[str, str]]


def load_prompt_asset(path: Path, expected_sha256: str) -> FinFlierPromptAsset:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FinFlierPromptError(f"FinFlier prompt asset is missing: {resolved}")
    actual_sha256 = sha256_bytes(resolved.read_bytes())
    if actual_sha256 != expected_sha256:
        raise FinFlierPromptError(
            f"FinFlier prompt asset SHA-256 mismatch: {actual_sha256}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "schemaVersion",
        "promptPolicyVersion",
        "source",
        "generalExampleCount",
        "defaultPrompt",
        "defaultPromptSha256",
        "dispatchOrder",
        "specialPrompts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise FinFlierPromptError("FinFlier prompt asset has an invalid envelope")
    if payload["schemaVersion"] != 1 or payload["generalExampleCount"] != 10:
        raise FinFlierPromptError("FinFlier prompt asset version/example count mismatch")
    default_prompt = payload["defaultPrompt"]
    if not isinstance(default_prompt, str) or sha256_text(default_prompt) != payload["defaultPromptSha256"]:
        raise FinFlierPromptError("FinFlier default prompt hash mismatch")
    if default_prompt.count("\nresult:") != payload["generalExampleCount"]:
        raise FinFlierPromptError("FinFlier default prompt example count mismatch")
    dispatch_order = payload["dispatchOrder"]
    special_prompts = payload["specialPrompts"]
    if not isinstance(dispatch_order, list) or not isinstance(special_prompts, dict):
        raise FinFlierPromptError("FinFlier special prompt contract is invalid")
    for phrase in dispatch_order:
        record = special_prompts.get(phrase)
        if not isinstance(phrase, str) or not isinstance(record, dict):
            raise FinFlierPromptError("FinFlier dispatch entry is invalid")
        value = record.get("text")
        if not isinstance(value, str) or sha256_text(value) != record.get("sha256"):
            raise FinFlierPromptError(f"FinFlier special prompt hash mismatch: {phrase}")
    return FinFlierPromptAsset(
        path=resolved,
        file_sha256=actual_sha256,
        policy_version=str(payload["promptPolicyVersion"]),
        source=payload["source"],
        general_example_count=int(payload["generalExampleCount"]),
        default_prompt=default_prompt,
        default_prompt_sha256=str(payload["defaultPromptSha256"]),
        dispatch_order=tuple(dispatch_order),
        special_prompts=special_prompts,
    )


def select_special_prompt(
    asset: FinFlierPromptAsset,
    chart_data: str,
    narrative: str,
) -> tuple[str | None, str]:
    query_text = f"{chart_data}\n{narrative}".lower()
    for phrase in asset.dispatch_order:
        if phrase in query_text:
            return phrase, str(asset.special_prompts[phrase]["text"])
    return None, ""


def build_prompt(
    asset: FinFlierPromptAsset,
    *,
    chart_data: str,
    narrative: str,
    coordinate_contract: str,
    output_contract: str,
) -> tuple[str, dict[str, Any]]:
    special_id, special_text = select_special_prompt(asset, chart_data, narrative)
    query = f"{chart_data}\ntext: {json.dumps([narrative], ensure_ascii=False)}"
    sections = [asset.default_prompt]
    if special_text:
        sections.append(special_text)
    sections.append(query)
    if coordinate_contract.strip():
        sections.append("## Binding coordinate contract\n" + coordinate_contract)
    sections.append("## Output contract\n" + output_contract)
    prompt = "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"
    special_sha256 = (
        str(asset.special_prompts[special_id]["sha256"])
        if special_id is not None
        else None
    )
    return prompt, {
        "finflierPromptApplied": True,
        "promptPolicyVersion": asset.policy_version,
        "assetSha256": asset.file_sha256,
        "source": dict(asset.source),
        "generalExampleCount": asset.general_example_count,
        "defaultPromptSha256": asset.default_prompt_sha256,
        "specialExampleId": special_id,
        "specialExampleSha256": special_sha256,
        "querySha256": sha256_text(query),
        "finalPromptSha256": sha256_text(prompt),
    }
