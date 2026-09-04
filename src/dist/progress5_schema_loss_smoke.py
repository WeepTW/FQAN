#!/usr/bin/env python3
"""CPU smoke checks for Progress 5 JSON target and RetFact-only masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retriever_json_schema import (  # noqa: E402
    label_for_prompt_mode,
    retfact_label_for_training,
    schema_and_retfact_char_masks,
    schema_and_retfact_token_masks,
    schema_parse_gate,
    schema_required,
)
from retriever_lmfe import lmfe_import_status  # noqa: E402


class CharTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        value = str(text)
        payload = {"input_ids": [ord(char) % 1000 + 1 for char in value]}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(value))]
        return payload


def main() -> None:
    retfact = "revenue increased; gross margin improved"
    target = label_for_prompt_mode(retfact, "zero-shot")
    assert retfact_label_for_training(retfact, "zero-shot") == target

    schema_chars, retfact_chars = schema_and_retfact_char_masks(target)
    assert any(schema_chars), "schema char mask is empty"
    assert any(retfact_chars), "RetFact char mask is empty"

    tokenizer = CharTokenizer()
    schema_tokens, retfact_tokens = schema_and_retfact_token_masks(tokenizer, target)
    assert any(schema_tokens), "schema token mask is empty"
    assert any(retfact_tokens), "RetFact token mask is empty"

    assert schema_parse_gate(target), "valid target should pass h gate"
    assert not schema_parse_gate('{"RetFact":"","Binding":[],"Reason":""}'), "empty RetFact must fail h gate"
    assert not schema_parse_gate("Binding text without JSON"), "malformed output must fail h gate"

    assert not schema_required("original"), "original prompt must not require schema loss"
    assert label_for_prompt_mode(retfact, "original") == retfact, "original label must remain RetFact text"
    assert retfact_label_for_training(retfact, "original") == retfact

    lmfe_status = lmfe_import_status()
    assert lmfe_status["available"], lmfe_status

    print(
        json.dumps(
            {
                "status": "ok",
                "schema_token_count": sum(schema_tokens),
                "retfact_token_count": sum(retfact_tokens),
                "non_retfact_target_tokens_ignored": len(retfact_tokens) - sum(retfact_tokens),
                "lmfe_import_status": lmfe_status,
                "original_schema_required": schema_required("original"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
