"""LMFE compatibility helpers for retriever JSON generation."""

from __future__ import annotations

import os
from typing import Any

import torch

from retriever_json_schema import RETRIEVER_OUTPUT_JSON_SCHEMA, RETRIEVER_RETFACT_GENERATION_MAX_CHARS


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


RETRIEVER_RETFACT_REGEX_MIN_CHARS = _positive_int_env("RETRIEVER_RETFACT_REGEX_MIN_CHARS", 1)
RETRIEVER_RETFACT_REGEX_MAX_CHARS = _positive_int_env("RETRIEVER_RETFACT_REGEX_MAX_CHARS", 128)
if RETRIEVER_RETFACT_REGEX_MAX_CHARS < RETRIEVER_RETFACT_REGEX_MIN_CHARS:
    raise ValueError(
        "RETRIEVER_RETFACT_REGEX_MAX_CHARS must be greater than or equal to "
        "RETRIEVER_RETFACT_REGEX_MIN_CHARS"
    )

RETRIEVER_OUTPUT_REGEX = (
    rf'\{{"RetFact":"[^"]{{{RETRIEVER_RETFACT_REGEX_MIN_CHARS},{RETRIEVER_RETFACT_REGEX_MAX_CHARS}}}",'
    r'"Binding":\[\{"ObjectName":\[\],"DataName":"",'
    r'"Position":\[\{"Begin":\[\],"End":\[\]\}\],'
    r'"Trend":"None","Num":\[\],"Text":""\}\],'
    r'"Reason":""\}'
)


def _patch_transformers_tokenizer_base_import() -> None:
    """Keep lm-format-enforcer 0.11.x usable with Transformers 5.x.

    LMFE imports PreTrainedTokenizerBase from transformers.tokenization_utils,
    while Transformers 5 exposes it from tokenization_utils_base.  This shim is
    repo-local and does not modify installed packages on disk.
    """

    import transformers.tokenization_utils as tokenization_utils
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase


def retfact_json_schema_parser():
    from lmformatenforcer import JsonSchemaParser

    return JsonSchemaParser(RETRIEVER_OUTPUT_JSON_SCHEMA)


def retfact_json_generation_parser():
    parser_kind = os.environ.get("RETRIEVER_LMFE_PARSER", "json_schema").strip().lower()
    if parser_kind in {"json", "json_schema", "schema"}:
        return retfact_json_schema_parser()
    if parser_kind not in {"regex", "fixed_regex"}:
        raise ValueError(
            "RETRIEVER_LMFE_PARSER must be 'json_schema' or 'regex', "
            f"got {parser_kind!r}"
        )

    from lmformatenforcer import RegexParser
    return RegexParser(RETRIEVER_OUTPUT_REGEX)


def build_retfact_prefix_allowed_tokens_fn(tokenizer: Any, strip_leading_special: bool = False):
    _patch_transformers_tokenizer_base_import()
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, retfact_json_generation_parser())

    try:
        initial_allowed = prefix_fn(0, torch.empty(0, dtype=torch.long))
    except Exception as exc:
        raise RuntimeError(f"LMFE prefix function failed during initialization: {exc}") from exc
    if not initial_allowed:
        raise RuntimeError(
            "LMFE cannot constrain this tokenizer to the JSON skeleton. "
            "The tokenizer likely cannot emit required JSON punctuation such as '{'. "
            "Use a tokenizer/model with JSON punctuation support for formal non-original inference; "
            "wrapper mode is debug-only."
        )
    if not strip_leading_special:
        return prefix_fn

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    def wrapped_prefix_allowed_tokens_fn(batch_id: int, sent):
        if len(sent) > 0 and int(sent[0]) in special_ids:
            sent = sent[1:]
        return prefix_fn(batch_id, sent)

    return wrapped_prefix_allowed_tokens_fn


def lmfe_import_status() -> dict[str, Any]:
    try:
        _patch_transformers_tokenizer_base_import()
        from lmformatenforcer.integrations.transformers import (  # noqa: F401
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = retfact_json_generation_parser()
        return {
            "available": True,
            "parser": type(parser).__name__,
            "constraint": os.environ.get("RETRIEVER_LMFE_PARSER", "json_schema"),
            "retfact_generation_max_chars": RETRIEVER_RETFACT_GENERATION_MAX_CHARS,
            "retfact_min_chars": RETRIEVER_RETFACT_REGEX_MIN_CHARS,
            "retfact_max_chars": RETRIEVER_RETFACT_REGEX_MAX_CHARS,
            "binding_max_items": 10,
            "transformers_compat_shim": True,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "transformers_compat_shim": True,
        }
